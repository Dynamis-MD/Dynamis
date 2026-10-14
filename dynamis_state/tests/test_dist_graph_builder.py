"""Correctness tests for dist_graph_builder.

Verifies v5's per-rank PartitionMessagePassingInfo matches both (a) a single-GPU
oracle that mirrors `main::dynamis_state/src/partition/build_halo_maps.cpp`, and
(b) the *actual* C++ `AbstractPartitioner::build_halo_maps` bit-exactly
(test_v5_halo_map_vs_cpp_build_halo_maps).

Run:
    pytest dynamis_state/tests/test_dist_v5_correctness.py

The multi-rank tests use torchrun launched internally via subprocess.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

import dynamis_state.dynamis_state_impl as _impl
from dynamis_state.dist_graph_builder import (
    DistGraphBuilder,
    BuildMode,
    _extract_halo_info_local,
)

# ─── ground-truth oracle (mirrors main/build_halo_maps.cpp) ────────────────


def oracle_build_halo_maps(
    pos: torch.Tensor,
    partition_id: torch.Tensor,  # [N] int32/int64
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff_plus_skin: float,
    rank: int,
    world_size: int,
):
    """Pure-Python single-GPU reference, matching `main::build_halo_maps`.

    Returns dict with send_offsets, send_local_indices, recv_offsets,
    recv_ghost_indices, interior_mask, edges_for_this_rank.
    """
    device = pos.device
    N = pos.shape[0]
    W = world_size
    R = rank

    pid = partition_id.to(torch.int64)

    # Full graph at r+skin (replicated)
    edges, cell_shifts, _, _ = _impl.radius_graph_pbc_ex(
        pos,
        cell,
        pbc,
        cutoff_plus_skin,
        1000,
        False,
        None,
    )
    src = edges[0]
    dst = edges[1]

    # needed_by[N, W]
    needed_by = torch.zeros((N, W), dtype=torch.bool, device=device)
    needed_by[torch.arange(N, device=device), pid] = True  # self
    owner_dst = pid.index_select(0, dst)
    needed_by[src, owner_dst] = True

    # my atoms
    my_indices = torch.nonzero(pid.eq(R), as_tuple=False).squeeze(-1)
    num_local = my_indices.numel()

    # SEND: my atoms needed by other ranks
    my_needed = needed_by.index_select(0, my_indices).clone()  # [num_local, W]
    my_needed[:, R] = False  # zero self col
    send_pairs = torch.nonzero(
        my_needed, as_tuple=False
    )  # [E_send, 2]: (local_pos, peer)
    if send_pairs.numel() > 0:
        send_atom_global = my_indices[send_pairs[:, 0]]
        send_peer = send_pairs[:, 1]
        # Sort by (peer, atom_global) for deterministic CSR + Fix-1-equivalent order
        sort_key = send_peer * N + send_atom_global
        order = torch.argsort(sort_key, stable=True)
        send_atom_global = send_atom_global[order]
        send_peer = send_peer[order]
        # Convert back to local positions
        send_local_indices = send_pairs[order, 0]
    else:
        send_atom_global = torch.empty(0, dtype=torch.int64, device=device)
        send_peer = torch.empty(0, dtype=torch.int64, device=device)
        send_local_indices = torch.empty(0, dtype=torch.int64, device=device)

    send_counts = torch.zeros(W, dtype=torch.int64, device=device)
    if send_peer.numel() > 0:
        send_counts.scatter_add_(0, send_peer, torch.ones_like(send_peer))
    send_offsets = torch.zeros(W + 1, dtype=torch.int64, device=device)
    send_offsets[1:] = torch.cumsum(send_counts, 0)

    # RECV: atoms I need but don't own
    i_need = needed_by[:, R].clone()
    i_need[my_indices] = False
    recv_global = torch.nonzero(i_need, as_tuple=False).squeeze(-1)  # global atom IDs
    recv_peer = pid.index_select(0, recv_global)
    if recv_global.numel() > 0:
        sort_key = recv_peer * N + recv_global
        order = torch.argsort(sort_key, stable=True)
        recv_global = recv_global[order]
        recv_peer = recv_peer[order]

    recv_counts = torch.zeros(W, dtype=torch.int64, device=device)
    if recv_peer.numel() > 0:
        recv_counts.scatter_add_(0, recv_peer, torch.ones_like(recv_peer))
    recv_offsets = torch.zeros(W + 1, dtype=torch.int64, device=device)
    recv_offsets[1:] = torch.cumsum(recv_counts, 0)
    total_recv = recv_global.numel()
    recv_ghost_indices = torch.arange(
        num_local,
        num_local + total_recv,
        dtype=torch.int64,
        device=device,
    )

    # interior_mask: my atoms with no remote-src edge
    dst_is_mine = pid.index_select(0, dst).eq(R)
    src_of_my = src[dst_is_mine]
    dst_of_my = dst[dst_is_mine]
    src_is_ghost = pid.index_select(0, src_of_my).ne(R)
    has_ghost = torch.zeros(N, dtype=torch.bool, device=device)
    boundary = dst_of_my[src_is_ghost]
    if boundary.numel() > 0:
        has_ghost.scatter_(0, boundary, True)
    interior_mask = ~has_ghost.index_select(0, my_indices)

    # Edges with dst ∈ my_outer (the slice v5 produces)
    dst_mine_mask = pid.index_select(0, dst).eq(R)
    edges_mine = edges[:, dst_mine_mask]
    shifts_mine = cell_shifts[dst_mine_mask]

    return dict(
        send_offsets=send_offsets,
        send_local_indices=send_local_indices,
        send_atom_global=send_atom_global,  # for cross-validation
        send_peer=send_peer,
        recv_offsets=recv_offsets,
        recv_ghost_indices=recv_ghost_indices,
        recv_atom_global=recv_global,  # for cross-validation
        recv_peer=recv_peer,
        interior_mask=interior_mask,
        my_indices=my_indices,
        edges_mine=edges_mine,
        shifts_mine=shifts_mine,
    )


# ─── single-rank smoke tests ────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_v5_single_rank_smoke():
    """W=1: all atoms are mine; halo is empty; interior_mask all True."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    N = 200
    L = 30.0
    pos = torch.rand(N, 3, device=device) * L
    cell = torch.eye(3, device=device, dtype=torch.float32) * L
    pbc = torch.tensor([True, True, True])

    env = _impl.DistEnvironment()
    env.global_rank = 0
    env.world_size = 1
    env.local_rank = 0
    env.gpus_per_node = 1
    env.num_nodes = 1

    builder = DistGraphBuilder(
        cell,
        pbc,
        cutoff=4.0,
        ghost_buffer=0.5,
        env=env,
        refresh_interval=100,
    )
    res = builder.build(pos)

    assert res.mode == BuildMode.REFRESH
    assert res.my_outer_atoms.numel() == N
    assert res.halo_info.send_offsets[-1].item() == 0
    assert res.halo_info.recv_offsets[-1].item() == 0
    assert bool(res.interior_mask.all().item())
    assert res.halo_recv_global_ids.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("W", [2, 4])
def test_v5_halo_map_vs_cpp_build_halo_maps(W):
    """v5's `extract_halo_info` must match the production C++
    `AbstractPartitioner::build_halo_maps` bit-exactly, for the same partition.

    Single-process check (no torchrun): for each rank R of a W-way partition,
    build the halo map both ways on one GPU and compare the send/recv CSR +
    interior mask. This pins v5 to the *actual* serial reference, not just the
    Python oracle (which could drift from build_halo_maps.cpp).
    """
    torch.manual_seed(13)
    device = torch.device("cuda:0")
    N, L = 4000, 40.0
    cutoff, ghost = 6.0, 0.5
    r_skin = cutoff + ghost
    pos = (torch.rand(N, 3, device=device) * L).contiguous()
    cell = torch.eye(3, device=device, dtype=torch.float32) * L
    pbc = torch.tensor([True, True, True])
    # Spatial split by x → real cross-rank boundary edges. The halo logic is
    # partition-agnostic, so any partition_id exercises it.
    pid = torch.clamp((pos[:, 0] / L * W).floor().long(), 0, W - 1).contiguous()

    total_send = 0
    for R in range(W):
        # --- serial C++ build_halo_maps ---
        mol = _impl.MolecularInfo()
        mol.num_atoms = N
        mol.atom_positions = pos
        env = _impl.DistEnvironment()
        env.global_rank = R
        env.world_size = W
        env.local_rank = 0
        env.gpus_per_node = W
        env.num_nodes = 1
        cfg = _impl.SimulationConfig()
        cfg.cutoff = cutoff
        cfg.ghost_cutoff_buffer = ghost
        cfg.cell = cell
        cfg.pbc = pbc
        info = _impl.PartitionMessagePassingInfo()
        info.partition_id = pid
        info.global_send_recv_matrix = torch.zeros(
            W, W, dtype=torch.int64, device=device
        )
        _impl.SimpleSpatialPartitioner().build_halo_maps(info, mol, env, cfg)

        # --- v5 extract_halo_info on the same partition (edges at r+skin) ---
        my_outer = torch.nonzero(pid.eq(R), as_tuple=False).squeeze(-1)
        edges, _, _, _ = _impl.radius_graph_pbc_ex(
            pos, cell, pbc, r_skin, 0, False, my_outer
        )
        (
            send_offsets,
            send_local_indices,
            recv_offsets,
            recv_ghost_indices,
            interior_mask,
            _,
        ) = _impl.extract_halo_info(edges, pid, my_outer, R, W, N)

        assert torch.equal(info.send_offsets, send_offsets), f"send_offsets rank {R}"
        assert torch.equal(
            info.send_local_indices, send_local_indices
        ), f"send_local_indices rank {R}"
        assert torch.equal(info.recv_offsets, recv_offsets), f"recv_offsets rank {R}"
        assert torch.equal(
            info.recv_ghost_indices, recv_ghost_indices
        ), f"recv_ghost_indices rank {R}"
        assert torch.equal(info.interior_mask, interior_mask), f"interior_mask rank {R}"
        total_send += int(send_local_indices.numel())

    # Guard against a trivially-empty halo (would make the comparison vacuous).
    assert total_send > 0, "expected a non-empty halo for this partition"


# ─── multi-rank correctness via torchrun subprocess ────────────────────────


def _multirank_runner_script() -> str:
    """The body of the multi-rank test run as a subprocess by torchrun."""
    return r"""
import os, sys, json
import torch
import torch.distributed as dist
import dynamis_state.dynamis_state_impl as _impl
from dynamis_state.dist_graph_builder import DistGraphBuilder, BuildMode


def oracle_build_halo_maps(pos, partition_id, cell, pbc, cutoff_plus_skin, rank, world_size):
    device = pos.device
    N = pos.shape[0]
    W = world_size
    R = rank
    pid = partition_id.to(torch.int64)
    edges, cell_shifts, _, _ = _impl.radius_graph_pbc_ex(
        pos, cell, pbc, cutoff_plus_skin, 1000, False, None,
    )
    src, dst = edges[0], edges[1]
    needed_by = torch.zeros((N, W), dtype=torch.bool, device=device)
    needed_by[torch.arange(N, device=device), pid] = True
    needed_by[src, pid.index_select(0, dst)] = True
    my_indices = torch.nonzero(pid.eq(R), as_tuple=False).squeeze(-1)
    num_local = my_indices.numel()
    my_needed = needed_by.index_select(0, my_indices).clone()
    my_needed[:, R] = False
    send_pairs = torch.nonzero(my_needed, as_tuple=False)
    if send_pairs.numel() > 0:
        send_atom_global = my_indices[send_pairs[:, 0]]
        send_peer = send_pairs[:, 1]
        sort_key = send_peer * N + send_atom_global
        order = torch.argsort(sort_key, stable=True)
        send_atom_global = send_atom_global[order]
        send_peer = send_peer[order]
        send_local_indices = send_pairs[order, 0]
    else:
        send_atom_global = torch.empty(0, dtype=torch.int64, device=device)
        send_peer = torch.empty(0, dtype=torch.int64, device=device)
        send_local_indices = torch.empty(0, dtype=torch.int64, device=device)
    send_counts = torch.zeros(W, dtype=torch.int64, device=device)
    if send_peer.numel() > 0:
        send_counts.scatter_add_(0, send_peer, torch.ones_like(send_peer))
    send_offsets = torch.zeros(W + 1, dtype=torch.int64, device=device)
    send_offsets[1:] = torch.cumsum(send_counts, 0)
    i_need = needed_by[:, R].clone()
    i_need[my_indices] = False
    recv_global = torch.nonzero(i_need, as_tuple=False).squeeze(-1)
    recv_peer = pid.index_select(0, recv_global)
    if recv_global.numel() > 0:
        sort_key = recv_peer * N + recv_global
        order = torch.argsort(sort_key, stable=True)
        recv_global = recv_global[order]
        recv_peer = recv_peer[order]
    recv_counts = torch.zeros(W, dtype=torch.int64, device=device)
    if recv_peer.numel() > 0:
        recv_counts.scatter_add_(0, recv_peer, torch.ones_like(recv_peer))
    recv_offsets = torch.zeros(W + 1, dtype=torch.int64, device=device)
    recv_offsets[1:] = torch.cumsum(recv_counts, 0)
    recv_ghost_indices = torch.arange(num_local, num_local + recv_global.numel(),
                                       dtype=torch.int64, device=device)
    dst_is_mine = pid.index_select(0, dst).eq(R)
    src_of_my = src[dst_is_mine]
    dst_of_my = dst[dst_is_mine]
    src_is_ghost = pid.index_select(0, src_of_my).ne(R)
    has_ghost = torch.zeros(N, dtype=torch.bool, device=device)
    boundary = dst_of_my[src_is_ghost]
    if boundary.numel() > 0:
        has_ghost.scatter_(0, boundary, True)
    interior_mask = ~has_ghost.index_select(0, my_indices)
    return dict(
        send_offsets=send_offsets, send_local_indices=send_local_indices,
        send_atom_global=send_atom_global, send_peer=send_peer,
        recv_offsets=recv_offsets, recv_ghost_indices=recv_ghost_indices,
        recv_atom_global=recv_global, recv_peer=recv_peer,
        interior_mask=interior_mask, my_indices=my_indices,
    )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.manual_seed(12345)
    N = 500
    L = 30.0
    pos = torch.rand(N, 3, device=device) * L
    cell = torch.eye(3, device=device, dtype=torch.float32) * L
    pbc = torch.tensor([True, True, True])

    env = _impl.DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.local_rank = local_rank
    env.gpus_per_node = world_size
    env.num_nodes = 1

    builder = DistGraphBuilder(
        cell, pbc, cutoff=4.0, ghost_buffer=0.5, env=env,
        refresh_interval=100,
    )
    res = builder.build(pos)

    # Compare with oracle
    pid_outer = res.pid_outer
    oracle = oracle_build_halo_maps(
        pos, pid_outer, cell, pbc,
        cutoff_plus_skin=4.5,
        rank=rank, world_size=world_size,
    )

    info = res.halo_info
    errors = []

    # send_offsets / recv_offsets bit-exact
    if not torch.equal(info.send_offsets, oracle["send_offsets"]):
        errors.append(f"send_offsets mismatch:\n  v5={info.send_offsets.tolist()}\n  oracle={oracle['send_offsets'].tolist()}")
    if not torch.equal(info.recv_offsets, oracle["recv_offsets"]):
        errors.append(f"recv_offsets mismatch:\n  v5={info.recv_offsets.tolist()}\n  oracle={oracle['recv_offsets'].tolist()}")

    # send_local_indices: bit-exact (sorted by Fix 1 within each peer)
    if not torch.equal(info.send_local_indices, oracle["send_local_indices"]):
        errors.append(
            f"send_local_indices mismatch:\n"
            f"  v5={info.send_local_indices.tolist()[:20]}...\n"
            f"  oracle={oracle['send_local_indices'].tolist()[:20]}..."
        )

    # recv_ghost_indices: same arange semantics; bit-exact
    if not torch.equal(info.recv_ghost_indices, oracle["recv_ghost_indices"]):
        errors.append(
            f"recv_ghost_indices mismatch:\n"
            f"  v5 len={info.recv_ghost_indices.numel()}\n"
            f"  oracle len={oracle['recv_ghost_indices'].numel()}"
        )

    # halo_recv_global_ids: should match oracle's recv_atom_global order
    if not torch.equal(res.halo_recv_global_ids, oracle["recv_atom_global"]):
        errors.append(
            f"halo_recv_global_ids vs oracle.recv_atom_global mismatch:\n"
            f"  v5={res.halo_recv_global_ids.tolist()[:20]}...\n"
            f"  oracle={oracle['recv_atom_global'].tolist()[:20]}..."
        )

    # interior_mask
    if not torch.equal(info.interior_mask, oracle["interior_mask"]):
        diff = (info.interior_mask != oracle["interior_mask"]).sum().item()
        errors.append(f"interior_mask differs in {diff} positions out of {info.interior_mask.numel()}")

    # Mirror-symmetry test (Fix 1 verification):
    # rank 0's send_to_1 == rank 1's recv_from_0
    if world_size == 2:
        peer = 1 - rank
        # My send-to-peer atoms (global IDs)
        s_lo = info.send_offsets[peer].item()
        s_hi = info.send_offsets[peer + 1].item()
        my_send_to_peer_local = info.send_local_indices[s_lo:s_hi]
        my_send_to_peer_global = res.my_outer_atoms[my_send_to_peer_local]
        # My recv-from-peer atoms (global IDs)
        r_lo = info.recv_offsets[peer].item()
        r_hi = info.recv_offsets[peer + 1].item()
        my_recv_from_peer_global = res.halo_recv_global_ids[r_lo:r_hi]

        # Exchange to verify
        my_send_buf = my_send_to_peer_global.contiguous()
        my_recv_buf = my_recv_from_peer_global.contiguous()

        # gather size from peer
        peer_send_size = torch.zeros(1, dtype=torch.int64, device=device)
        my_size = torch.tensor([my_send_buf.numel()], dtype=torch.int64, device=device)
        ws = [torch.zeros_like(my_size) for _ in range(world_size)]
        dist.all_gather(ws, my_size)
        peer_send_size_val = ws[peer].item()

        if peer_send_size_val != my_recv_buf.numel():
            errors.append(
                f"size mismatch: peer's send-to-me ({peer_send_size_val}) "
                f"!= my recv-from-peer ({my_recv_buf.numel()})"
            )
        else:
            # Gather peer's send atoms
            peer_send_buf = torch.empty(peer_send_size_val, dtype=torch.int64, device=device)
            send_list = [my_send_buf, peer_send_buf] if rank == 0 else [peer_send_buf, my_send_buf]
            for src_rank in range(world_size):
                buf = my_send_buf if src_rank == rank else torch.empty(int(ws[src_rank].item()), dtype=torch.int64, device=device)
                dist.broadcast(buf, src=src_rank)
                if src_rank == peer:
                    peer_send_buf = buf
            if not torch.equal(my_recv_buf, peer_send_buf):
                errors.append(
                    f"mirror-symmetry violated:\n"
                    f"  my recv-from-peer={my_recv_buf.tolist()[:20]}\n"
                    f"  peer's send-to-me={peer_send_buf.tolist()[:20]}"
                )

    # Print result
    print(f"[rank {rank}] errors: {len(errors)}")
    for e in errors:
        print(f"[rank {rank}] ERROR: {e}")

    # Gather pass/fail across ranks
    err_count = torch.tensor([len(errors)], dtype=torch.int64, device=device)
    all_errs = [torch.zeros_like(err_count) for _ in range(world_size)]
    dist.all_gather(all_errs, err_count)
    total = sum(t.item() for t in all_errs)

    if rank == 0:
        print(f"\n[summary] total errors across ranks: {total}")
        print(f"[summary] N={N}, W={world_size}, cutoff=4.0+0.5")
        print(f"[summary] rank 0: num_local={res.my_outer_atoms.numel()}, "
              f"halo={res.halo_recv_global_ids.numel()}, "
              f"interior={int(info.interior_mask.sum().item())}")

    dist.destroy_process_group()
    sys.exit(0 if total == 0 else 1)

if __name__ == "__main__":
    main()
"""


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="needs ≥ 2 GPUs",
)
def test_v5_w2_correctness_vs_oracle():
    """Run W=2 via torchrun subprocess; compare v5 to oracle."""
    script = _multirank_runner_script()
    script_path = "/tmp/_test_v5_runner.py"
    with open(script_path, "w") as f:
        f.write(script)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/dynamis_state/src:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            script_path,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)
    assert result.returncode == 0, "v5 correctness test failed (see stdout)"


def _multirank_modeb_runner_script() -> str:
    """W=2 test for Mode B: edges_at_r should equal a fresh Mode A's edges_at_r."""
    return r"""
import os, sys
import torch
import torch.distributed as dist
import dynamis_state.dynamis_state_impl as _impl
from dynamis_state.dist_graph_builder import DistGraphBuilder, BuildMode


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.manual_seed(9999)
    N = 500
    L = 30.0
    pos = torch.rand(N, 3, device=device) * L
    cell = torch.eye(3, device=device, dtype=torch.float32) * L
    pbc = torch.tensor([True, True, True])

    env = _impl.DistEnvironment()
    env.global_rank = rank; env.world_size = world_size
    env.local_rank = local_rank; env.gpus_per_node = world_size; env.num_nodes = 1

    builder = DistGraphBuilder(
        cell, pbc, cutoff=4.0, ghost_buffer=0.5, env=env, refresh_interval=100,
    )
    # 1st call: Mode A (refresh) at pos
    res_a = builder.build(pos)
    assert res_a.mode == BuildMode.REFRESH

    # 2nd call (same pos): Mode B; edges should match Mode A's edges_at_r
    res_b = builder.build(pos)
    assert res_b.mode == BuildMode.STEADY

    # Canonicalize edges (sort by (dst, src)) for comparison
    def canon(e):
        if e.numel() == 0:
            return e
        key = e[1].to(torch.int64) * (10 * N) + e[0].to(torch.int64)
        order = torch.argsort(key)
        return e[:, order]

    ea = canon(res_a.edges_outer)
    eb = canon(res_b.edges_outer)

    errors = []
    if ea.shape != eb.shape:
        errors.append(f"shape mismatch A={ea.shape} B={eb.shape}")
    elif not torch.equal(ea, eb):
        errors.append(f"edges differ; A E={ea.shape[1]} B E={eb.shape[1]}")

    # Mode B small drift test: uniform |drift| <= 0.1 << skin/2 = 0.25
    pos_drift = pos + (torch.rand_like(pos) - 0.5) * 0.2   # |drift| <= 0.1 Å
    res_b2 = builder.build(pos_drift)
    if res_b2.mode != BuildMode.STEADY:
        errors.append(f"small drift should NOT trigger Mode A (got mode={res_b2.mode})")

    # Big drift test: drift = 1 Å >> skin/2 = 0.25, should trigger Mode A
    pos_big = pos + torch.ones_like(pos) * 1.0
    res_b3 = builder.build(pos_big)
    if res_b3.mode != BuildMode.REFRESH:
        errors.append(f"big drift should trigger Mode A (got mode={res_b3.mode})")

    err_count = torch.tensor([len(errors)], dtype=torch.int64, device=device)
    all_errs = [torch.zeros_like(err_count) for _ in range(world_size)]
    dist.all_gather(all_errs, err_count)
    total = sum(t.item() for t in all_errs)
    print(f"[rank {rank}] mode_b errors: {len(errors)}")
    for e in errors:
        print(f"[rank {rank}] ERROR: {e}")
    if rank == 0:
        print(f"[summary] mode_b test, total errors: {total}")

    dist.destroy_process_group()
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
"""


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="needs ≥ 2 GPUs",
)
def test_v5_w2_mode_b_consistency():
    """W=2: verify Mode B edges match Mode A's edges at the same pos."""
    script = _multirank_modeb_runner_script()
    script_path = "/tmp/_test_v5_modeb_runner.py"
    with open(script_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/dynamis_state/src:" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            script_path,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    print("=== STDOUT ===\n" + result.stdout)
    print("=== STDERR ===\n" + result.stderr)
    assert result.returncode == 0
