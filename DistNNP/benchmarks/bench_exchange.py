"""Benchmark: IPC one-sided push vs NCCL all_to_all_single for feature exchange.

Launch with:
  torchrun --nproc_per_node=4 benchmarks/bench_exchange.py
  torchrun --nproc_per_node=4 benchmarks/bench_exchange.py --num_atoms 4000 --feature_dim 1152
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

from dynamis_state import (
    MolecularInfo,
    DistEnvironment,
    SimulationConfig,
    SimpleSpatialPartitioner,
    KDTreePartitioner,
    IPCExchanger,
)


def make_global_state(num_atoms, device):
    state = MolecularInfo()
    state.num_atoms = num_atoms
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long, device=device)
    state.atom_positions = torch.rand(num_atoms, 3, device=device) * 10.0
    state.atom_velocities = torch.zeros(num_atoms, 3, device=device)
    return state


def make_env(rank, world_size):
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = world_size
    env.num_nodes = 1
    env.local_rank = rank
    return env


def make_sim_config(cutoff, device, box_size=10.0):
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * box_size
    cfg.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return cfg


# ---------------------------------------------------------------------------
# NCCL all_to_all exchange (baseline approach)
# ---------------------------------------------------------------------------

def nccl_exchange(x_buffer, send_indices, send_splits, recv_splits,
                  recv_indices, num_local, feature_dim):
    """Replicate the baseline FeatureExchanger.exchange() logic."""
    # 1. Gather send buffer
    if send_indices.numel() > 0:
        send_tensor = x_buffer[send_indices].contiguous()
    else:
        send_tensor = torch.empty(0, feature_dim, device=x_buffer.device,
                                  dtype=x_buffer.dtype)

    # 2. all_to_all
    total_recv = sum(recv_splits)
    recv_tensor = torch.empty(total_recv, feature_dim, device=x_buffer.device,
                              dtype=x_buffer.dtype)
    dist.all_to_all_single(
        recv_tensor, send_tensor,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    # 3. Scatter to ghost slots
    if recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
        x_buffer.index_copy_(0, recv_indices, recv_tensor)

    return x_buffer


# ---------------------------------------------------------------------------
# IPC one-sided push exchange
# ---------------------------------------------------------------------------

def ipc_exchange_barrier(x_buffer, exchanger):
    """Push via IPC kernel, then dist.barrier() (slow)."""
    exchanger.push_forward(x_buffer)
    torch.cuda.synchronize()
    dist.barrier()
    return x_buffer


def ipc_exchange_events(x_buffer, exchanger, ce_threshold_bytes=0):
    """Push via IPC kernel, then IPC event sync (fast)."""
    exchanger.push_forward(x_buffer, ce_threshold_bytes=ce_threshold_bytes)
    exchanger.record_push_complete()
    exchanger.wait_for_pushes()
    return x_buffer


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark_fn(fn, warmup=10, iters=100, label=""):
    """Time fn() over iters iterations, return (avg_ms, min_ms, max_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Trim outliers: use middle 80%
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times), max(times), times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_atoms", type=int, default=4000)
    parser.add_argument("--feature_dim", type=int, default=1152)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--box_size", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--partitioner", choices=["simple", "kdtree"], default="kdtree")
    args = parser.parse_args()

    # Init torch.distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Deterministic positions (same across ranks)
    torch.manual_seed(42)
    state = make_global_state(args.num_atoms, device)
    env = make_env(rank, world_size)
    cfg = make_sim_config(args.cutoff, device, args.box_size)

    # Partition + halo maps
    if args.partitioner == "kdtree":
        p = KDTreePartitioner()
        info = p.partition(state, env, cfg)
    else:
        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
    p.build_halo_maps(info, state, env, cfg)

    num_local = info.partition_indices.shape[0]
    so = info.send_offsets.cpu()
    ro = info.recv_offsets.cpu()
    total_send = so[-1].item()
    total_recv = ro[-1].item()
    F = args.feature_dim

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Feature Exchange Benchmark")
        print(f"  atoms={args.num_atoms}, feature_dim={F}, cutoff={args.cutoff}")
        print(f"  box={args.box_size}, partitioner={args.partitioner}")
        print(f"  world_size={world_size}, warmup={args.warmup}, iters={args.iters}")
        print(f"{'='*70}")

    # Print per-rank stats
    for r in range(world_size):
        if rank == r:
            print(f"  Rank {rank}: local={num_local}, send={total_send}, "
                  f"recv={total_recv}, "
                  f"send_bytes={total_send * F * 4 / 1024:.1f} KB, "
                  f"recv_bytes={total_recv * F * 4 / 1024:.1f} KB")
        dist.barrier()

    # -----------------------------------------------------------------------
    # Setup NCCL baseline
    # -----------------------------------------------------------------------
    # x_buffer for NCCL (regular PyTorch allocation is fine)
    x_nccl = torch.randn(num_local + total_recv, F, device=device)

    # Pre-compute indices
    send_splits = [(so[r + 1] - so[r]).item() for r in range(world_size)]
    recv_splits = [(ro[r + 1] - ro[r]).item() for r in range(world_size)]

    send_indices = info.send_local_indices.to(device)
    recv_indices = info.recv_ghost_indices.to(device)

    # -----------------------------------------------------------------------
    # Setup IPC push
    # -----------------------------------------------------------------------
    x_ipc = IPCExchanger.allocate_ipc_buffer(num_local + total_recv, F, rank)
    grad_ipc = IPCExchanger.allocate_ipc_buffer(max(total_send, 1), F, rank)

    # Fill with same random data
    x_ipc[:num_local].copy_(x_nccl[:num_local])

    exchanger = IPCExchanger()
    x_handle = exchanger.export_ipc_handle(x_ipc)
    grad_handle = exchanger.export_ipc_handle(grad_ipc)

    # Exchange IPC handles via NCCL all_gather
    handle_size = len(x_handle)  # 64
    x_handle_tensor = torch.tensor(x_handle, dtype=torch.uint8, device=device)
    grad_handle_tensor = torch.tensor(grad_handle, dtype=torch.uint8, device=device)

    all_x_handles_tensor = torch.empty(world_size, handle_size, dtype=torch.uint8,
                                       device=device)
    all_grad_handles_tensor = torch.empty(world_size, handle_size, dtype=torch.uint8,
                                          device=device)

    dist.all_gather_into_tensor(all_x_handles_tensor, x_handle_tensor)
    dist.all_gather_into_tensor(all_grad_handles_tensor, grad_handle_tensor)

    all_x_handles = [all_x_handles_tensor[r].cpu().tolist() for r in range(world_size)]
    all_grad_handles = [all_grad_handles_tensor[r].cpu().tolist() for r in range(world_size)]

    # Gather recv_offsets and num_local from all ranks
    ro_tensor = info.recv_offsets.cpu().to(torch.long)
    all_ro_tensor = torch.empty(world_size, world_size + 1, dtype=torch.long,
                                device=device)
    dist.all_gather_into_tensor(all_ro_tensor, ro_tensor.to(device))

    nl_tensor = torch.tensor([num_local], dtype=torch.long, device=device)
    all_nl_tensor = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_nl_tensor, nl_tensor)

    exchanger.setup(
        x_ipc, grad_ipc,
        all_x_handles, all_grad_handles,
        info.send_offsets, info.send_local_indices,
        info.recv_offsets,
        all_ro_tensor, all_nl_tensor,
        num_local, F, rank, world_size,
    )

    # -----------------------------------------------------------------------
    # Setup IPC push-sync counter for stream-level sync
    # -----------------------------------------------------------------------
    sync_handle = exchanger.create_and_export_sync_handle()
    sync_handle_tensor = torch.tensor(sync_handle, dtype=torch.uint8, device=device)
    all_sync_tensors = torch.empty(world_size, len(sync_handle),
                                   dtype=torch.uint8, device=device)
    dist.all_gather_into_tensor(all_sync_tensors, sync_handle_tensor)
    all_sync_handles = [all_sync_tensors[r].cpu().tolist() for r in range(world_size)]
    exchanger.setup_sync(all_sync_handles)

    dist.barrier()
    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Verify correctness: both methods should produce the same ghost values
    # -----------------------------------------------------------------------
    x_nccl[:num_local] = torch.randn(num_local, F, device=device)
    x_ipc[:num_local].copy_(x_nccl[:num_local])

    nccl_exchange(x_nccl, send_indices, send_splits, recv_splits,
                  recv_indices, num_local, F)
    # Use barrier sync for correctness check (proven correct)
    ipc_exchange_barrier(x_ipc, exchanger)

    if total_recv > 0:
        max_diff = (x_nccl[num_local:] - x_ipc[num_local:]).abs().max().item()
        if rank == 0:
            print(f"\n  Correctness check (SM): max |NCCL - IPC| = {max_diff:.2e}")
        assert max_diff < 1e-6, f"Rank {rank}: NCCL vs IPC mismatch: {max_diff}"

    dist.barrier()

    # Verify CE path correctness
    x_ipc[num_local:].zero_()
    torch.cuda.synchronize()
    dist.barrier()

    exchanger.push_forward(x_ipc, ce_threshold_bytes=1)  # always CE
    torch.cuda.synchronize()
    dist.barrier()

    if total_recv > 0:
        max_diff_ce = (x_nccl[num_local:] - x_ipc[num_local:]).abs().max().item()
        if rank == 0:
            print(f"  Correctness check (CE): max |NCCL - IPC| = {max_diff_ce:.2e}")
        assert max_diff_ce < 1e-6, f"Rank {rank}: NCCL vs CE mismatch: {max_diff_ce}"

    dist.barrier()

    # -----------------------------------------------------------------------
    # Benchmark NCCL all_to_all
    # -----------------------------------------------------------------------
    def run_nccl():
        nccl_exchange(x_nccl, send_indices, send_splits, recv_splits,
                      recv_indices, num_local, F)

    nccl_avg, nccl_min, nccl_max, _ = benchmark_fn(
        run_nccl, warmup=args.warmup, iters=args.iters, label="NCCL")

    # -----------------------------------------------------------------------
    # Benchmark IPC push + IPC event sync
    # -----------------------------------------------------------------------
    def run_ipc_events():
        ipc_exchange_events(x_ipc, exchanger)

    ipc_e_avg, ipc_e_min, ipc_e_max, _ = benchmark_fn(
        run_ipc_events, warmup=args.warmup, iters=args.iters, label="IPC-events")

    # -----------------------------------------------------------------------
    # Benchmark IPC push + dist.barrier (slow, for reference)
    # -----------------------------------------------------------------------
    def run_ipc_barrier():
        ipc_exchange_barrier(x_ipc, exchanger)

    ipc_b_avg, ipc_b_min, ipc_b_max, _ = benchmark_fn(
        run_ipc_barrier, warmup=args.warmup, iters=args.iters, label="IPC-barrier")

    # -----------------------------------------------------------------------
    # Benchmark IPC push kernel only (no sync, for roofline)
    # -----------------------------------------------------------------------
    def run_ipc_kernel_only():
        exchanger.push_forward(x_ipc)
        torch.cuda.synchronize()

    ipc_k_avg, ipc_k_min, ipc_k_max, _ = benchmark_fn(
        run_ipc_kernel_only, warmup=args.warmup, iters=args.iters, label="IPC-kernel")

    # -----------------------------------------------------------------------
    # Benchmark IPC CE path at various thresholds
    # -----------------------------------------------------------------------
    # Threshold 1 means "always CE" (all peers exceed 1 byte)
    ce_thresholds = [1, 64*1024, 256*1024, 1024*1024]
    ce_results = []
    for thresh in ce_thresholds:
        def run_ce(t=thresh):
            ipc_exchange_events(x_ipc, exchanger, ce_threshold_bytes=t)
        avg, mn, mx, _ = benchmark_fn(
            run_ce, warmup=args.warmup, iters=args.iters,
            label=f"IPC-CE-{thresh//1024}K")
        ce_results.append((thresh, avg, mn, mx))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    num_metrics = 4 + len(ce_results)
    avg_tensor = torch.tensor(
        [nccl_avg, ipc_e_avg, ipc_b_avg, ipc_k_avg] +
        [r[1] for r in ce_results],
        device=device)
    all_avgs = torch.empty(world_size, num_metrics, device=device)
    dist.all_gather_into_tensor(all_avgs, avg_tensor)

    if rank == 0:
        data_bytes = max(total_send, total_recv) * F * 4
        print(f"\n{'='*70}")
        print(f"  Results (trimmed avg over {args.iters} iters, middle 80%)")
        print(f"{'='*70}")
        print(f"  {'Method':<30} {'Avg ms':>10} {'Min ms':>10} {'Max ms':>10} {'BW GB/s':>10}")
        print(f"  {'-'*70}")

        results = [
            ("NCCL all_to_all",      nccl_avg,  nccl_min,  nccl_max),
            ("IPC SM + events",      ipc_e_avg, ipc_e_min, ipc_e_max),
            ("IPC SM + barrier",     ipc_b_avg, ipc_b_min, ipc_b_max),
            ("IPC SM kernel only",   ipc_k_avg, ipc_k_min, ipc_k_max),
        ]
        for thresh, avg, mn, mx in ce_results:
            label = f"IPC CE (>={thresh//1024}KB) + events"
            results.append((label, avg, mn, mx))

        for method, avg, mn, mx in results:
            bw = data_bytes / (avg * 1e-3) / 1e9 if avg > 0 else 0
            print(f"  {method:<30} {avg:>10.3f} {mn:>10.3f} {mx:>10.3f} {bw:>10.2f}")

        speedup = nccl_avg / ipc_e_avg if ipc_e_avg > 0 else float('inf')
        print(f"\n  Speedup (IPC SM+events vs NCCL): {speedup:.2f}x")

        best_ce = min(ce_results, key=lambda x: x[1])
        speedup_ce = nccl_avg / best_ce[1] if best_ce[1] > 0 else float('inf')
        print(f"  Speedup (best CE vs NCCL): {speedup_ce:.2f}x "
              f"(threshold={best_ce[0]//1024}KB)")

        print(f"  Data per exchange: {data_bytes / 1024:.1f} KB "
              f"({data_bytes / 1024 / 1024:.2f} MB)")

        # Per-rank breakdown
        print(f"\n  Per-rank avg (ms):")
        header = f"  {'Rank':<6} {'NCCL':>10} {'SM+evt':>10} {'SM+bar':>10} {'SM-kern':>10}"
        for thresh, _, _, _ in ce_results:
            header += f" {'CE'+str(thresh//1024)+'K':>10}"
        print(header)
        for r in range(world_size):
            row = f"  {r:<6}"
            for c in range(num_metrics):
                row += f" {all_avgs[r,c].item():>10.3f}"
            print(row)
        print()

    exchanger.teardown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
