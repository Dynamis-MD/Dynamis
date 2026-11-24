"""Tests for in-kernel max_num_neighbors truncation of the radius graph.

Semantics (see include/dynamis_state/radius_graph.hpp), per target atom (edge_index[1]):
  strict     — keep the k nearest (Orb); exact ties at the k-th neighbour are all kept.
  non-strict — keep d^2 <= d^2_(k+1) + 0.01 Å^2 (fairchem get_max_neighbors_mask / UMA).
Both are a threshold tau on d^2; radius_graph_pbc_select returns it and radius_graph_pbc_ex
accepts it back, which is how a rank builds edges for ghost centres (distributed Orb).

The reference here recomputes distances in float64 from the untruncated graph, so an edge
may only differ from it at a near-tie (|d^2 - boundary| below float32 resolution).
"""

import importlib.util
import os

import pytest
import torch

import dynamis_state.dynamis_state_impl as _impl
from dynamis_state.dist_graph_builder import BuildMode, DistGraphBuilder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEGENERACY_TOL = 0.01  # Å^2, fairchem default
NEAR_TIE = 1e-4  # Å^2: float32 vs float64 distance disagreement allowed at the boundary


# ===== Helpers =====


def make_system(n, box, seed, triclinic=False, pbc=(True, True, True)):
    g = torch.Generator().manual_seed(seed)
    cell = torch.eye(3) * box
    if triclinic:
        cell[1, 0], cell[2, 0], cell[2, 1] = 0.3 * box, 0.2 * box, 0.25 * box
    pos = (torch.rand(n, 3, generator=g) @ cell).float().cuda()
    return pos, cell.float().cuda(), torch.tensor(pbc, dtype=torch.bool, device="cuda")


SYSTEMS = {
    # name: (n, box, triclinic, pbc) — density ~0.1-0.2 /Å^3, so degree >> k at r = 5-6 Å
    "cubic": (3000, 31.0, False, (True, True, True)),
    "triclinic": (1500, 25.0, True, (True, True, True)),
    "slab": (1500, 25.0, False, (True, True, False)),
    "small_box": (300, 12.0, False, (True, True, True)),  # box ~ 2r: multiple images
    # ~7.6 atoms per cell-list cell at r = 6: B4 takes the nosort (direct-write) path
    "dense": (3000, 22.0, False, (True, True, True)),
}


def edge_key_list(edge_index, cell_shifts):
    """(src, dst, shift) per edge, in edge order."""
    cs = cell_shifts.round().long()
    return list(
        zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
            cs[:, 0].tolist(),
            cs[:, 1].tolist(),
            cs[:, 2].tolist(),
        )
    )


def edge_keys(edge_index, cell_shifts):
    return set(edge_key_list(edge_index, cell_shifts))


def dist_sq64(edge_index, cell_shifts, pos, cell):
    p, c = pos.double(), cell.double()
    disp = p[edge_index[0]] + cell_shifts.double() @ c - p[edge_index[1]]
    return (disp * disp).sum(-1)


def reference_truncate(edge_index, cell_shifts, pos, cell, k, strict):
    """Per-target truncation of a full graph.

    Returns ((edge_index, cell_shifts) kept, {edge key: |d^2 - its target's boundary|}).
    """
    d2 = dist_sq64(edge_index, cell_shifts, pos, cell)
    dst = edge_index[1]
    n = int(dst.max()) + 1 if dst.numel() else 0
    order = torch.argsort(d2, stable=True)
    order = order[torch.argsort(dst[order], stable=True)]  # by dst, then d^2
    dst_s, d2_s = dst[order], d2[order]
    counts = torch.bincount(dst_s, minlength=n)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(dst_s.numel(), device=dst.device) - starts[dst_s]
    deg = counts[dst_s]
    truncated = deg > k
    kth1 = d2_s[(starts[dst_s] + k).clamp(max=d2_s.numel() - 1)]  # (k+1)-th smallest
    if strict:
        kth = d2_s[(starts[dst_s] + k - 1).clamp(max=d2_s.numel() - 1)]
        boundary = torch.where(truncated, 0.5 * (kth + kth1), torch.inf)
        keep = (~truncated) | (rank < k)
    else:
        boundary = torch.where(truncated, kth1 + DEGENERACY_TOL, torch.inf)
        keep = (~truncated) | (d2_s <= boundary)
    keep_idx = order[keep]
    edge_boundary = torch.empty_like(d2)
    edge_boundary[order] = boundary
    gap = (d2 - edge_boundary).abs()
    return (
        (edge_index[:, keep_idx], cell_shifts[keep_idx]),
        dict(zip(edge_key_list(edge_index, cell_shifts), gap.tolist())),
    )


def assert_same_up_to_near_ties(got, ref, gap_to_boundary):
    a, b = edge_keys(*got), edge_keys(*ref)
    diff = a ^ b
    far = [e for e in diff if gap_to_boundary.get(e, 0.0) >= NEAR_TIE]
    assert not far, f"{len(far)} edges differ away from the truncation boundary: {far[:5]}"
    assert len(diff) <= max(2, len(b) // 10000), f"too many near-tie differences: {len(diff)}"


def load_fairchem_reference():
    try:
        spec = importlib.util.find_spec("fairchem.core")
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = os.path.join(spec.submodule_search_locations[0], "graph", "radius_graph_pbc.py")
    if not os.path.exists(path):
        return None
    mod_spec = importlib.util.spec_from_file_location("_radius_graph_pbc_ref", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


class FakeData:
    """Minimal data object for fairchem's radius_graph_pbc_v2."""

    def __init__(self, pos, cell, pbc):
        self.pos = pos
        self.cell = cell.unsqueeze(0)
        self.natoms = torch.tensor([pos.shape[0]], device=pos.device)
        self.pbc = pbc.unsqueeze(0).cpu()
        self.batch = torch.zeros(pos.shape[0], dtype=torch.int, device=pos.device)


# ===== Truncation semantics =====


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("strict", [True, False])
# k <= 31: warp register top-k selector; k >= 32: radix selector
@pytest.mark.parametrize("k", [1, 20, 31, 32, 60])
def test_truncation_matches_reference(system, strict, k):
    n, box, tri, pbc_t = SYSTEMS[system]
    pos, cell, pbc = make_system(n, box, seed=7, triclinic=tri, pbc=pbc_t)
    r = 5.0 if system == "small_box" else 6.0

    full = _impl.radius_graph_pbc(pos, cell, pbc, r, 0, False)[:2]
    ref, gap = reference_truncate(*full, pos, cell, k, strict)
    got = _impl.radius_graph_pbc(pos, cell, pbc, r, k, strict)[:2]
    assert_same_up_to_near_ties(got, ref, gap)

    full_deg = torch.bincount(full[0][1], minlength=n)
    deg = torch.bincount(got[0][1], minlength=n)
    if strict:
        assert torch.equal(deg, full_deg.clamp(max=k))
    else:
        assert bool((deg >= full_deg.clamp(max=k + 1)).all())


@pytest.mark.parametrize("strict", [True, False])
def test_no_truncation_below_cap(strict):
    """k = 0, or k >= every degree, leaves the full radius graph."""
    pos, cell, pbc = make_system(1500, 25.0, seed=3)
    full = _impl.radius_graph_pbc(pos, cell, pbc, 6.0, 0, False)
    max_deg = int(torch.bincount(full[0][1]).max())
    for k in [max_deg, max_deg + 10]:
        got = _impl.radius_graph_pbc(pos, cell, pbc, 6.0, k, strict)
        assert edge_keys(*got[:2]) == edge_keys(*full[:2])
    tau, cnt = _impl.radius_graph_pbc_select(pos, cell, pbc, 6.0, max_deg, strict)
    assert bool(torch.isinf(tau).all())
    assert torch.equal(cnt, torch.bincount(full[0][1], minlength=1500).int())


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("k", [20, 60])
def test_matches_fairchem(strict, k):
    ref_mod = load_fairchem_reference()
    if ref_mod is None:
        pytest.skip("fairchem-core not installed")
    pos, cell, pbc = make_system(3000, 31.0, seed=11)
    got = _impl.radius_graph_pbc(pos, cell, pbc, 6.0, k, strict)[:2]
    ref = ref_mod.radius_graph_pbc_v2(FakeData(pos, cell, pbc), 6.0, k, strict, pbc=pbc.cpu())
    full = _impl.radius_graph_pbc(pos, cell, pbc, 6.0, 0, False)[:2]
    _, gap = reference_truncate(*full, pos, cell, k, strict)
    assert_same_up_to_near_ties(got, ref[:2], gap)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("k", [20, 60])
def test_select_counts_thresholds_round_trip(strict, k):
    pos, cell, pbc = make_system(3000, 31.0, seed=5)
    n = pos.shape[0]
    ei, cs, nn, cnt = _impl.radius_graph_pbc_ex(pos, cell, pbc, 6.0, k, strict)
    deg = torch.bincount(ei[1], minlength=n).int()
    assert torch.equal(cnt, deg) and int(nn) == ei.shape[1]

    tau, cnt_sel = _impl.radius_graph_pbc_select(pos, cell, pbc, 6.0, k, strict)
    assert tau.dtype == torch.float32 and tau.shape == (n,)
    assert torch.equal(cnt_sel, deg)
    assert torch.equal(_impl.radius_graph_pbc_counts_ex(pos, cell, pbc, 6.0, None, k, strict), deg)

    # tau separates kept from dropped edges of every target
    full = _impl.radius_graph_pbc(pos, cell, pbc, 6.0, 0, False)[:2]
    d2 = dist_sq64(*full, pos, cell)
    kept_keys = edge_keys(ei, cs)
    kept = torch.tensor([e in kept_keys for e in edge_key_list(*full)], device="cuda")
    t = tau.double()[full[0][1]]
    assert bool((d2[kept] <= t[kept] + NEAR_TIE).all())
    assert bool((d2[~kept] > t[~kept] - NEAR_TIE).all())

    # thresholds in → same graph, k and strict ignored
    ei2, cs2, _, cnt2 = _impl.radius_graph_pbc_ex(pos, cell, pbc, 6.0, 0, False, None, tau)
    assert edge_keys(ei2, cs2) == edge_keys(ei, cs) and torch.equal(cnt2, deg)

    # target subset: per-target outputs follow target_indices order
    tgt = torch.randperm(n, device="cuda")[: n // 3]
    tau_s, cnt_s = _impl.radius_graph_pbc_select(pos, cell, pbc, 6.0, k, strict, tgt)
    assert torch.equal(tau_s, tau[tgt]) and torch.equal(cnt_s, deg[tgt])
    ei_s, cs_s, _, cnt_s2 = _impl.radius_graph_pbc_ex(pos, cell, pbc, 6.0, k, strict, tgt)
    in_tgt = torch.zeros(n, dtype=torch.bool, device="cuda")
    in_tgt[tgt] = True
    sel = in_tgt[ei[1]]
    assert edge_keys(ei_s, cs_s) == edge_keys(ei[:, sel], cs[sel])
    assert torch.equal(cnt_s2, deg[tgt])
    ei_t, cs_t, _, _ = _impl.radius_graph_pbc_ex(pos, cell, pbc, 6.0, 0, False, tgt, tau_s)
    assert edge_keys(ei_t, cs_t) == edge_keys(ei_s, cs_s)


def test_select_requires_positive_k():
    pos, cell, pbc = make_system(100, 10.0, seed=1)
    with pytest.raises(RuntimeError, match="max_num_neighbors > 0"):
        _impl.radius_graph_pbc_select(pos, cell, pbc, 4.0, 0, True)
    with pytest.raises(RuntimeError, match="thresholds must have shape"):
        _impl.radius_graph_pbc_ex(pos, cell, pbc, 4.0, 0, False, None, torch.zeros(5, device="cuda"))


@pytest.mark.parametrize("k,strict_degree", [(6, 6), (10, 18), (20, 26), (40, 56), (100, 80)])
def test_exact_ties_on_lattice(k, strict_degree):
    """Simple cubic lattice (exact float distances): shells of 6, 12, 8, 6, 24, ... atoms.

    Strict keeps every neighbour tied with the k-th; non-strict matches fairchem.
    """
    a, m = 1.5, 10
    g = torch.arange(m, dtype=torch.float32)
    pos = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).cuda() * a
    cell = torch.eye(3, device="cuda") * (a * m)
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    ei, _, _ = _impl.radius_graph_pbc(pos, cell, pbc, 4.0, k, True)
    deg = torch.bincount(ei[1], minlength=pos.shape[0])
    assert bool((deg == strict_degree).all())

    ref_mod = load_fairchem_reference()
    if ref_mod is not None:
        got = _impl.radius_graph_pbc(pos, cell, pbc, 4.0, k, False)[:2]
        ref = ref_mod.radius_graph_pbc_v2(FakeData(pos, cell, pbc), 4.0, k, False, pbc=pbc.cpu())
        assert edge_keys(*got) == edge_keys(*ref[:2])


# ===== Distributed use (single-GPU emulation) =====


def _emulate_ranks(pos, cell, pbc, r, skin, k, strict, W, exchange_tau=True):
    """Slab-partition the atoms over W emulated ranks; each rank sees local + ghost atoms
    (within r + skin of a local atom) only — what a rank has after the halo exchange.
    Pass 1 computes tau for local centres, tau is "exchanged" for the ghosts, and pass 2
    builds with every centre's tau (exchange_tau=False: each rank instead truncates its
    local+ghost graph by itself). Returns, per rank, (global ids of its local atoms, the
    Orb-style edges it keeps — centre or neighbour local — in global ids)."""
    n = pos.shape[0]
    owner = torch.clamp((pos[:, 0] / cell[0, 0] * W).floor().long(), 0, W - 1)
    halo_ei = _impl.radius_graph_pbc(pos, cell, pbc, r + skin, 0, False)[0]
    tau_global = torch.empty(n, device=pos.device)
    ranks = []
    for R in range(W):
        local = torch.nonzero(owner == R).squeeze(1)
        need = torch.zeros(n, dtype=torch.bool, device=pos.device)
        need[halo_ei[0][owner[halo_ei[1]] == R]] = True
        need[local] = False
        ids = torch.cat([local, torch.nonzero(need).squeeze(1)])
        assert ids.numel() < n, "ghosts cover the whole system: the test cannot see tau"
        tau_local, _ = _impl.radius_graph_pbc_select(
            pos[ids], cell, pbc, r, k, strict, torch.arange(local.numel(), device=pos.device)
        )
        tau_global[local] = tau_local
        ranks.append((local, ids))
    out = []
    for local, ids in ranks:
        if exchange_tau:
            tau = tau_global[ids].contiguous()  # own tau + tau received for ghosts
            ei, cs, _, _ = _impl.radius_graph_pbc_ex(pos[ids], cell, pbc, r, 0, False, None, tau)
        else:
            ei, cs, _, _ = _impl.radius_graph_pbc_ex(pos[ids], cell, pbc, r, k, strict)
        n_local = local.numel()
        keep = (ei[0] < n_local) | (ei[1] < n_local)
        out.append((set(local.tolist()), edge_keys(ids[ei[:, keep]], cs[keep])))
    return out


def _edges_touching(edges, local_ids):
    return {e for e in edges if e[0] in local_ids or e[1] in local_ids}


@pytest.mark.parametrize("W", [2, 4])
@pytest.mark.parametrize("k,strict", [(20, True), (120, True), (20, False), (60, False)])
def test_tau_exchange_reproduces_global_graph(W, k, strict):
    """With tau exchanged for ghosts, every rank holds exactly the global truncated edges
    that touch its local atoms — including those whose centre is a ghost, which no rank
    could decide from its own atoms. Per rank, not the union: a ghost-centred edge missing
    on the neighbour's rank would still show up in the union via the centre's owner."""
    g = torch.Generator().manual_seed(11)
    pos = (torch.rand(3000, 3, generator=g) * 31.0).cuda()
    cell = torch.eye(3, device="cuda") * 31.0
    pbc = torch.ones(3, dtype=torch.bool, device="cuda")
    ref = edge_keys(*_impl.radius_graph_pbc(pos, cell, pbc, 6.0, k, strict)[:2])
    for local_ids, edges in _emulate_ranks(pos, cell, pbc, 6.0, 0.5, k, strict, W):
        assert edges == _edges_touching(ref, local_ids)
    # Control: without the exchange some rank gets ghost-centred edges wrong, so the check
    # above is able to fail. (k=120 is too loose to bind for most atoms; skip it there.)
    if k == 20 or k == 60:
        naive = _emulate_ranks(pos, cell, pbc, 6.0, 0.5, k, strict, W, exchange_tau=False)
        assert any(edges != _edges_touching(ref, local_ids) for local_ids, edges in naive)


@pytest.mark.parametrize("strict", [True, False])
def test_dist_graph_builder_single_rank(strict):
    """W=1 DistGraphBuilder with a cap: Mode A and Mode B both equal the global graph."""
    pos, cell, pbc = make_system(2000, 27.0, seed=9)
    env = _impl.DistEnvironment()
    env.global_rank, env.world_size, env.local_rank = 0, 1, 0
    env.gpus_per_node, env.num_nodes = 1, 1
    k = 20
    builder = DistGraphBuilder(
        cell, pbc.cpu(), cutoff=6.0, ghost_buffer=0.5, env=env, refresh_interval=100,
        max_num_neighbors=k, enforce_max_strictly=strict,
    )
    ref_ei, ref_cs, _, ref_cnt = _impl.radius_graph_pbc_ex(pos, cell, pbc, 6.0, k, strict)
    ref = edge_keys(ref_ei, ref_cs)

    res_a = builder.build(pos)
    assert res_a.mode == BuildMode.REFRESH
    assert edge_keys(res_a.edges_outer, res_a.cell_shifts_outer) == ref
    assert torch.equal(res_a.full_degrees, ref_cnt)

    res_b = builder.build(pos)
    assert res_b.mode == BuildMode.STEADY
    assert edge_keys(res_b.edges_outer, res_b.cell_shifts_outer) == ref
