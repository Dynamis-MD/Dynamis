"""Unified tool for v5 distributed build_graph (DistGraphBuilder).

One script, four modes (--mode), sharing the dataset loader and a launcher-aware
distributed init:

  timing  (default)  Mode A/B/amortized timing + single-GPU baseline speedup +
                     halo/interior stats + distributed per-rank peak memory,
                     one row per dataset (--systems).
  verify             distributed edge set == single-GPU serial build, bit-exact
                     (unique (src,dst) multiset); reports duplicate-edge count.
  sizes              sweep dataset sizes (largest->smallest, ch4 ladder) until
                     OOM; per-stage timing + peak memory.
  diag               decompose ONE dataset's distributed-vs-serial edge diff
                     (--system): duplicates, set-difference, min-image distances.

World size W is set by the launcher; the script auto-detects whether it was
started by torchrun (RANK/WORLD_SIZE) or SLURM srun (SLURM_PROCID/NTASKS), or
runs single-process otherwise. --w, if given, is asserted against the detected
world size (a guard against launcher/argument mismatch).

Launch:
  torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode timing
  srun --ntasks=8 ... python benchmarks/dist_build_graph_benchmark.py --mode timing --w 8
  torchrun ... benchmarks/dist_build_graph_benchmark.py --mode verify --systems 9gmo_density,2rik_density
  torchrun ... benchmarks/dist_build_graph_benchmark.py --mode diag --system 9gmo_512k_solvated
  torchrun --nproc_per_node=2 benchmarks/dist_build_graph_benchmark.py --mode sizes
"""
import argparse
import gc
import os
import re
import statistics
import sys
import traceback

import torch
import torch.distributed as dist

import dynamis_state.dynamis_state_impl as _impl
from dynamis_state.dist_graph_builder import (
    DistGraphBuilder,
    _cell_block_partition,
    _all_reduce_degrees,
    _compute_sfc_weighted_partition,
    _extract_halo_info_local,
    _filter_edges_to_radius,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))

CUTOFF, GHOST = 6.0, 0.5
R_SKIN = CUTOFF + GHOST
MAX_NBRS = 1000

# timing / verify default systems: ch4 + solvated proteins + membrane + water
DEFAULT_SYSTEMS = [
    "ch4_8k_splited",
    "ch4_128k_splited",
    "ch4_512k_splited",
    "1aki_density",
    "2rik_density",
    "dppc_membrane_28k",
    "pt_water_17k_density",
    "9gmo_density",
    "9gmo_512k_solvated",
]

# sizes mode: ch4 ladder (k units), largest -> smallest
SIZES = [1024, 512, 256, 128, 64, 32, 16, 8]


# ─────────────────────────────────────────────────────────────────────────────
# Shared infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def init_dist():
    """Auto-detect launcher (torchrun vs SLURM srun vs single-process)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        launcher = "torchrun"
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        launcher = "slurm"
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ["SLURM_LOCALID"])
        # env:// rendezvous needs RANK/WORLD_SIZE; MASTER_ADDR/PORT come from the
        # sbatch env (see benchmarks/_bench_env.sh).
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
    else:
        launcher = "single"
        rank, world, local = 0, 1, 0
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    ndev = torch.cuda.device_count()
    if ndev == 0:
        raise RuntimeError(
            "init_dist: no CUDA device visible (torch.cuda.device_count()==0). "
            "Run on a GPU node / inside a GPU allocation.")
    torch.cuda.set_device(local % ndev)
    device = torch.device(f"cuda:{local % ndev}")
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world)
    return rank, world, local, device, launcher


def load_and_broadcast(name, rank, device):
    """rank 0 reads <name>.xyz; broadcast N, positions, cell to all ranks."""
    path = os.path.join(_DATA, f"{name}.xyz")
    if rank == 0:
        with open(path) as f:
            n = int(f.readline())
            hdr = f.readline()
        vals = list(map(float, re.search(r'Lattice="([^"]+)"', hdr).group(1).split()))
        cell = torch.tensor(vals, dtype=torch.float32).reshape(3, 3).to(device)
        positions = []
        with open(path) as f:
            f.readline(); f.readline()
            for _ in range(n):
                p = f.readline().split()
                positions.append([float(p[1]), float(p[2]), float(p[3])])
        pos = torch.tensor(positions, dtype=torch.float32, device=device)
        n_t = torch.tensor([n], dtype=torch.int64, device=device)
        cell_flat = cell.reshape(-1).contiguous()
    else:
        n_t = torch.zeros(1, dtype=torch.int64, device=device)
        cell_flat = torch.zeros(9, dtype=torch.float32, device=device)
    dist.broadcast(n_t, src=0)
    N = int(n_t.item())
    if rank != 0:
        pos = torch.zeros(N, 3, dtype=torch.float32, device=device)
    dist.broadcast(pos, src=0)
    dist.broadcast(cell_flat, src=0)
    return pos.contiguous(), cell_flat.reshape(3, 3), N


def make_env(rank, world, local):
    env = _impl.DistEnvironment()
    env.global_rank = rank
    env.world_size = world
    env.local_rank = local % torch.cuda.device_count()
    gpn = int(os.environ.get("SLURM_NTASKS_PER_NODE",
                             os.environ.get("LOCAL_WORLD_SIZE", world)))
    env.gpus_per_node = gpn
    env.num_nodes = max(1, world // max(1, gpn))
    return env


def time_dist(fn, device, warmup, iters):
    """Median ms of fn, reduced as max-over-ranks (distributed wall time)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        fn(); e.record(); torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    med = statistics.median(samples)
    t = torch.tensor([med], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def gather_rows(local_rows, world, device):
    """all_gather a [E_r, C] int64 tensor of differing E_r; return concatenated."""
    C = local_rows.shape[1]
    E_r = local_rows.shape[0]
    sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world)]
    dist.all_gather(sizes, torch.tensor([E_r], dtype=torch.int64, device=device))
    sizes = [int(s.item()) for s in sizes]
    maxE = max(sizes) if sizes else 0
    pad = torch.zeros(maxE, C, dtype=torch.int64, device=device)
    if E_r:
        pad[:E_r] = local_rows
    buf = [torch.zeros(maxE, C, dtype=torch.int64, device=device) for _ in range(world)]
    dist.all_gather(buf, pad)
    return torch.cat([buf[r][:sizes[r]] for r in range(world)], dim=0)


def canon_key_pairs(src, dst, N):
    return torch.sort(dst.to(torch.int64) * (N + 1) + src.to(torch.int64)).values


# ─────────────────────────────────────────────────────────────────────────────
# Mode: timing
# ─────────────────────────────────────────────────────────────────────────────

def _timing_one(name, rank, world, local, device):
    pos, cell, N = load_and_broadcast(name, rank, device)
    box = cell.diag().tolist()
    pbc = torch.tensor([True, True, True])
    env = make_env(rank, world, local)

    def mk():
        return DistGraphBuilder(cell, pbc, CUTOFF, GHOST, env, refresh_interval=100)

    t_a = time_dist(lambda: mk().build(pos), device, warmup=2, iters=3)
    b = mk(); b.build(pos)
    t_b = time_dist(lambda: b.build(pos), device, warmup=2, iters=10)
    amort = (t_a + 99 * t_b) / 100

    n_outer = int(b._my_outer_atoms.numel())
    n_halo = int(b._halo_recv_global_ids.numel())
    n_int = int(b._interior_mask.sum().item())

    # Distributed per-rank peak: captured BEFORE the rank-0 single-GPU baseline
    # below (which builds the whole graph and would otherwise dominate it).
    dpk = torch.tensor([torch.cuda.max_memory_allocated(device)],
                       dtype=torch.int64, device=device)
    dist.all_reduce(dpk, op=dist.ReduceOp.MAX)
    peak_gb = float(dpk.item()) / 1024**3

    base_ms, base_edges = 0.0, 0
    if rank == 0:
        cell_d = cell.to(device); pbc_d = pbc.to(device)
        def base():
            return _impl.radius_graph_pbc(pos, cell_d, pbc_d, CUTOFF, MAX_NBRS, False)
        for _ in range(2):
            base()
        ev, out = [], None
        for _ in range(5):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record()
            out = base(); e.record(); torch.cuda.synchronize()
            ev.append(s.elapsed_time(e))
        base_ms = statistics.median(ev)
        base_edges = int(out[0].shape[1])

    return dict(N=N, box=box, t_a=t_a, t_b=t_b, amort=amort,
                n_outer=n_outer, n_halo=n_halo, n_int=n_int,
                base_ms=base_ms, base_edges=base_edges, peak_gb=peak_gb)


def mode_timing(systems, rank, world, local, device):
    if rank == 0:
        print(f"\n{'='*92}")
        print(f"  v5 distributed build_graph — timing — W={world}  "
              f"(cutoff={CUTOFF} + ghost={GHOST})")
        print(f"  baseline = single-GPU radius_graph_pbc on rank 0")
        print(f"{'='*92}\n")
        print(f"  {'system':<22}{'N':>9}{'box':>7}  {'A_ms':>8}{'B_ms':>8}{'amort':>8}  "
              f"{'base_ms':>8}{'speedup':>8}  {'r0_outer':>9}{'r0_halo':>8}{'int%':>6}  {'peakGB':>7}")
        print(f"  {'-'*88}")

    for name in systems:
        if not os.path.exists(os.path.join(_DATA, f"{name}.xyz")):
            if rank == 0:
                print(f"  {name:<22}  SKIP — file missing")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        status, errmsg, r = 0, "", None
        try:
            r = _timing_one(name, rank, world, local, device)
        except torch.cuda.OutOfMemoryError as e:
            status, errmsg = 1, str(e)[:90]; torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            status, errmsg = 2, str(e)[:140]
            if rank == 0:
                traceback.print_exc()
            torch.cuda.empty_cache()
        st = torch.tensor([status], dtype=torch.int64, device=device)
        dist.all_reduce(st, op=dist.ReduceOp.MAX)
        gstatus = int(st.item())
        if rank == 0:
            if gstatus == 0:
                speedup = (r["base_ms"] / r["amort"]) if r["amort"] > 0 else 0.0
                intpct = 100.0 * r["n_int"] / max(r["n_outer"], 1)
                print(f"  {name:<22}{r['N']:>9}{r['box'][0]:>7.0f}  "
                      f"{r['t_a']:>8.2f}{r['t_b']:>8.2f}{r['amort']:>8.2f}  "
                      f"{r['base_ms']:>8.2f}{speedup:>7.2f}x  "
                      f"{r['n_outer']:>9}{r['n_halo']:>8}{intpct:>5.0f}%  {r['peak_gb']:>7.2f}",
                      flush=True)
            elif gstatus == 1:
                print(f"  {name:<22}  OOM on some rank ({errmsg})", flush=True)
            else:
                print(f"  {name:<22}  ERROR on some rank ({errmsg})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: verify  (distributed edge set == single-GPU serial)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_one(name, rank, world, local, device):
    pos, cell, N = load_and_broadcast(name, rank, device)
    pbc = torch.tensor([True, True, True])
    env = make_env(rank, world, local)
    res = DistGraphBuilder(cell, pbc, CUTOFF, GHOST, env, refresh_interval=100).build(pos)
    e = res.edges_outer
    local_rows = torch.stack([e[0].to(torch.int64), e[1].to(torch.int64)], dim=1)
    allp = gather_rows(local_rows, world, device)
    if rank != 0:
        return None
    dsrc, ddst = allp[:, 0], allp[:, 1]
    s_e, _, _ = _impl.radius_graph_pbc(pos, cell.to(device), pbc.to(device),
                                       CUTOFF, MAX_NBRS, False)
    dkey_u = torch.unique(canon_key_pairs(dsrc, ddst, N))
    skey_u = torch.unique(canon_key_pairs(s_e[0], s_e[1], N))
    pair_ok = (dkey_u.numel() == skey_u.numel()) and torch.equal(dkey_u, skey_u)
    dups = int(dsrc.numel() - dkey_u.numel())
    return dict(N=N, E_dist=int(dsrc.numel()), E_serial=int(s_e.shape[1]),
                pair_ok=pair_ok, dups=dups)


def mode_verify(systems, rank, world, local, device):
    if rank == 0:
        print(f"\n{'='*72}")
        print(f"  v5 distributed vs single-GPU serial — edge-set equality — W={world}")
        print(f"{'='*72}")
        print(f"  {'system':<22}{'N':>9}{'E_dist':>12}{'E_serial':>12}  {'uniq=set':>9}{'dups':>6}")
        print(f"  {'-'*70}")
    n_pass, total = 0, 0
    for name in systems:
        if not os.path.exists(os.path.join(_DATA, f"{name}.xyz")):
            if rank == 0:
                print(f"  {name:<22}  SKIP — missing")
            continue
        total += 1
        torch.cuda.empty_cache()
        r = _verify_one(name, rank, world, local, device)
        if rank == 0:
            n_pass += int(r["pair_ok"])
            print(f"  {name:<22}{r['N']:>9}{r['E_dist']:>12,}{r['E_serial']:>12,}  "
                  f"{'PASS' if r['pair_ok'] else 'FAIL':>9}{r['dups']:>6}", flush=True)
    if rank == 0:
        print(f"  {'-'*70}")
        print(f"  {n_pass}/{total}: distributed edge multiset == serial (bit-exact)\n")
        if n_pass != total:
            return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Mode: sizes  (ch4 ladder, per-stage timing + OOM probing)
# ─────────────────────────────────────────────────────────────────────────────

def _sizes_one(pos, cell, rank, world, local, device, N, iters=3, warmup=1):
    r_plus = R_SKIN
    env = make_env(rank, world, local)
    L_vec = cell.diag().contiguous().to(device)
    pbc_d = torch.tensor([True, True, True], device=device)
    cell_d = cell.to(device).to(pos.dtype)
    times = {}

    def tic(name, fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        out = fn(); e.record(); torch.cuda.synchronize()
        times.setdefault(name, []).append(s.elapsed_time(e))
        return out

    for it in range(warmup + iters):
        rec = (it >= warmup)
        run = (lambda n, f: tic(n, f)) if rec else (lambda n, f: f())
        pos_w = run("pbc",
            lambda: pos - (pos / L_vec.unsqueeze(0)).floor() * L_vec.unsqueeze(0))
        my_cells = run("cell_block",
            lambda: _cell_block_partition(pos_w, L_vec, r_plus, rank, world))
        deg = run("count",
            lambda: _impl.radius_graph_pbc_counts_ex(pos_w, cell_d, pbc_d, CUTOFF, my_cells))
        full_deg = run("ar_N", lambda: _all_reduce_degrees(deg, my_cells, N))
        pid = run("sfc", lambda: _compute_sfc_weighted_partition(pos_w, full_deg, env))
        my_outer = run("my_outer",
            lambda: torch.nonzero(pid.eq(rank), as_tuple=False).squeeze(-1))
        edges, cs = run("edge",
            lambda: _impl.radius_graph_pbc_ex(pos_w, cell_d, pbc_d, r_plus, 0, False, my_outer)[:2])
        run("halo",
            lambda: _extract_halo_info_local(edges, pid, my_outer, rank, world, N, device))
        run("filter", lambda: _filter_edges_to_radius(edges, cs, pos_w, cell_d, CUTOFF))

    meds = {k: statistics.median(v) for k, v in times.items()}
    return meds, sum(meds.values()), torch.cuda.max_memory_allocated(device) / 1024**3


def mode_sizes(rank, world, local, device):
    stages = ("pbc", "cell_block", "count", "ar_N", "sfc", "my_outer", "edge", "halo", "filter")
    if rank == 0:
        print(f"\n  v5 size sweep (per-stage) — W={world}\n")
    for kk in SIZES:
        path = os.path.join(_DATA, f"ch4_{kk}k_splited.xyz")
        if not os.path.exists(path):
            if rank == 0:
                print(f"  SKIP ch4_{kk}k — missing", flush=True)
            continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        status, errmsg, meds, total, peak = 0, "", None, None, None
        try:
            pos, cell, N = load_and_broadcast(f"ch4_{kk}k_splited", rank, device)
            meds, total, peak = _sizes_one(pos, cell, rank, world, local, device, N)
            del pos, cell
        except torch.cuda.OutOfMemoryError as e:
            status, errmsg = 1, str(e)[:90]; torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            status, errmsg = 2, str(e)[:140]
            if rank == 0:
                traceback.print_exc()
            torch.cuda.empty_cache()
        st = torch.tensor([status], dtype=torch.int64, device=device)
        dist.all_reduce(st, op=dist.ReduceOp.MAX)
        gstatus = int(st.item())
        if rank == 0:
            if gstatus == 0:
                print(f"  === ch4_{kk}k  N={N} ===", flush=True)
                for k in stages:
                    print(f"    {k:<12} {meds[k]:8.2f} ms", flush=True)
                print(f"    {'TOTAL':<12} {total:8.2f} ms   peak {peak:.2f} GB", flush=True)
            elif gstatus == 1:
                print(f"  ch4_{kk}k — OOM ({errmsg}) — stopping sweep", flush=True)
            else:
                print(f"  ch4_{kk}k — ERROR ({errmsg}) — stopping sweep", flush=True)
        if gstatus != 0:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Mode: diag  (decompose one dataset's distributed-vs-serial edge diff)
# ─────────────────────────────────────────────────────────────────────────────

def mode_diag(system, rank, world, local, device):
    if system is None:
        if rank == 0:
            print("  --mode diag requires --system <name>")
        return 2
    pos, cell, N = load_and_broadcast(system, rank, device)
    pbc = torch.tensor([True, True, True])
    env = make_env(rank, world, local)
    res = DistGraphBuilder(cell, pbc, CUTOFF, GHOST, env, refresh_interval=100).build(pos)
    e = res.edges_outer
    local_rows = torch.stack([e[0].to(torch.int64), e[1].to(torch.int64)], dim=1)
    allp = gather_rows(local_rows, world, device)

    my_outer = res.my_outer_atoms.to(torch.int64)
    own = torch.zeros(N, dtype=torch.int64, device=device)
    own[my_outer] = 1
    dist.all_reduce(own, op=dist.ReduceOp.SUM)
    if rank != 0:
        return 0

    multi_owned = int((own > 1).sum().item())
    unowned = int((own == 0).sum().item())
    key = allp[:, 1] * (N + 1) + allp[:, 0]
    uniq, counts = torch.unique(key, return_counts=True)
    n_dup_edges = int((counts[counts > 1] - 1).sum().item())

    s_e, _, _ = _impl.radius_graph_pbc(pos, cell.to(device), pbc.to(device),
                                       CUTOFF, MAX_NBRS, False)
    skey = s_e[1].to(torch.int64) * (N + 1) + s_e[0].to(torch.int64)
    suniq = torch.unique(skey)
    only_dist = uniq[~torch.isin(uniq, suniq)]
    only_serial = suniq[~torch.isin(suniq, uniq)]

    print(f"\n=== {system}  N={N:,}  W={world} ===")
    print(f"  partition: atoms owned by >1 rank = {multi_owned}, unowned = {unowned}")
    print(f"  dist edges raw      = {allp.shape[0]:,}")
    print(f"  dist edges unique   = {uniq.numel():,}   (duplicate rows = {n_dup_edges:,})")
    print(f"  serial edges raw    = {s_e.shape[1]:,}")
    print(f"  serial edges unique = {suniq.numel():,}")
    print(f"  |dist_unique \\ serial| = {only_dist.numel():,}")
    print(f"  |serial \\ dist_unique| = {only_serial.numel():,}")

    cell_d = cell.to(device)
    inv_cell = torch.linalg.inv(cell_d)
    sample = torch.cat([only_dist[:5], only_serial[:5]])
    if sample.numel():
        print(f"  min-image distances of differing edges (cutoff={CUTOFF}):")
        only_dist_set = set(only_dist.tolist())
        for k in sample.tolist():
            src = int(k % (N + 1)); dst = int(k // (N + 1))
            d = (pos[dst] - pos[src]).unsqueeze(0)
            frac = d @ inv_cell
            frac = frac - frac.round()
            dd = float(torch.linalg.norm(frac @ cell_d).item())
            side = "dist-only" if k in only_dist_set else "serial-only"
            print(f"    {side:11s}  src={src:>7} dst={dst:>7}  |r|={dd:.6f}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["timing", "verify", "sizes", "diag"],
                    default="timing")
    ap.add_argument("--systems", default=None,
                    help="comma-separated .xyz stems (timing/verify; default built-in)")
    ap.add_argument("--system", default=None, help="single dataset (diag mode)")
    ap.add_argument("--w", type=int, default=None,
                    help="expected world size; asserted against the detected one")
    args = ap.parse_args()

    rank, world, local, device, launcher = init_dist()
    if args.w is not None and args.w != world:
        if rank == 0:
            print(f"ERROR: --w={args.w} but launcher gave world_size={world} "
                  f"({launcher}). Check --nproc_per_node / --ntasks.", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(2)
    if rank == 0:
        print(f"[launcher={launcher}  world={world}  mode={args.mode}]", flush=True)

    systems = args.systems.split(",") if args.systems else DEFAULT_SYSTEMS
    rc = 0
    if args.mode == "timing":
        mode_timing(systems, rank, world, local, device)
    elif args.mode == "verify":
        rc = mode_verify(systems, rank, world, local, device) or 0
    elif args.mode == "sizes":
        mode_sizes(rank, world, local, device)
    elif args.mode == "diag":
        rc = mode_diag(args.system, rank, world, local, device) or 0

    dist.destroy_process_group()
    if rank == 0 and rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
