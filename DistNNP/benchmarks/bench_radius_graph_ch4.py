"""Benchmark: CUDA radius_graph_pbc vs v2 on real CH4 splited data.

Reports: timing, GPU memory, correctness (edge count + edge-set match).
"""

import torch
import time
import importlib.util
import os
import sys
import re
import gc

# Load reference implementation directly (avoid baseline __init__.py)
_ref_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "baseline",
    "local_fairchem", "graph", "radius_graph_pbc.py"
)
_spec = importlib.util.spec_from_file_location("_ref", _ref_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
radius_graph_pbc_v2 = _mod.radius_graph_pbc_v2

import dynamis_state.dynamis_state_impl as _impl

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RADIUS = 6.0
MAX_NEIGHBORS = 50


class FakeData:
    def __init__(self, pos, cell, natoms):
        self.pos = pos
        self.cell = cell
        self.natoms = natoms
        self.pbc = torch.tensor([[True, True, True]])
        self.batch = torch.zeros(pos.size(0), dtype=torch.int, device=pos.device)


def parse_xyz(filepath, device):
    """Parse extended XYZ file → (pos [N,3], cell [1,3,3], natoms [1])."""
    with open(filepath, "r") as f:
        n_atoms = int(f.readline().strip())
        header = f.readline().strip()

        # Parse Lattice="a1 a2 a3 b1 b2 b3 c1 c2 c3"
        m = re.search(r'Lattice="([^"]+)"', header)
        lattice_vals = list(map(float, m.group(1).split()))
        cell = torch.tensor(lattice_vals, dtype=torch.float32).reshape(3, 3)

        positions = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    pos = torch.tensor(positions, dtype=torch.float32, device=device)
    cell = cell.to(device).unsqueeze(0)  # [1, 3, 3]
    natoms = torch.tensor([n_atoms], device=device)
    return pos, cell, natoms


def reset_gpu():
    """Reset GPU memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def verify_correctness(ref_ei, ref_co, cuda_ei, cuda_co):
    """Check edge-set equivalence (order-independent)."""
    if ref_ei.shape[1] != cuda_ei.shape[1]:
        return False, f"edge count {ref_ei.shape[1]} vs {cuda_ei.shape[1]}"

    if ref_ei.shape[1] == 0:
        return True, "both empty"

    # Build canonical edge tuples: (src, tgt, cs0, cs1, cs2)
    def edge_set(ei, co):
        src = ei[0].cpu()
        tgt = ei[1].cpu()
        cs = co.cpu().to(torch.int)
        # Pack into single int64 for fast set comparison
        keys = (src.long() << 40) | (tgt.long() << 26) | \
               ((cs[:, 0].long() + 4) << 6) | ((cs[:, 1].long() + 4) << 3) | (cs[:, 2].long() + 4)
        return set(keys.tolist())

    ref_set = edge_set(ref_ei, ref_co)
    cuda_set = edge_set(cuda_ei, cuda_co)

    if ref_set == cuda_set:
        return True, "exact match"
    else:
        missing = len(ref_set - cuda_set)
        extra = len(cuda_set - ref_set)
        return False, f"missing={missing} extra={extra}"


def bench_file(filepath, n_warmup=3, n_iter=10):
    """Benchmark a single xyz file."""
    device = torch.device("cuda")
    fname = os.path.basename(filepath)

    # Parse
    pos, cell, natoms = parse_xyz(filepath, device)
    n_atoms = pos.size(0)
    pbc_flags = torch.tensor([True, True, True])
    data = FakeData(pos, cell, natoms)

    # ===== V2 Reference =====
    reset_gpu()
    # Warmup
    v2_oom = False
    try:
        for _ in range(n_warmup):
            ref_ei, ref_co, _ = radius_graph_pbc_v2(data, RADIUS, MAX_NEIGHBORS)
        torch.cuda.synchronize()

        reset_gpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            ref_ei, ref_co, _ = radius_graph_pbc_v2(data, RADIUS, MAX_NEIGHBORS)
        torch.cuda.synchronize()
        v2_ms = (time.perf_counter() - t0) / n_iter * 1000
        v2_mem = get_peak_mb()
        ref_edges = ref_ei.shape[1]
    except torch.cuda.OutOfMemoryError:
        v2_oom = True
        v2_ms = float("inf")
        v2_mem = float("inf")
        ref_ei = ref_co = None
        ref_edges = -1
        torch.cuda.empty_cache()
        gc.collect()

    # ===== CUDA =====
    reset_gpu()
    cuda_oom = False
    try:
        for _ in range(n_warmup):
            cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
                pos, cell[0], pbc_flags, RADIUS, MAX_NEIGHBORS, False
            )
        torch.cuda.synchronize()

        reset_gpu()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
                pos, cell[0], pbc_flags, RADIUS, MAX_NEIGHBORS, False
            )
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) / n_iter * 1000
        cuda_mem = get_peak_mb()
        cuda_edges = cuda_ei.shape[1]
    except torch.cuda.OutOfMemoryError:
        cuda_oom = True
        cuda_ms = float("inf")
        cuda_mem = float("inf")
        cuda_ei = cuda_co = None
        cuda_edges = -1
        torch.cuda.empty_cache()
        gc.collect()

    # ===== Correctness =====
    if v2_oom and cuda_oom:
        check = "both OOM"
    elif v2_oom:
        check = f"v2 OOM, cuda edges={cuda_edges}"
    elif cuda_oom:
        check = f"cuda OOM, v2 edges={ref_edges}"
    elif ref_ei is not None and cuda_ei is not None:
        ok, detail = verify_correctness(ref_ei, ref_co, cuda_ei, cuda_co)
        check = detail if ok else f"FAIL: {detail}"
    else:
        check = "N/A"

    # ===== Speedup =====
    if v2_oom or cuda_oom:
        speedup_str = "N/A"
    else:
        speedup_str = f"{v2_ms / cuda_ms:.2f}x"

    # Format memory
    def fmt_mem(m):
        if m == float("inf"):
            return "OOM"
        elif m > 1024:
            return f"{m / 1024:.1f}GB"
        else:
            return f"{m:.0f}MB"

    def fmt_time(t):
        if t == float("inf"):
            return "OOM"
        elif t > 1000:
            return f"{t / 1000:.2f}s"
        else:
            return f"{t:.1f}ms"

    edges_str = str(ref_edges) if ref_edges >= 0 else str(cuda_edges) if cuda_edges >= 0 else "?"

    print(
        f"{fname:30s}  N={n_atoms:>8d}  edges={edges_str:>10s}  "
        f"v2={fmt_time(v2_ms):>9s} ({fmt_mem(v2_mem):>7s})  "
        f"cuda={fmt_time(cuda_ms):>9s} ({fmt_mem(cuda_mem):>7s})  "
        f"speedup={speedup_str:>7s}  {check}"
    )
    sys.stdout.flush()

    # Free memory before next test
    del pos, cell, natoms, data
    if ref_ei is not None:
        del ref_ei, ref_co
    if cuda_ei is not None:
        del cuda_ei, cuda_co
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    dev_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {dev_name} ({total_mem:.0f} GB)")
    print(f"Cutoff: {RADIUS} Å, max_neighbors: {MAX_NEIGHBORS}")
    print("=" * 145)
    print(
        f"{'File':30s}  {'N':>10s}  {'edges':>12s}  "
        f"{'v2 time':>9s} ({'v2 mem':>7s})  "
        f"{'cuda time':>9s} ({'cuda mem':>7s})  "
        f"{'speedup':>7s}  correctness"
    )
    print("=" * 145)

    sizes = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "100k", "128k", "256k", "512k", "1024k"]
    for sz in sizes:
        fpath = os.path.join(DATA_DIR, f"ch4_{sz}_splited.xyz")
        if os.path.exists(fpath):
            bench_file(fpath)
        else:
            print(f"ch4_{sz}_splited.xyz  -- NOT FOUND, skipping")

    print("=" * 145)
