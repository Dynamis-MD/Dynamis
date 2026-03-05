"""Benchmark: CUDA radius_graph_pbc vs v2 reference."""

import torch
import time
import importlib.util
import os

# Load reference implementation directly
_ref_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "baseline",
    "local_fairchem", "graph", "radius_graph_pbc.py"
)
_spec = importlib.util.spec_from_file_location("_ref", _ref_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
radius_graph_pbc_v2 = _mod.radius_graph_pbc_v2

import distnnp_state.distnnp_state_impl as _impl


class FakeData:
    def __init__(self, pos, cell, natoms):
        self.pos = pos
        self.cell = cell
        self.natoms = natoms
        self.pbc = torch.tensor([[True, True, True]])
        self.batch = torch.zeros(pos.size(0), dtype=torch.int, device=pos.device)


def bench(name, n_atoms, box_size, radius, max_nbrs, n_warmup=5, n_iter=20):
    torch.manual_seed(42)
    dev = torch.device("cuda")
    pos = torch.rand(n_atoms, 3, device=dev) * box_size
    cell = (torch.eye(3, device=dev) * box_size).unsqueeze(0)
    natoms = torch.tensor([n_atoms], device=dev)
    pbc_flags = torch.tensor([True, True, True])
    data = FakeData(pos, cell, natoms)

    # Warmup + correctness check
    ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_nbrs)
    cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
        pos, cell[0], pbc_flags, radius, max_nbrs, False
    )
    ref_edges = ref_ei.shape[1]
    cuda_edges = cuda_ei.shape[1]
    match = "OK" if ref_edges == cuda_edges else "MISMATCH"

    for _ in range(n_warmup):
        radius_graph_pbc_v2(data, radius, max_nbrs)
        _impl.radius_graph_pbc(pos, cell[0], pbc_flags, radius, max_nbrs, False)
    torch.cuda.synchronize()

    # Benchmark v2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        radius_graph_pbc_v2(data, radius, max_nbrs)
    torch.cuda.synchronize()
    v2_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark CUDA
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _impl.radius_graph_pbc(pos, cell[0], pbc_flags, radius, max_nbrs, False)
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) / n_iter * 1000

    speedup = v2_ms / cuda_ms
    print(
        f"{name:30s}  N={n_atoms:>6d}  edges={ref_edges:>8d}  "
        f"v2={v2_ms:8.2f}ms  cuda={cuda_ms:8.2f}ms  "
        f"speedup={speedup:5.2f}x  {match}"
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 115)
    print(
        f"{'Config':30s}  {'N':>8s}  {'edges':>10s}  "
        f"{'v2':>10s}  {'cuda':>10s}  {'speedup':>8s}  check"
    )
    print("=" * 115)
    bench("100 atoms, r=5, box=20",      100,  20.0, 5.0, 200)
    bench("500 atoms, r=5, box=25",      500,  25.0, 5.0, 200)
    bench("1K atoms, r=5, box=30",      1000,  30.0, 5.0, 200)
    bench("2K atoms, r=6, box=40",      2000,  40.0, 6.0, 500)
    bench("4K atoms, r=6, box=50",      4000,  50.0, 6.0, 500)
    bench("10K atoms, r=6, box=65",    10000,  65.0, 6.0, 500)
    print("=" * 115)
