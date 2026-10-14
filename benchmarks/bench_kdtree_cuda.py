#!/usr/bin/env python3
"""
Benchmark: CUDA C++ KD-tree partitioner vs baseline Python recursive KD-tree.

Compares:
  1. C++ CUDA path  (KDTreePartitioner on device=cuda)
  2. Baseline Python (kdtree_weighted_gpu from scripts/kdtree_benchmark.py)

For both unweighted and weighted (edge-cost) modes.

Usage:
    python benchmarks/bench_kdtree_cuda.py [--sizes 1000 8000 64000 256000] [--k 4 8 16 64 256]
"""

import os
import sys
import time
import argparse
from typing import List, Dict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DYNAMIS = os.path.abspath(os.path.join(_HERE, os.pardir))
_ROOT = os.path.abspath(os.path.join(_DYNAMIS, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dynamis_state import MolecularInfo, DistEnvironment, KDTreePartitioner


def _make_state(N: int, device: torch.device) -> MolecularInfo:
    state = MolecularInfo()
    state.num_atoms = N
    state.atom_vertex_numbers = torch.arange(N, dtype=torch.long, device=device)
    state.atom_positions = torch.rand(N, 3, dtype=torch.float32, device=device) * 20.0
    state.atom_velocities = torch.zeros(N, 3, dtype=torch.float32, device=device)
    return state


def _make_env(W: int) -> DistEnvironment:
    env = DistEnvironment()
    env.global_rank = 0
    env.world_size = W
    env.gpus_per_node = W
    env.num_nodes = 1
    env.local_rank = 0
    return env


def time_fn(fn, device, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def run_benchmark(sizes: List[int], k_values: List[int], warmup: int, repeats: int):
    device = torch.device("cuda")

    print(f"{'='*90}")
    print(f"CUDA KD-Tree Partitioner Benchmark")
    print(f"{'='*90}")
    print(f"Warmup: {warmup}, Repeats: {repeats}")
    print(f"Sizes: {sizes}")
    print(f"K values: {k_values}")
    print()

    baseline_available = False
    try:
        from scripts.kdtree_benchmark import kdtree_weighted_gpu

        baseline_available = True
    except ImportError:
        print("  [WARN] Baseline not importable, skipping baseline comparison\n")

    all_results = []

    for N in sizes:
        print(f"\n{'='*90}")
        print(f"N = {N:,} atoms")
        print(f"{'='*90}")

        # Pre-create all data ONCE outside the timing loop
        positions = torch.rand(N, 3, dtype=torch.float32, device=device) * 20.0
        velocities = torch.zeros(N, 3, dtype=torch.float32, device=device)
        indices = torch.arange(N, dtype=torch.long, device=device)
        weights = torch.randint(1, 50, (N,), dtype=torch.long, device=device)

        for W in k_values:
            if W > N:
                continue

            env = _make_env(W)
            p = KDTreePartitioner()

            print(f"\n  W = {W}")
            hdr = f"  {'Method':<35s} {'Time(ms)':>10s} {'vs CUDA':>8s}"
            print(hdr)
            print(f"  {'-'*35} {'-'*10} {'-'*8}")

            row = {"N": N, "W": W}

            # --- C++ CUDA unweighted ---
            def fn_cuda_uw():
                s = MolecularInfo()
                s.num_atoms = N
                s.atom_vertex_numbers = indices
                s.atom_positions = positions
                s.atom_velocities = velocities
                return p.partition(s, env)

            t = time_fn(fn_cuda_uw, device, warmup, repeats)
            row["cuda_uw"] = t
            print(f"  {'C++ CUDA (unweighted)':<35s} {t:10.3f} {'ref':>8s}")

            # --- C++ CUDA weighted ---
            def fn_cuda_wt():
                s = MolecularInfo()
                s.num_atoms = N
                s.atom_vertex_numbers = indices
                s.atom_positions = positions
                s.atom_velocities = velocities
                s.edge_counts = weights
                return p.partition(s, env)

            t = time_fn(fn_cuda_wt, device, warmup, repeats)
            row["cuda_wt"] = t
            print(f"  {'C++ CUDA (weighted)':<35s} {t:10.3f} {'ref':>8s}")

            # --- Baseline Python GPU weighted ---
            if baseline_available:

                def fn_base():
                    return kdtree_weighted_gpu(positions, W, weights)

                t_base = time_fn(fn_base, device, warmup, repeats)
                row["base_wt"] = t_base
                ratio = t_base / row["cuda_wt"]
                print(
                    f"  {'Baseline Python GPU (weighted)':<35s} {t_base:10.3f} {ratio:7.2f}x"
                )

            all_results.append(row)

    # Summary
    print(f"\n\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    cols = ["N", "W", "CUDA-UW", "CUDA-WT"]
    if baseline_available:
        cols += ["Base-WT", "Base/CUDA"]
    hdr = "  ".join(f"{c:>12s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        vals = [
            f"{r['N']:>12,d}",
            f"{r['W']:>12d}",
            f"{r.get('cuda_uw',0):>12.3f}",
            f"{r.get('cuda_wt',0):>12.3f}",
        ]
        if baseline_available and "base_wt" in r:
            ratio = r["base_wt"] / r["cuda_wt"]
            vals += [f"{r['base_wt']:>12.3f}", f"{ratio:>11.2f}x"]
        print("  ".join(vals))


def main():
    parser = argparse.ArgumentParser(description="CUDA KD-tree partitioner benchmark")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 8000, 64000, 256000],
        help="Atom counts to benchmark",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4, 8, 16, 64, 256],
        help="Partition counts to test",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    run_benchmark(args.sizes, args.k, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
