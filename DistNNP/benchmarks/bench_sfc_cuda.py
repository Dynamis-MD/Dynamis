#!/usr/bin/env python3
"""Benchmark SFC partitioner (CUDA) vs KDTree partitioner and baseline Python."""

import os
import torch
import time
import sys

from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
    SFCPartitioner,
    SFCCurveType,
    KDTreePartitioner,
)

# Add baseline to path for comparison
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "baseline"))
try:
    from sfc import sfc_partition as baseline_sfc_partition
    HAS_BASELINE = True
except ImportError:
    HAS_BASELINE = False


def make_state(N, device):
    state = MolecularInfo()
    state.num_atoms = N
    state.atom_vertex_numbers = torch.arange(N, dtype=torch.long, device=device)
    state.atom_positions = torch.rand(N, 3, device=device) * 100.0
    state.atom_velocities = torch.zeros(N, 3, device=device)
    return state


def make_env(W):
    env = DistEnvironment()
    env.global_rank = 0
    env.world_size = W
    env.gpus_per_node = W
    env.num_nodes = 1
    env.local_rank = 0
    return env


def bench(fn, warmup=3, iters=10):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    device = torch.device("cuda")
    sizes = [1_000, 10_000, 100_000, 500_000]
    W_values = [4, 8, 16]

    print(f"{'N':>8} {'W':>4} {'Morton':>10} {'Hilbert':>10} {'KDTree':>10}", end="")
    if HAS_BASELINE:
        print(f" {'BL-Morton':>10} {'BL-Hilb':>10}", end="")
    print(f" {'Speedup':>10}")
    print("-" * 85)

    for N in sizes:
        state = make_state(N, device)
        for W in W_values:
            env = make_env(W)

            # C++ SFC Morton
            p_morton = SFCPartitioner(SFCCurveType.Morton)
            t_morton = bench(lambda: p_morton.partition(state, env))

            # C++ SFC Hilbert
            p_hilbert = SFCPartitioner(SFCCurveType.Hilbert)
            t_hilbert = bench(lambda: p_hilbert.partition(state, env))

            # C++ KDTree
            p_kd = KDTreePartitioner()
            t_kd = bench(lambda: p_kd.partition(state, env))

            line = f"{N:>8} {W:>4} {t_morton:>9.2f}ms {t_hilbert:>9.2f}ms {t_kd:>9.2f}ms"

            # Baseline Python
            if HAS_BASELINE:
                pos = state.atom_positions
                t_bl_morton = bench(lambda: baseline_sfc_partition(pos, W, "morton", device))
                t_bl_hilbert = bench(lambda: baseline_sfc_partition(pos, W, "hilbert", device))
                line += f" {t_bl_morton:>9.2f}ms {t_bl_hilbert:>9.2f}ms"
            else:
                t_bl_morton = None

            speedup_vs_kd = t_kd / t_morton if t_morton > 0 else float("inf")
            line += f" {speedup_vs_kd:>9.2f}x"
            print(line)

    # Weighted benchmark
    print("\n--- Weighted (edge_counts) ---")
    print(f"{'N':>8} {'W':>4} {'SFC-M-W':>10} {'SFC-H-W':>10} {'KD-W':>10} {'Speedup':>10}")
    print("-" * 55)

    for N in sizes:
        state = make_state(N, device)
        state.edge_counts = torch.randint(1, 100, (N,), dtype=torch.long, device=device)
        for W in [4, 16]:
            env = make_env(W)

            p_morton = SFCPartitioner(SFCCurveType.Morton)
            t_morton = bench(lambda: p_morton.partition(state, env))

            p_hilbert = SFCPartitioner(SFCCurveType.Hilbert)
            t_hilbert = bench(lambda: p_hilbert.partition(state, env))

            p_kd = KDTreePartitioner()
            t_kd = bench(lambda: p_kd.partition(state, env))

            speedup = t_kd / t_morton if t_morton > 0 else float("inf")
            print(f"{N:>8} {W:>4} {t_morton:>9.2f}ms {t_hilbert:>9.2f}ms {t_kd:>9.2f}ms {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
