"""Benchmark: Does NCCL all_to_all_single actually skip 0-byte messages?

Measures NCCL all_to_all latency as a function of sparsity
(fraction of peers with 0-byte messages).

Launch with:
  torchrun --nproc_per_node=4 benchmarks/bench_nccl_sparse.py
"""

import argparse
import torch
import torch.distributed as dist


def benchmark_a2a(send_splits, recv_splits, feature_dim, device,
                  warmup=20, iters=200):
    """Time all_to_all_single with given splits."""
    total_send = sum(send_splits)
    total_recv = sum(recv_splits)

    send_tensor = torch.randn(total_send, feature_dim, device=device)
    recv_tensor = torch.empty(total_recv, feature_dim, device=device)

    # Warmup
    for _ in range(warmup):
        dist.all_to_all_single(recv_tensor, send_tensor,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits)

    torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        dist.all_to_all_single(recv_tensor, send_tensor,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits)
        end_events[i].record()

    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return sum(trimmed) / len(trimmed), min(times), max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dim", type=int, default=1152)
    parser.add_argument("--atoms_per_peer", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    F = args.feature_dim
    atoms = args.atoms_per_peer

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"NCCL all_to_all Sparsity Benchmark")
        print(f"  world_size={world_size}, feature_dim={F}, atoms_per_peer={atoms}")
        print(f"  bytes_per_active_peer = {atoms * F * 4 / 1024:.1f} KB")
        print(f"{'='*70}")
        print(f"\n  {'Active peers':<15} {'Sparsity':>10} {'Avg ms':>10} {'Min ms':>10}"
              f" {'Data KB':>10} {'BW GB/s':>10}")
        print(f"  {'-'*65}")

    # Test different sparsity levels
    for num_active in range(world_size, 0, -1):
        # Each rank sends to `num_active` peers (excluding self)
        # For simplicity: send to ranks 0..num_active-1 (skipping self)
        send_splits = []
        recv_splits = []
        for r in range(world_size):
            if r == rank:
                send_splits.append(0)
                recv_splits.append(0)
            elif r < num_active:
                send_splits.append(atoms)
                recv_splits.append(atoms)
            else:
                send_splits.append(0)
                recv_splits.append(0)

        sparsity = 1.0 - num_active / (world_size - 1) if world_size > 1 else 0
        data_kb = sum(send_splits) * F * 4 / 1024

        avg, mn, mx = benchmark_a2a(send_splits, recv_splits, F, device)

        bw = data_kb * 1024 / (avg * 1e-3) / 1e9 if avg > 0 and data_kb > 0 else 0

        if rank == 0:
            print(f"  {num_active:<15} {sparsity:>9.0%} {avg:>10.3f} {mn:>10.3f}"
                  f" {data_kb:>10.1f} {bw:>10.2f}")

    # Also test with uniform vs non-uniform splits
    if rank == 0:
        print(f"\n  --- Uniform (equal split) vs all_to_all ---")

    # Uniform: equal data to all peers
    uniform_splits = [atoms if r != rank else 0 for r in range(world_size)]
    avg_u, mn_u, _ = benchmark_a2a(uniform_splits, uniform_splits, F, device)

    # Pure zero: all zeros (should be fastest if NCCL skips)
    zero_splits = [0] * world_size
    avg_z, mn_z, _ = benchmark_a2a(zero_splits, zero_splits, F, device)

    if rank == 0:
        print(f"  {'All active':<15} {'':>10} {avg_u:>10.3f} {mn_u:>10.3f}")
        print(f"  {'All zero':<15} {'':>10} {avg_z:>10.3f} {mn_z:>10.3f}")
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
