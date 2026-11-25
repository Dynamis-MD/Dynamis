"""2-rank check of PaddedSymmMemExchanger (docs/symm_exchanger.md): padded vs unpadded SymmMemExchanger, eager and under a
CUDA graph captured ONCE and replayed across several re-setups ("halo rebuilds") with changing ghost lists.
torchrun --nproc_per_node=2 benchmarks/uma_fused/test_padded_exchanger.py"""
import os
import types

import torch
import torch.distributed as dist

from dynamis_model.comm.symm_exchanger import SymmMemExchanger
from dynamis_model.comm.symm_exchanger_padded import PaddedSymmMemExchanger


def make_pi(rank, W, n_local, counts, seed):
    """counts[s][r] = rows s sends to r; send lists = random distinct local rows (same RNG on every rank)."""
    g = torch.Generator().manual_seed(seed)
    lists = {}
    for s in range(W):
        for r in range(W):
            lists[s, r] = torch.randperm(n_local[s], generator=g)[: counts[s][r]] if s != r else torch.zeros(0, dtype=torch.long)
    so = torch.tensor([0] + [counts[rank][r] for r in range(W)]).cumsum(0)
    ro = torch.tensor([0] + [counts[q][rank] for q in range(W)]).cumsum(0)
    sli = torch.cat([lists[rank, r] for r in range(W)])
    return types.SimpleNamespace(partition_indices=torch.arange(n_local[rank]), send_offsets=so, recv_offsets=ro,
                                 send_local_indices=sli)


def main():
    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    F = 64
    n_local = [1000, 1000]
    base = SymmMemExchanger()
    pad = PaddedSymmMemExchanger()
    seq = [([[0, 500], [420, 0]], 1), ([[0, 507], [415, 0]], 2), ([[0, 493], [430, 0]], 3),
           ([[0, 540], [455, 0]], 4), ([[0, 700], [420, 0]], 5)]   # last one overflows the 1.3 x + 16 capacity
    torch.manual_seed(100 + rank)
    x_static = torch.zeros(n_local[rank], F, device=dev)
    gw_static = None
    graph = None
    ok = True
    for i, (counts, seed) in enumerate(seq):
        pi = make_pi(rank, W, n_local, counts, seed)
        base.setup_exchanger(F, pi, dev)
        gen0 = getattr(pad, "generation", 0)
        pad.setup_exchanger(F, pi, dev)
        regen = pad.generation != gen0
        x = torch.randn(n_local[rank], F, device=dev)
        n_real = pad.n_real
        gw_real = torch.randn(n_real, F, device=dev)
        # reference: unpadded, eager
        y_ref = base.feature_push_impl(x, pi)
        gin_ref = base.feature_push_backward_impl(gw_real, None, pi)
        # padded, eager
        y_pad = pad.feature_push_impl(x, pi)
        gw_pad = torch.zeros(pad.n_full, F, device=dev)
        gw_pad[pad.real_rows] = gw_real
        gin_pad = pad.feature_push_backward_impl(gw_pad, None, pi)
        e_fwd = (y_pad[pad.real_rows] - y_ref).abs().max().item()
        e_bwd = (gin_pad - gin_ref).abs().max().item()
        # pooled exchanger (#148 API): narrower pushes reuse the same buffers; back-to-back pushes must not race
        e_pool = 0.0
        for ex in (base, pad):
            outs = [ex.feature_push_impl(x[:, :f].contiguous(), pi) for f in (3, 1, F, 3)]
            for f, o in zip((3, 1, F, 3), outs):
                rows = o if ex is base else o[pad.real_rows]
                e_pool = max(e_pool, (rows - y_ref[:, :f]).abs().max().item())
        # padded, CUDA graph (captured at i == 0 or after a capacity re-allocation, replayed otherwise)
        if graph is None or regen:
            x_static = torch.zeros(n_local[rank], F, device=dev)
            gw_static = torch.zeros(pad.n_full, F, device=dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                pad.feature_push_impl(x_static, pi)
                pad.feature_push_backward_impl(gw_static, None, pi)
            torch.cuda.current_stream().wait_stream(s)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y_g = pad.feature_push_impl(x_static, pi)
                gin_g = pad.feature_push_backward_impl(gw_static, None, pi)
            captured = True
        else:
            captured = False
        x_static.copy_(x)
        gw_static.copy_(gw_pad)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        e_gf = (y_g[pad.real_rows] - y_ref).abs().max().item()
        e_gb = (gin_g - gin_ref).abs().max().item()
        good = e_fwd == 0 and e_bwd < 1e-6 and e_gf == 0 and e_gb < 1e-6 and e_pool == 0
        ok &= good
        print(f"[rank {rank}] setup {i}: counts {counts} caps {pad.cap.tolist()} n_full pad {pad.n_full} real {n_real} "
              f"realloc {regen} captured {captured} | eager fwd {e_fwd:.1e} bwd {e_bwd:.1e} | graph fwd {e_gf:.1e} "
              f"bwd {e_gb:.1e} | pooled widths {e_pool:.1e} {'OK' if good else 'FAIL'}", flush=True)
    print(f"[rank {rank}] ALL {'OK' if ok else 'FAIL'}", flush=True)
    graph = None
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
