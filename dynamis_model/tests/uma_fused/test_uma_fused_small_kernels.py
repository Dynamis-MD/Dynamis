"""Small kernels of the fused UMA-s path: exact l=2 Wigner (wigner_l2), fused edge-degree embedding (ede_fused),
GPU tile packer (gpu_pack). Synthetic inputs, fp64 references; no model, no captured data."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sm100 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
                           reason="needs an sm_100 GPU (kernels built for sm_100a)")


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# ------------------------------------------------------------------------------------------------ l=2 Wigner
@cuda
def test_wigner_l2_matches_fairchem_einsum_fp64():
    from fairchem.core.models.uma.common.quaternion.wigner_d_custom_kernels import (
        CustomKernelModule, quaternion_to_wigner_d_l2_einsum)
    from dynamis_model.nnp.uma_fused.wigner_l2 import quaternion_to_wigner_d_l2_triton
    C_l2 = CustomKernelModule().C_l2.cuda()
    g = torch.Generator().manual_seed(0)
    q = torch.randn(5000, 4, generator=g, dtype=torch.float64)
    q = (q / q.norm(dim=1, keepdim=True)).cuda()
    dd = torch.randn(5000, 5, 5, generator=g, dtype=torch.float64).cuda()

    q64 = q.clone().requires_grad_(True)
    D64 = quaternion_to_wigner_d_l2_einsum(q64, C_l2.double())
    dq64, = torch.autograd.grad(D64, q64, dd)

    def run():
        q32 = q.float().clone().requires_grad_(True)
        D = quaternion_to_wigner_d_l2_triton(q32, C_l2)
        return D.detach(), torch.autograd.grad(D, q32, dd.float())[0]
    D, dq = run()
    assert D.dtype == torch.float32 and D.shape == (5000, 5, 5)
    assert (D.double() - D64).abs().max().item() < 2e-6        # exact fp32 (no TF32 GEMM): ~1e-7
    assert _rel(dq, dq64) < 1e-5
    # orthogonality of the l=2 block (real Wigner matrices)
    eye = torch.eye(5, device="cuda", dtype=torch.float64)
    assert (D.double() @ D.double().transpose(1, 2) - eye).abs().max().item() < 1e-5
    D2, dq2 = run()
    assert torch.equal(D, D2) and torch.equal(dq, dq2)


# ------------------------------------------------------------------------------------------------ EDE
def _fairchem_ede(x, radial, winv, tgt, rescale):
    r = radial.reshape(-1, 3, x.shape[2])
    e = torch.bmm(winv[:, :, [0, 2, 6]], r).to(x.dtype)
    return x.index_add(0, tgt, e / rescale)


@cuda
def test_edge_degree_embedding_matches_fairchem_fp64():
    from dynamis_model.nnp.uma_fused.ede_fused import edge_degree_scatter_fused
    g = torch.Generator().manual_seed(0)
    N, C = 500, 128
    tgt = torch.repeat_interleave(torch.arange(N), torch.randint(0, 40, (N,), generator=g)).cuda()
    E = tgt.numel()
    x = torch.randn(N, 9, C, generator=g).cuda()
    radial = torch.randn(E, 3 * C, generator=g).cuda()
    winv = torch.randn(E, 9, 9, generator=g).cuda()
    gy = torch.randn(N, 9, C, generator=g).cuda()
    rescale = 5.0

    def run():
        leaves = [t.clone().requires_grad_(True) for t in (x, radial, winv)]
        y = edge_degree_scatter_fused(leaves[0], leaves[1], leaves[2], tgt, 3, C, rescale)
        return [y.detach()] + list(torch.autograd.grad(y, leaves, gy))
    res = run()
    rl = [t.double().clone().requires_grad_(True) for t in (x, radial, winv)]
    yr = _fairchem_ede(rl[0], rl[1], rl[2], tgt, rescale)
    ref = [yr.detach()] + list(torch.autograd.grad(yr, rl, gy.double()))
    for name, a, b in zip(("y", "dx", "dradial", "dwinv"), res, ref):
        assert _rel(a, b) < 1e-5, name            # fp32 FMAs, no TF32: ~1e-7
    res2 = run()
    assert all(torch.equal(a, b) for a, b in zip(res, res2))


# ------------------------------------------------------------------------------------------------ GPU packer
def _check_packing(tgt, perm, slot, tnodes, counts, tile_cap, max_slots=7):
    """validity of a packing: every edge exactly once, whole target segments, <= 128 edges and <= max_slots targets
    per tile, slot / tile_nodes consistent, padding (-1) beyond the used tiles"""
    tgt, perm, slot, tnodes = tgt.cpu(), perm.cpu(), slot.cpu(), tnodes.cpu()
    T, nseg, over, bad = counts.cpu().tolist()
    E = tgt.numel()
    assert over == 0 and bad == 0 and 0 < T <= tile_cap
    assert nseg == int((tgt[1:] != tgt[:-1]).sum()) + 1
    real = perm[perm >= 0]
    assert real.numel() == E and torch.equal(real.sort().values, torch.arange(E))
    assert bool((perm[T * 128:] == -1).all()) and bool((tnodes[T:] == -1).all())
    p = perm[: T * 128].view(T, 128)
    s = slot[: T * 128].view(T, 128)
    for t in range(T):
        m = p[t] >= 0
        n = int(m.sum())
        assert bool(m[:n].all()) and not bool(m[n:].any()), "real edges first in a tile"
        e = p[t][:n]
        assert torch.equal(tnodes[t][s[t][:n].long()], tgt[e].int()), "slot -> tile_nodes -> target"
        used = tnodes[t][tnodes[t] >= 0]
        assert used.numel() <= max_slots and bool((tnodes[t][used.numel():] == -1).all())
        for node in used.tolist():      # whole segment of this target in this tile
            assert int((tgt[e] == node).sum()) == int((tgt == node).sum())
    fill = E / (T * 128)
    return T, fill


@sm100
@pytest.mark.parametrize("seed,N,deg", [(0, 2000, (1, 60)), (1, 300, (60, 128)), (2, 5000, (0, 20))])
def test_gpu_pack_valid(seed, N, deg):
    from dynamis_model.nnp.uma_fused.gpu_pack import pack_node_tiles_gpu
    g = torch.Generator().manual_seed(seed)
    tgt = torch.repeat_interleave(torch.randperm(N, generator=g),
                                  torch.randint(deg[0], deg[1] + 1, (N,), generator=g)).cuda()
    tile_cap = tgt.numel() // 64 + 8
    out = pack_node_tiles_gpu(tgt, tile_cap)
    T, fill = _check_packing(tgt, *out, tile_cap)
    nseg = int(out[3][1])
    assert -(-tgt.numel() // 128) <= T <= nseg and T >= -(-nseg // 7)
    out2 = pack_node_tiles_gpu(tgt, tile_cap)
    assert all(torch.equal(a, b) for a, b in zip(out, out2))


@sm100
def test_gpu_pack_flags_overflow_and_bad_degree():
    from dynamis_model.nnp.uma_fused.gpu_pack import pack_node_tiles_gpu
    tgt = torch.repeat_interleave(torch.arange(100), 50).cuda()
    *_, counts = pack_node_tiles_gpu(tgt, 10)       # needs 50 tiles (2 targets of 50 per tile)
    assert int(counts[2]) == 1
    tgt = torch.repeat_interleave(torch.arange(10), torch.tensor([5] * 9 + [129])).cuda()
    *_, counts = pack_node_tiles_gpu(tgt, 100)
    assert int(counts[3]) == 1
