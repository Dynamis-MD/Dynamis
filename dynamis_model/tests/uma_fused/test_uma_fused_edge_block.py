"""Fused UMA-s edge block (``edge_block.FusedEdgeBlock.op``: radial fold, cap mode) vs the fp64 reference.

Synthetic weights and graphs (no captured data, no model download). Covers forward + backward against autograd of
``reference.edge_block_reference`` in fp64, bitwise reproducibility, a neighbour-list change within capacity (with and
without a topology epoch), capacity growth, and rows that are no edge's target. Needs an sm_100 GPU.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="fused UMA-s kernels need an sm_100 GPU")

C = H = 128


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _weights(seed):
    from dynamis_model.nnp.uma_fused.reference import EdgeBlockWeights
    g = torch.Generator().manual_seed(seed)

    def mat(o, i, s=1.0):
        return (torch.randn(o, i, generator=g, dtype=torch.float64) * s / i ** 0.5)
    def so2(o, i):
        # SO(2) m-block weight [[Wa, -Wb], [Wb, Wa]] (real / imaginary parts), as fairchem's _w_block
        a, b = mat(o // 2, i // 2, 0.5 ** 0.5), mat(o // 2, i // 2, 0.5 ** 0.5)
        return torch.cat((torch.cat((a, -b), 1), torch.cat((b, a), 1)), 0)
    w = EdgeBlockWeights(W0=mat(5 * H, 6 * C), b0=0.1 * torch.randn(5 * H, generator=g, dtype=torch.float64),
                         W1=so2(4 * H, 8 * C), W2=so2(2 * H, 4 * C),
                         V0=mat(3 * C, 3 * H), c0=0.1 * torch.randn(3 * C, generator=g, dtype=torch.float64),
                         V1=so2(4 * C, 4 * H), V2=so2(2 * C, 2 * H), H=H, C=C)
    W_last = mat(1536, 128)
    b_last = 0.1 * torch.randn(1536, generator=g, dtype=torch.float64)
    return w, W_last, b_last


def _blockdiag_rot(E, g):
    """random block-diagonal orthogonal [E, 9, 9] (l = 0, 1, 2 blocks), like a real Wigner D"""
    D = torch.zeros(E, 9, 9, dtype=torch.float64)
    D[:, 0, 0] = 1.0
    for lo, n in ((1, 3), (4, 5)):
        q, _ = torch.linalg.qr(torch.randn(E, n, n, generator=g, dtype=torch.float64))
        D[:, lo:lo + n, lo:lo + n] = q
    return D


def _graph(seed, N, n_tgt, deg=(8, 60), dev="cuda"):
    """edges grouped by target (sorted), targets = a random subset of n_tgt of the N rows (the others: zero rows)"""
    g = torch.Generator().manual_seed(seed)
    tg = torch.randperm(N, generator=g)[:n_tgt].sort().values
    d = torch.randint(deg[0], deg[1] + 1, (n_tgt,), generator=g)
    tgt = torch.repeat_interleave(tg, d)
    E = tgt.numel()
    src = torch.randint(0, N, (E,), generator=g)
    D = _blockdiag_rot(E, g)
    env = torch.rand(E, 1, 1, generator=g, dtype=torch.float64)
    Dinv = env * D.transpose(1, 2)
    x = torch.randn(N, 9, C, generator=g, dtype=torch.float64)
    h = torch.randn(E, 128, generator=g, dtype=torch.float64)
    dout = torch.randn(N, 9, C, generator=g, dtype=torch.float64)
    ei = torch.stack((src, tgt)).to(dev)
    f32 = lambda t: t.float().to(dev)  # noqa: E731
    return dict(ei=ei, x=f32(x), D=f32(D), Dinv=f32(Dinv), h=f32(h), dout=f32(dout), N=N, E=E, tgt=tgt.to(dev))


def _run_op(blk, gr):
    leaves = [gr[k].clone().requires_grad_(True) for k in ("x", "D", "Dinv", "h")]
    out = blk.op(leaves[0], gr["ei"], leaves[1], leaves[2], leaves[3])
    grads = torch.autograd.grad(out, leaves, gr["dout"])
    return [out.detach().clone()] + [t.clone() for t in grads]


def _reference(w64, W_last, b_last, gr):
    from dynamis_model.nnp.uma_fused.reference import edge_block_reference
    dev = gr["x"].device
    w = w64.to(device=dev)
    leaves = [gr[k].double().clone().requires_grad_(True) for k in ("x", "D", "Dinv", "h")]
    radial = leaves[3] @ W_last.to(dev).T + b_last.to(dev)
    out = edge_block_reference(leaves[0], gr["ei"], leaves[1], leaves[2], radial, w)
    grads = torch.autograd.grad(out, leaves, gr["dout"].double())
    return [out.detach()] + list(grads)


@pytest.fixture(scope="module")
def block():
    from dynamis_model.nnp.uma_fused.edge_block import FusedEdgeBlock, set_topology_epoch
    w, W_last, b_last = _weights(0)
    blk = FusedEdgeBlock(w.to(dtype=torch.float32), W_last.float(), b_last.float(), "cuda")
    set_topology_epoch(None)
    yield blk, w, W_last, b_last
    set_topology_epoch(None)


NAMES = ("out", "dx", "dD", "dDinv", "dh")
# measured (synthetic weights, FP16 operands / fp32 accumulation): out ~3e-4, dx ~5e-4, dD ~4e-4, dDinv ~3e-4,
# dh ~6e-4 relative to fp64
TOL = dict(out=2e-3, dx=3e-3, dD=3e-3, dDinv=3e-3, dh=3e-3)


def _check_vs_reference(blk, w, W_last, b_last, gr, res=None):
    res = _run_op(blk, gr) if res is None else res
    ref = _reference(w, W_last, b_last, gr)
    # dD / dDinv: only the block-diagonal (l = 0, 1, 2) entries exist in a Wigner matrix; the op writes zeros
    # elsewhere, the dense fp64 reference has nonzero off-block gradients
    bd = torch.zeros(9, 9, dtype=torch.bool, device=res[0].device)
    for lo, hi in ((0, 1), (1, 4), (4, 9)):
        bd[lo:hi, lo:hi] = True
    for i in (2, 3):
        assert torch.count_nonzero(res[i].view(-1, 9, 9)[:, ~bd]) == 0
        ref[i] = ref[i].view(-1, 9, 9) * bd
    errs = {n: _rel(a, b) for n, a, b in zip(NAMES, res, ref)}
    for n in NAMES:
        assert errs[n] < TOL[n], errs
    # rows of x that are no edge's target get zero messages
    is_tgt = torch.zeros(gr["N"], dtype=torch.bool, device=res[0].device)
    is_tgt[gr["tgt"]] = True
    assert torch.count_nonzero(res[0][~is_tgt]) == 0
    return res, errs


def test_op_matches_fp64_reference_and_is_bitwise_repeatable(block):
    blk, w, W_last, b_last = block
    gr = _graph(1, N=700, n_tgt=600)
    res, errs = _check_vs_reference(blk, w, W_last, b_last, gr)
    print("rel errors vs fp64:", {k: f"{v:.2e}" for k, v in errs.items()})
    res2 = _run_op(blk, gr)
    for n, a, b in zip(NAMES, res, res2):
        assert torch.equal(a, b), f"{n} not bitwise reproducible"


def test_neighbour_list_change_within_capacity(block):
    """a new neighbour list (fewer edges, other targets) is re-packed in place: no new capacity / generation"""
    from dynamis_model.nnp.uma_fused import edge_block as eb
    from dynamis_model.nnp.uma_fused.edge_block import set_topology_epoch
    blk, w, W_last, b_last = block
    dev = torch.device("cuda")
    _check_vs_reference(blk, w, W_last, b_last, _graph(1, N=700, n_tgt=600))
    gen = eb._CAP[dev]["gen"]
    epoch = iter(range(1, 100))

    def new_topology():
        set_topology_epoch(next(epoch))

    try:
        # epoch-declared topology (the MD driver's path: no identity / content check)
        new_topology()
        gr2 = _graph(2, N=650, n_tgt=500, deg=(5, 50))
        _check_vs_reference(blk, w, W_last, b_last, gr2)
        new_topology()
        gr3 = _graph(3, N=700, n_tgt=550, deg=(5, 55))
        _check_vs_reference(blk, w, W_last, b_last, gr3)
        assert eb._CAP[dev]["gen"] == gen, "capacity re-allocated for a list that fits"
    finally:
        set_topology_epoch(None)
    # back to identity-based detection: a fresh edge_index tensor is re-packed
    _check_vs_reference(blk, w, W_last, b_last, _graph(4, N=700, n_tgt=600, deg=(8, 50)))
    assert eb._CAP[dev]["gen"] == gen


def test_capacity_growth(block):
    """more edges / nodes than the capacity -> re-allocation (new generation), results still exact"""
    from dynamis_model.nnp.uma_fused import edge_block as eb
    blk, w, W_last, b_last = block
    dev = torch.device("cuda")
    _check_vs_reference(blk, w, W_last, b_last, _graph(1, N=700, n_tgt=600))
    cs = eb._CAP[dev]
    gen, E_cap, N_cap = cs["gen"], cs["E_cap"], cs["N_cap"]
    gr = _graph(5, N=N_cap + 200, n_tgt=N_cap + 100, deg=(30, 90))
    assert gr["E"] > E_cap
    _check_vs_reference(blk, w, W_last, b_last, gr)
    cs = eb._CAP[dev]
    assert cs["gen"] == gen + 1 and cs["E_cap"] > E_cap and cs["N_cap"] > N_cap
    # and back down to a small list: stays at the grown capacity
    _check_vs_reference(blk, w, W_last, b_last, _graph(1, N=700, n_tgt=600))
    assert eb._CAP[dev]["gen"] == gen + 1


def test_degree_above_tile_size_is_reported(block):
    """a target with > 128 edges cannot be packed: the lazy capacity check raises at the next rebuild"""
    from dynamis_model.nnp.uma_fused import edge_block as eb
    blk, w, W_last, b_last = block
    gr = _graph(6, N=300, n_tgt=200, deg=(129, 140))
    _run_op(blk, gr)
    with pytest.raises(ValueError):
        eb._cap_check(eb._CAP[torch.device("cuda")])
    _check_vs_reference(blk, w, W_last, b_last, _graph(1, N=700, n_tgt=600))
