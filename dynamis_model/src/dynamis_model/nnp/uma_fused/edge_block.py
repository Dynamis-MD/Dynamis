"""Fused UMA-s edge-wise block (lmax = mmax = 2, C = H = 128, frozen weights), capacity ("cap") CUDA-graph mode.

    blk = FusedEdgeBlock(EdgeBlockWeights.from_edgewise(edgewise), W_last, b_last)
    out = blk.op(x, edge_index, wigner, wigner_inv_env, h)     # torch.library custom op dynamis::uma_edge_block

Inputs: x [N, 9, C] fp32 (N may include ghost rows), edge_index [2, E] grouped by target (contiguous segments),
wigner / wigner_inv_env [E, 9, 9] fp32 (block diagonal; inv_env = envelope * D^T), h [E, 128] = the radial MLP's last
hidden layer (its last Linear W_last [1536, 128], b_last is folded into K0r). Output [N, 9, C] fp32 messages
(zero rows for nodes that are no edge's target); backward gives dx, d wigner, d wigner_inv_env, dh (block-diagonal
entries; others exactly zero).

Forward (per block, one CUDA graph):  K0r (CuTe, k0r_cute)   radial = h W_last^T + b, A = radial * D x[src|tgt]
                                      K1  (CuTe, k1_cute)    conv1 GEMMs + bias + gate -> zp / g (FP16, saved)
                                      K2  (CuTe, k2_cute)    act(zp) * g, conv2, D^-1 rotation, node sum
Backward (one CUDA graph):            S0+S1 (s1_mma)       dOut . V per node, dz / dh / dgate / dDinv per edge
                                      S2  (cuBLAS FP16)      da = dh . W
                                      S3  (node_geo_cu)      dx, dD, d radial (FP16 packed) ; dh = d radial . W_last
FP16 operands, fp32 accumulation; fixed reduction orders, no atomics -> bitwise reproducible.

Capacity mode: the neighbour list is packed on the GPU (gpu_pack.GPUTopology) into 128-edge, <= 7-target tiles in
buffers of fixed capacity (tiles, nodes, edges), rebuilt in place without host sync whenever the list changes;
padding tiles have Dinv = 0 and contribute nothing. Kernel shapes and buffer addresses depend on the capacity only ->
one compile and one graph capture per capacity; E > E_cap or N > N_cap re-allocates (new generation).
"""
import os
import weakref

import torch
import triton
import triton.language as tl

from dynamis_model.nnp.uma_fused.common import ilv_alloc
from dynamis_model.nnp.uma_fused.reference import EdgeBlockWeights

# ------------------------------------------------------------------------------------------------ module state
# neighbour-list generation declared by the caller (dynamis_fused.new_topology); None = detect by tensor identity
_TOPO_EPOCH = [None]
# token of the current OUTER CUDA-graph capture (graphed_predict.GraphedPredict bumps it before each capture): inside
# one outer capture the first block's geometry pack is recorded into the graph and the later blocks may reuse it
CAPTURE_TOKEN = [None]
GEO_STATS = {"hit": 0, "miss": 0}

CAP_HEADROOM = float(os.environ.get("DYNAMIS_UMA_CAP_HEADROOM", "1.15"))
# node-capacity headroom: multi-GPU MD regrows the padded ghost capacity (~+10 %); uma.py raises it to 1.5 there
CAP_N_HEADROOM = float(os.environ.get("DYNAMIS_UMA_CAP_N_HEADROOM", str(CAP_HEADROOM)))
_CAP = {}                  # device -> shared capacity state (topology + per-edge buffer sizes)

# one CUDA-graph memory pool shared by all blocks' graphs above this many packed edges (large systems: per-block pools
# would hold L copies of the transients). DYNAMIS_UMA_SHARED_POOL: auto (default) | 1 always | 0 never
_SHARED_POOLS = {}
_SHARED_POOL_MIN_EP = int(os.environ.get("DYNAMIS_UMA_SHARED_POOL_MIN_EP", "1000000"))


def set_topology_epoch(epoch):
    """Declare the neighbour-list generation (a counter bumped at every rebuild). While an epoch is set, a new list is
    detected by the epoch alone (no tensor identity / version check). None = identity check."""
    _TOPO_EPOCH[0] = epoch


# ------------------------------------------------------------------------------------------------ weights / block
_REGISTRY = []


class FusedEdgeBlock:
    """One fused UMA-s edge-wise block (cap mode). ``op`` is the only entry point."""

    def __init__(self, w: EdgeBlockWeights, W_last, b_last, device="cuda"):
        from dynamis_model.nnp.uma_fused.k0r_cute import permute_w
        from dynamis_model.nnp.uma_fused.k1_cute import K1
        from dynamis_model.nnp.uma_fused.s1_mma import S1Weights, dh_perm_weights
        self.device = dev = torch.device(device)
        assert tuple(W_last.shape) == (1536, 128) and tuple(b_last.shape) == (1536,)
        self.C, self.H = w.C, w.H

        # K2 arguments: conv2 weights (M-row blocks) and biases
        self.weights = dict(
            v1=w.V1.to(dev).half().contiguous(),
            v2=w.V2.to(dev).half().contiguous(),
            v0=w.V0.to(dev).half().contiguous(),
            b0=w.b0.to(dev).float().contiguous(),
            c0=w.c0.to(dev).float().contiguous(),
        )
        self.num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
        # K0r: W_last rows in MMA-unit order; K1: conv1 weights
        self.wperm16, self.bperm = permute_w(W_last.detach().to(dev).float(), b_last.detach().to(dev).float())
        self.k1 = K1(w.W0.to(dev), w.W1.to(dev), w.W2.to(dev), w.b0.to(dev))
        # backward: S1 weights, S2 FP16 weights (rows permuted to S1's dh column order), W_last for the dh GEMM
        self.s1w = S1Weights(w, dev)
        self.W16 = [w.W0.to(dev).half().contiguous(), w.W1.to(dev).half().contiguous(),
                    w.W2.to(dev).half().contiguous()]
        self.W16 = dh_perm_weights(self.W16)
        self.wl16 = W_last.detach().to(dev).half().contiguous()        # [1536, 128]
        self._hbuf = None      # h staged at a fixed address (K0r input inside the graph)
        self._gs = None        # per-capacity state: static buffers + captured graphs
        self._stash = None
        self.block_id = len(_REGISTRY)
        _REGISTRY.append(self)   # eagerly: registry side effects inside torch.compile tracing are lost

    def op(self, x, edge_index, wigner, wigner_inv_env, h):
        return torch.ops.dynamis.uma_edge_block(x, edge_index, wigner, wigner_inv_env, h, self.block_id)


# ------------------------------------------------------------------------------------------------ capacity topology
def _cap_topology(dev, edge_index, N):
    from dynamis_model.nnp.uma_fused.gpu_pack import GPUTopology, pack_node_tiles_gpu
    E = edge_index.shape[1]
    cs = _CAP.get(dev)
    if cs is None or E > cs["E_cap"] or N > cs["N_cap"]:
        E_cap = int(E * CAP_HEADROOM) + 1
        N_cap = int(N * CAP_N_HEADROOM) + 1
        # (re)allocation only: pack once at a safe capacity (<= one tile per segment) and read the tile count
        # (the only host sync of the capacity path); tile capacity = measured tiles * headroom
        T = int(pack_node_tiles_gpu(edge_index[1], N + 1, seg_cap=N)[3][0])
        tile_cap = int(T * CAP_HEADROOM) + 1
        cs = dict(E_cap=E_cap, N_cap=N_cap, gen=(cs["gen"] + 1 if cs else 0),
                  topo=GPUTopology(tile_cap, N_cap, dev), ei=None, ver=None, ep=None, pending=None, upd=0,
                  geo=None)
        _CAP[dev] = cs
    ep = _TOPO_EPOCH[0]
    same = (cs["ei"] is edge_index and cs["ver"] == edge_index._version) or (ep is not None and cs["ep"] == ep)
    if not same or cs["topo"].d is None:
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            _cap_check(cs)
        cs["topo"].update(edge_index)
        cs["ei"], cs["ver"], cs["ep"] = edge_index, edge_index._version, ep
        cs["upd"] += 1
        if not capturing:
            # lazy capacity check: counts copied to pinned memory now, verified at the next rebuild (no sync here).
            # Inside an outer capture (torch.compile reduce-overhead / GraphedPredict) no check / pinned copy.
            hc = torch.empty(4, dtype=torch.int32, pin_memory=True)
            hc.copy_(cs["topo"].d["counts"], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            cs["pending"] = (hc, ev)
    return cs


def _cap_check(cs):
    if cs.get("pending") is None:
        return
    hc, ev = cs["pending"]
    ev.synchronize()
    cs["pending"] = None
    if hc[3]:
        raise ValueError("a target degree exceeds the tile size 128" if hc[3] == 1 else
                         "edges not grouped by target / more target segments than the node capacity")
    if hc[2]:
        raise OverflowError(f"tile capacity {cs['topo'].tile_cap} exceeded ({int(hc[0])} tiles): results of the "
                            "previous neighbour list are invalid; raise DYNAMIS_UMA_CAP_HEADROOM")


def _cap_geo(cs, wigner, wigner_inv_env):
    """Packed wigner `d` / masked `dinv` at capacity, shared by all blocks on the device: packed by the first block of
    a predict, reused by the others (their graphs read the same fixed-address buffers). Valid while the wigner tensors
    are the same objects (weakrefs) with unchanged _version and the topology was not rebuilt (cs["upd"]); the backward
    of every block reads them too (a predict's backward runs before the next predict's first forward). Inside an outer
    capture the buffers are re-packed unless CAPTURE_TOKEN identifies the capture."""
    geo = cs["geo"]
    if geo is None:
        Ep = cs["topo"].d["slot"].numel()
        dev = wigner.device
        geo = cs["geo"] = dict(d=torch.empty(Ep, 81, device=dev, dtype=torch.float32),
                               dinv=torch.empty(Ep, 81, device=dev, dtype=torch.float32), key=None)
    key = (cs["upd"], wigner._version, wigner_inv_env._version)
    capturing = torch.cuda.is_current_stream_capturing()
    tok = CAPTURE_TOKEN[0] if capturing else None
    if (geo["key"] == key and geo.get("tok") == tok and (not capturing or tok is not None)
            and geo["w"]() is wigner and geo["wi"]() is wigner_inv_env):
        GEO_STATS["hit"] += 1
        return geo
    topo = cs["topo"].d
    E = wigner.shape[0]
    torch.index_select(wigner.reshape(E, 81), 0, topo["pc"], out=geo["d"])
    torch.index_select(wigner_inv_env.reshape(E, 81), 0, topo["pc"], out=geo["dinv"])
    geo["dinv"].masked_fill_(topo["padz"].view(-1, 1), 0.0)
    if capturing and tok is None:
        geo["key"] = None
    else:
        geo.update(key=key, tok=tok, w=weakref.ref(wigner), wi=weakref.ref(wigner_inv_env))
        GEO_STATS["miss"] += 1
    return geo


# ------------------------------------------------------------------------------------------------ graphs / pools
def _pool_for(dev, Ep):
    mode = os.environ.get("DYNAMIS_UMA_SHARED_POOL", "auto")
    if mode == "1" or (mode == "auto" and Ep > _SHARED_POOL_MIN_EP):
        key = torch.device(dev).index if torch.device(dev).index is not None else torch.cuda.current_device()
        if key not in _SHARED_POOLS:
            _SHARED_POOLS[key] = torch.cuda.graph_pool_handle()
        return _SHARED_POOLS[key], True
    return torch.cuda.graph_pool_handle(), False


def release_inner_graphs(blocks):
    """Drop the per-block CUDA graphs (+ their pool) captured by an eager warm-up. GraphedPredict calls this for large
    systems right before the whole-call capture (which records the kernels directly): otherwise the warm-up's inner
    pool, the allocator cache and the outer pool all hold the per-layer transients. Re-captured lazily if needed."""
    n = 0
    for blk in blocks:
        gs = getattr(blk, "_gs", None)
        if gs is None:
            continue
        for k in ("fwd", "fwd_out", "bwd", "bwd_out"):
            if gs.pop(k, None) is not None:
                n += 1
        if gs.get("shared_pool"):   # a pool whose last graph died cannot be captured into again: fresh handle
            gs["pool"] = None
    _SHARED_POOLS.clear()
    for blk in blocks:
        gs = getattr(blk, "_gs", None)
        if gs is not None and gs.get("pool", 0) is None:
            gs["pool"], gs["shared_pool"] = _pool_for(blk.device, gs["topo"]["slot"].numel())
    return n


def _capture(gs, fn):
    """warm up (compiles outside the capture), then capture fn into a CUDA graph"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    if gs.get("shared_pool"):     # hand the warm-up transients back before the pool grows (large systems)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg, pool=gs["pool"]):
        res = fn()
    return cg, res


# ------------------------------------------------------------------------------------------------ forward
def _fwd_body(blk, gs):
    """K0r -> K1 -> K2 on the static buffers of gs (captured into the block's forward graph)"""
    from dynamis_model.nnp.uma_fused.k0r_cute import k0r_cute
    from dynamis_model.nnp.uma_fused.k2_cute import k2_cute
    g = gs["g"]
    x16 = gs["x32"].half()
    Ep = g["slot"].numel()
    cnt = g["counts"]        # device tile count: K0r / K1 / K2 skip the padding tiles
    A = torch.empty(Ep, 2304, device=blk.device, dtype=torch.float16)
    k0r_cute(blk.wperm16, blk.bperm, blk._hbuf, g["pc"], g["src"], g["tgt"], g["d"], x16, A, g["r"], cnt,
             blk.num_ctas)
    sz = ilv_alloc(Ep, 9 * blk.H, torch.float16, blk.device)
    sg = ilv_alloc(Ep, 2 * blk.H, torch.float16, blk.device)
    blk.k1(A, sz, sg, cnt)
    del A
    out = torch.zeros(gs["x32"].shape[0], 9, blk.C, device=blk.device)
    wt = blk.weights
    k2_cute(wt["v1"], wt["v2"], wt["v0"], wt["b0"], wt["c0"], g["dinv"], g["slot"], g["tile_nodes"], out, cnt, sz, sg,
            blk.num_ctas)
    return out, sz, sg


def _fwd(blk, x, edge_index, wigner, wigner_inv_env, h):
    dev = blk.device
    N, E = x.shape[0], edge_index.shape[1]
    cs = _cap_topology(dev, edge_index, N)
    topo = cs["topo"].d
    geo = _cap_geo(cs, wigner, wigner_inv_env)
    gs = blk._gs
    if gs is None or gs["key"] != ("cap", cs["gen"]):
        Ep = topo["slot"].numel()
        gs = dict(key=("cap", cs["gen"]), topo=topo,
                  x32=torch.zeros(cs["N_cap"], *x.shape[1:], device=dev, dtype=torch.float32),
                  d=geo["d"], dinv=geo["dinv"],      # shared by all blocks (fixed addresses per capacity)
                  r=torch.empty(Ep, 1536, device=dev, dtype=torch.float16),
                  dout=torch.zeros(cs["N_cap"], 9, blk.C, device=dev, dtype=torch.float32))
        gs["pool"], gs["shared_pool"] = _pool_for(dev, Ep)
        gs["g"] = dict(topo, d=gs["d"], dinv=gs["dinv"], r=gs["r"])
        blk._gs = gs
    gs["x32"][:N].copy_(x)
    n = gs["r"].shape[0]     # K0r reads h at a fixed address
    if blk._hbuf is None or blk._hbuf.shape[0] != n:
        blk._hbuf = torch.zeros(n, 128, device=dev, dtype=torch.float32)
    blk._hbuf[:h.shape[0]].copy_(h.detach())
    if torch.cuda.is_current_stream_capturing():
        out, sz, sg = _fwd_body(blk, gs)     # outer capture: record the kernels directly
    else:
        if "fwd" not in gs:
            gs["fwd"], gs["fwd_out"] = _capture(gs, lambda: _fwd_body(blk, gs))
        gs["fwd"].replay()
        out, sz, sg = gs["fwd_out"]
    state = dict(gs=gs, save=(sz, sg), N=N, E=E, shapes=(wigner.shape, wigner_inv_env.shape, h.shape))
    return out[:N], state


# ------------------------------------------------------------------------------------------------ backward
@triton.jit
def _dh_gemm(dr_ptr, w_ptr, perm_ptr, out_ptr, BM: tl.constexpr, BK: tl.constexpr):
    """out[perm[p], :] = dr[p, :] @ W  (dr FP16 [Ep,1536] packed, W FP16 [1536,128]); rows with perm < 0 skipped"""
    pid = tl.program_id(0)
    rows = (pid * BM + tl.arange(0, BM)).to(tl.int64)
    n = tl.arange(0, 128)
    acc = tl.zeros([BM, 128], dtype=tl.float32)
    for k0 in range(0, 1536, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(dr_ptr + rows[:, None] * 1536 + k[None, :])
        b = tl.load(w_ptr + k[:, None] * 128 + n[None, :])
        acc = tl.dot(a, b, acc)
    p = tl.load(perm_ptr + rows).to(tl.int64)
    tl.store(out_ptr + p[:, None] * 128 + n[None, :], acc, mask=(p >= 0)[:, None])


def dh_gemm(drad16, wl16, perm, E, BM=64, BK=64, nw=4):
    Ep = drad16.shape[0]
    assert Ep % BM == 0 and drad16.is_contiguous() and wl16.is_contiguous()
    out = torch.empty(E, 128, device=drad16.device, dtype=torch.float32)
    _dh_gemm[(Ep // BM,)](drad16, wl16, perm, out, BM=BM, BK=BK, num_warps=nw)
    return out


def _bwd_body(blk, gs, sz, sg, num_out):
    """S0/S1 -> S2 -> S3 + dh GEMM; per-edge outputs [num_out, ...] in original edge order (capacity rows)"""
    from dynamis_model.nnp.uma_fused.node_geo_cu import node_geo_cu
    from dynamis_model.nnp.uma_fused.s1_mma import s1_mma
    g, x, dOut = gs["g"], gs["x32"], gs["dout"]
    dev = blk.device
    Ep = g["slot"].numel()
    orig = g["orig"]
    dh, ddinv = s1_mma(blk.s1w, sz, sg, g["dinv"], g["tgt"], dOut, orig, num_out, g["counts"])
    da = [torch.empty(Ep, n, device=dev, dtype=torch.float16) for n in (768, 1024, 512)]
    flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        torch.mm(dh[:, :640], blk.W16[0], out=da[0])
        torch.mm(dh[:, 640:1152], blk.W16[1], out=da[1])
        torch.mm(dh[:, 1152:], blk.W16[2], out=da[2])
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
    # S3: dradial FP16 in packed row order, then dh = dradial . W_last straight into the original edge order
    dx, dD, drad16 = node_geo_cu(x, g["d"], da, g["r"], g["csr"], orig, num_out)
    return dx, dD, ddinv, dh_gemm(drad16, blk.wl16, orig, num_out)


def _bwd(blk, state, dOut):
    gs, N, E = state["gs"], state["N"], state["E"]
    gs["dout"][:N].copy_(dOut)
    sz, sg = state["save"]
    Ec = gs["d"].shape[0]     # per-edge outputs at capacity rows (>= E_cap); views [:E] returned
    if torch.cuda.is_current_stream_capturing():
        dx, dD, ddinv, dh = _bwd_body(blk, gs, sz, sg, Ec)
    else:
        if "bwd" not in gs:
            gs["bwd"], gs["bwd_out"] = _capture(gs, lambda: _bwd_body(blk, gs, sz, sg, Ec))
        gs["bwd"].replay()
        dx, dD, ddinv, dh = gs["bwd_out"]
    ws, wis, hs = state["shapes"]
    return dx[:N], dD[:E].view(ws), ddinv[:E].view(wis), dh[:E].view(hs)


# ------------------------------------------------------------------------------------------------ custom ops
# Packed sizes are data dependent, so the saved activations are not op outputs: the forward op stashes its state on
# the block and the backward op consumes it (one forward per block between backwards: true for inference / MD, not
# for activation checkpointing).
@torch.library.custom_op("dynamis::uma_edge_block", mutates_args=())
def uma_edge_block(x: torch.Tensor, edge_index: torch.Tensor, wigner: torch.Tensor,
                   wigner_inv_env: torch.Tensor, radial: torch.Tensor, block_id: int) -> torch.Tensor:
    blk = _REGISTRY[block_id]
    out, blk._stash = _fwd(blk, x, edge_index, wigner, wigner_inv_env, radial)
    return out


@uma_edge_block.register_fake
def _(x, edge_index, wigner, wigner_inv_env, radial, block_id):
    return x.new_empty(x.shape[0], 9, x.shape[2], dtype=torch.float32)


@torch.library.custom_op("dynamis::uma_edge_block_bwd", mutates_args=())
def uma_edge_block_bwd(dout: torch.Tensor, x: torch.Tensor, wigner: torch.Tensor, wigner_inv_env: torch.Tensor,
                       radial: torch.Tensor, block_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                     torch.Tensor]:
    blk = _REGISTRY[block_id]
    state, blk._stash = blk._stash, None
    assert state is not None, "uma_edge_block backward without a pending forward"
    return _bwd(blk, state, dout)


@uma_edge_block_bwd.register_fake
def _(dout, x, wigner, wigner_inv_env, radial, block_id):
    return (torch.empty_like(x), torch.empty_like(wigner), torch.empty_like(wigner_inv_env),
            torch.empty_like(radial))


def _setup_context(ctx, inputs, output):
    x, edge_index, wigner, wigner_inv_env, radial, block_id = inputs
    ctx.save_for_backward(x, wigner, wigner_inv_env, radial)
    ctx.block_id = block_id


def _backward(ctx, dout):
    x, wigner, wigner_inv_env, radial = ctx.saved_tensors
    dx, dw, dwi, dr = uma_edge_block_bwd(dout, x, wigner, wigner_inv_env, radial, ctx.block_id)
    return dx, None, dw, dwi, dr, None


uma_edge_block.register_autograd(_backward, setup_context=_setup_context)
