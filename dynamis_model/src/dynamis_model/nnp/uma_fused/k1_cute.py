"""CuTe DSL (sm100) K1 of the fused forward: conv1 GEMMs + bias + gate as a paired-CTA GEMM.

    k1 = K1(W0, W1, W2, b0)        # raw EdgeBlockWeights matrices
    k1(A, save_z, save_g, cnt=None)  # A = K0r output [Ep, 2304] FP16; save_z / save_g contract ilv FP16

Cluster of 2 CTAs, cta_group::2 tcgen05 MMAs with M = 256 (CTA v owns the edges of 128-tile 2*pair + v), each CTA
streams half of every 256-row W tile (2-CTA TMA) -> per-SM weight bytes halved vs a 1-CTA GEMM.
m1 conv1 is SO(2)-deduplicated: W1 = [[Wa, -Wb], [Wb, Wa]] (Wa/Wb [256, 512]); only Wa / Wb are streamed:
    acc_re += A_re Wa^T + A_im (-Wb)^T (NEGATE_B),  acc_im += A_im Wa^T + A_re Wb^T.
Passes per pair (W stage = 256 rows x 64 K, A stage = 128 edges x 64 K per CTA, both by TMA):
    P0 m0 gate | z0a  (TMEM regions 0 | 1)  12 k-steps x (1 A, 2 W)    -> g cols 0-255 (sigmoid), z 0-255
    P1 m0 z0b + pad   (region 0)            12 k-steps x (1 A, 1 W)    -> z 256-383
    P2 m1 re | im     (regions 0 | 1)        8 k-steps x (2 A, 2 W)    -> z 384-639 | 640-895
    P3 m2             (region 0)             8 k-steps x (1 A, 1 W)    -> z 896-1151
Warps: 0..4*EPI_WG-1 epilogue (lane = edge, TMEM -> +bias (-> sigmoid) -> FP16 -> 16 B ilv stores),
then MMA (leader CTA issues), then TMA (A evict-first, W evict-last).
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import cuda.bindings.driver as cuda

from dynamis_model.nnp.uma_fused.common import CP_EVICT_FIRST, CP_EVICT_LAST, KERNELS

F16, F32 = cutlass.Float16, cutlass.Float32
TILE_K, TILE_KB = 64, 16
A_STAGES = 8      # A ring stages (128 edges x 64 K per CTA)
W_STAGES = 4      # W ring stages (256 rows x 64 K, split over the CTA pair)
EPI_WG = 4        # epilogue warpgroups
SUBW = 16         # columns per epilogue TMEM load
EPI_WARPS = 4 * EPI_WG
MMA_WARP = EPI_WARPS
TMA_WARP = EPI_WARPS + 1          # A and W loads
NTHREADS = 32 * (EPI_WARPS + 2)
NW_STAGES = 12 + 12 + 6 + 8 + 8 + 4   # W stages per pair (each = two 128-row N=128 B tiles)
# passes: (kind, TMEM region) ; kinds: 0 plain N=256 as two N=128 halves (gate / z0a), 1 z0b (one N=128 half per
# k-step, 2 k-steps per W stage), 2 SO(2) dedup (re | im halves of the region; W stage = [Wa | Wb] 128 rows each)
_PASS_DEFS = {   # name: (kind, trips, A col bases, W stages, epilogue segments)
    "gate": (0, 12, (0,), 12, ((0, 256, 0, 0, 1, 0),)),
    "z0a": (0, 12, (0,), 12, ((0, 256, 1, 0, 0, 256),)),
    "z0b": (1, 6, (0,), 6, ((0, 128, 1, 256, 0, 512),)),
    "m1h0": (2, 8, (12, 20), 8, ((0, 128, 1, 384, 0, -1), (128, 128, 1, 640, 0, -1))),
    "m1h1": (2, 8, (12, 20), 8, ((0, 128, 1, 512, 0, -1), (128, 128, 1, 768, 0, -1))),
    "m2": (2, 4, (28, 32), 4, ((0, 128, 1, 896, 0, -1), (128, 128, 1, 1024, 0, -1))),
}
PASS_NAMES = ("gate", "z0a", "z0b", "m1h0", "m1h1", "m2")
PASSES = tuple((_PASS_DEFS[n][0], i % 2) for i, n in enumerate(PASS_NAMES))
P_NK = tuple(_PASS_DEFS[n][1] for n in PASS_NAMES)
P_W0 = tuple(sum(_PASS_DEFS[m][3] for m in PASS_NAMES[:i]) for i in range(len(PASS_NAMES)))
P_ACOL = tuple(_PASS_DEFS[n][2] for n in PASS_NAMES)
EPI_SEGS = tuple(_PASS_DEFS[n][4] for n in PASS_NAMES)
# A loads without an evict-first hint = k-tiles that are read again by a later pass of the same pair
_last = {}
for _i, _n in enumerate(PASS_NAMES):
    _last[_PASS_DEFS[_n][2]] = _i
P_AKEEP = tuple(_last[_PASS_DEFS[n][2]] != i for i, n in enumerate(PASS_NAMES))


@cute.struct
class SharedStorage:
    a_mbar: cute.struct.MemRange[cutlass.Int64, A_STAGES * 2]
    w_mbar: cute.struct.MemRange[cutlass.Int64, W_STAGES * 2]
    acc0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    acc1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def k1c_kernel(mma: cute.TiledMma, mma1: cute.TiledMma, tma_a: cute.CopyAtom, mA: cute.Tensor,
               tma_w: cute.CopyAtom, mW: cute.Tensor, mBias: cute.Tensor, mZ: cute.Tensor, mG: cute.Tensor,
               mCnt: cute.Tensor, ntile_max: cutlass.Int32,
               a_l: cute.ComposedLayout, w_l: cute.ComposedLayout, vmnk: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    v = bidx % 2
    is_leader = v == 0
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    coord = vmnk.get_flat_coord(cta_rank)

    if warp_idx == TMA_WARP:
        cpasync.prefetch_descriptor(tma_a)
        cpasync.prefetch_descriptor(tma_w)

    smem = utils.SmemAllocator()
    st = smem.allocate(SharedStorage)
    one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    a_bytes = cute.size_in_bytes(F16, cute.select(a_l, mode=[0, 1, 2])) * 2
    w_bytes = cute.size_in_bytes(F16, cute.select(w_l, mode=[0, 1, 2])) * 2 * 2   # 2 halves x 2 CTAs
    a_prod, a_cons = pipeline.PipelineTmaUmma.create(
        num_stages=A_STAGES, producer_group=one, consumer_group=one, tx_count=a_bytes,
        barrier_storage=st.a_mbar.data_ptr(), cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
    w_prod, w_cons = pipeline.PipelineTmaUmma.create(
        num_stages=W_STAGES, producer_group=one, consumer_group=one, tx_count=w_bytes,
        barrier_storage=st.w_mbar.data_ptr(), cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
    gacc = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * EPI_WARPS)
    acc0_prod, acc0_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=gacc,
        barrier_storage=st.acc0_mbar.data_ptr(), cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
    acc1_prod, acc1_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=gacc,
        barrier_storage=st.acc1_mbar.data_ptr(), cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
    alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (EPI_WARPS + 1))
    epi_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * EPI_WARPS)
    tmem = utils.TmemAllocator(st.tmem_holding_buf.ptr, barrier_for_retrieve=alloc_bar, allocator_warp_id=0,
                               is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc_mbar.ptr)
    pipeline_init_arrive(cluster_shape_mn=(2, 1), is_relaxed=True)

    sA = smem.allocate_tensor(F16, a_l.outer, 128, a_l.inner)
    sW = smem.allocate_tensor(F16, w_l.outer, 128, w_l.inner)
    a_mask = cpasync.create_tma_multicast_mask(vmnk, coord, mcast_mode=2)
    w_mask = cpasync.create_tma_multicast_mask(vmnk, coord, mcast_mode=1)
    thr = mma.get_slice(v)
    gA = cute.local_tile(mA, (256, TILE_K), (None, None))
    tAsA, tAgA = cpasync.tma_partition(tma_a, coord[2], cute.make_layout(cute.slice_(vmnk, (0, 0, None, 0)).shape),
                                       cute.group_modes(sA, 0, 3), cute.group_modes(thr.partition_A(gA), 0, 3))
    gW = cute.local_tile(mW, (128, TILE_K), (None, None))
    tWsW, tWgW = cpasync.tma_partition(tma_w, coord[1], cute.make_layout(cute.slice_(vmnk, (0, None, 0, 0)).shape),
                                       cute.group_modes(sW, 0, 3), cute.group_modes(thr.partition_B(gW), 0, 3))
    tCrA = mma.make_fragment_A(sA)
    tCrW = mma.make_fragment_B(sW)

    pipeline_init_wait(cluster_shape_mn=(2, 1))

    num_tiles = cutlass.min(mCnt[0], ntile_max)
    npairs = (num_tiles + 1) // 2
    cid = bidx // 2
    gdim, _, _ = cute.arch.grid_dim()
    ncl = gdim // 2

    # ============================ TMA warp (A and W loads) ============================
    if warp_idx == TMA_WARP:
        for pt in cutlass.range(cid, npairs, ncl, unroll=1):
            for p in cutlass.range_constexpr(len(PASSES)):
                kind = PASSES[p][0]
                for k in cutlass.range(P_NK[p], unroll=1):
                    if cutlass.const_expr(kind == 1):
                        acols = (2 * k, 2 * k + 1)
                    else:
                        acols = tuple(c0 + k for c0 in P_ACOL[p])
                    for ac in acols:
                        ae = a_prod.acquire_and_advance()
                        if cutlass.const_expr(P_AKEEP[p]):   # read again by a later pass: no evict-first hint
                            cute.copy(tma_a, tAgA[(None, pt, ac)], tAsA[(None, ae.index)],
                                      tma_bar_ptr=ae.barrier, mcast_mask=a_mask)
                        else:
                            cute.copy(tma_a, tAgA[(None, pt, ac)], tAsA[(None, ae.index)],
                                      tma_bar_ptr=ae.barrier, mcast_mask=a_mask, cache_policy=cutlass.Int64(CP_EVICT_FIRST))
                    we = w_prod.acquire_and_advance()
                    wi = P_W0[p] + k
                    for h in cutlass.range_constexpr(2):
                        cute.copy(tma_w, tWgW[(None, 2 * wi + h, 0)], tWsW[(None, 2 * we.index + h)],
                                  tma_bar_ptr=we.barrier, mcast_mask=w_mask, cache_policy=cutlass.Int64(CP_EVICT_LAST))
        a_prod.tail()
        w_prod.tail()

    # ============================ MMA warp (leader issues) ============================
    if warp_idx == MMA_WARP:
        tmem.wait_for_alloc()
        tptr = tmem.retrieve_ptr(F32)
        accl = mma.make_fragment_C(mma.partition_shape_C((256, 128))).layout
        accs = [cute.make_tensor(tptr + c, accl) for c in (0, 128, 256, 384)]
        if is_leader:
            for pt in cutlass.range(cid, npairs, ncl, unroll=1):
                for p in cutlass.range_constexpr(len(PASSES)):
                    kind, rg = PASSES[p]
                    lo, hi = accs[2 * rg], accs[2 * rg + 1]
                    if cutlass.const_expr(rg == 0):
                        ee = acc0_prod.acquire_and_advance()
                    else:
                        ee = acc1_prod.acquire_and_advance()
                    for k in cutlass.range(P_NK[p], unroll=1):
                        if cutlass.const_expr(kind == 0):
                            af = a_cons.wait_and_advance()
                            wf = w_cons.wait_and_advance()
                            for kb in cutlass.range_constexpr(4):
                                mma.set(tcgen05.Field.ACCUMULATE, (k + kb) > 0)
                                cute.gemm(mma, lo, tCrA[(None, None, kb, af.index)], tCrW[(None, None, kb, 2 * wf.index)], lo)
                                cute.gemm(mma, hi, tCrA[(None, None, kb, af.index)], tCrW[(None, None, kb, 2 * wf.index + 1)], hi)
                            wf.release()
                            af.release()
                        elif cutlass.const_expr(kind == 1):
                            wf = w_cons.wait_and_advance()
                            for h in cutlass.range_constexpr(2):
                                af = a_cons.wait_and_advance()
                                for kb in cutlass.range_constexpr(4):
                                    mma.set(tcgen05.Field.ACCUMULATE, (k + h + kb) > 0)
                                    cute.gemm(mma, lo, tCrA[(None, None, kb, af.index)], tCrW[(None, None, kb, 2 * wf.index + h)], lo)
                                af.release()
                            wf.release()
                        else:
                            are = a_cons.wait_and_advance()
                            aim = a_cons.wait_and_advance()
                            wf = w_cons.wait_and_advance()
                            for kb in cutlass.range_constexpr(4):
                                mma.set(tcgen05.Field.ACCUMULATE, (k + kb) > 0)
                                cute.gemm(mma, lo, tCrA[(None, None, kb, are.index)], tCrW[(None, None, kb, 2 * wf.index)], lo)
                                cute.gemm(mma, hi, tCrA[(None, None, kb, aim.index)], tCrW[(None, None, kb, 2 * wf.index)], hi)
                                mma.set(tcgen05.Field.ACCUMULATE, True)
                                cute.gemm(mma, hi, tCrA[(None, None, kb, are.index)], tCrW[(None, None, kb, 2 * wf.index + 1)], hi)
                                mma.set(tcgen05.Field.NEGATE_B, True)
                                cute.gemm(mma, lo, tCrA[(None, None, kb, aim.index)], tCrW[(None, None, kb, 2 * wf.index + 1)], lo)
                                mma.set(tcgen05.Field.NEGATE_B, False)
                            wf.release()
                            are.release()
                            aim.release()
                    ee.commit()
            acc0_prod.tail()
            acc1_prod.tail()

    # ============================ epilogue ============================
    if warp_idx < EPI_WARPS:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tptr = tmem.retrieve_ptr(F32)
        l512 = mma1.make_fragment_C(mma1.partition_shape_C((128, 512))).layout
        tAcc512 = cute.make_tensor(tptr, l512)
        epi_tiler = ((128, SUBW),)
        ld_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(SUBW)), F32),
                                         cute.zipped_divide(tAcc512, epi_tiler)[None, 0])
        lane = tidx % 128
        wg = warp_idx // 4
        ld_thr = ld_copy.get_slice(lane)
        tD512 = ld_thr.partition_S(cute.zipped_divide(tAcc512, epi_tiler))
        frag_shape = ld_thr.partition_D(cute.zipped_divide(tAcc512, epi_tiler))[None, None, 0].shape
        rAcc = cute.make_rmem_tensor(frag_shape, F32)
        rAccf = cute.make_tensor(rAcc.iterator, cute.make_layout(SUBW))
        o16 = cute.make_rmem_tensor(cute.make_layout(SUBW), F16)
        for pt in cutlass.range(cid, npairs, ncl, unroll=1):
            tile = pt * 2 + v   # mG/mZ row offsets in Int64: zp row 1152 x Ep exceeds 2^31 above 1.86M padded edges
            live = tile < num_tiles
            for p in cutlass.range_constexpr(len(PASSES)):
                rg = PASSES[p][1]
                if cutlass.const_expr(rg == 0):
                    full = acc0_cons.wait_and_advance()
                else:
                    full = acc1_cons.wait_and_advance()
                if live:
                    for seg in EPI_SEGS[p]:
                        c0, nval, dst, dcol, sig, boff = seg
                        nper = nval // SUBW // EPI_WG
                        for jj in cutlass.range_constexpr(nper):
                            j = wg * nper + jj
                            cute.copy(ld_copy, tD512[None, None, (rg * 256 + c0) // SUBW + j], rAcc)
                            vv = rAccf.load()
                            if cutlass.const_expr(boff >= 0):
                                vv = vv + cute.local_tile(mBias, (SUBW,), (boff // SUBW + j,)).load()
                            if cutlass.const_expr(sig):
                                vv = 1.0 / (1.0 + cute.math.exp(-vv, fastmath=True))
                            o16.store(vv.to(F16))
                            for hh in cutlass.range_constexpr(SUBW // 8):
                                cg = (dcol + j * SUBW) // 8 + hh
                                if cutlass.const_expr(dst == 0):
                                    cute.autovec_copy(cute.local_tile(o16, (8,), (hh,)), mG[(cutlass.Int64(tile) * 32 + cg) * 128 + lane, None])
                                else:
                                    cute.autovec_copy(cute.local_tile(o16, (8,), (hh,)), mZ[(cutlass.Int64(tile) * 144 + cg) * 128 + lane, None])
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    full.release()
        epi_bar.arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tptr, 512)


@cute.jit
def k1c_host(a: cute.Tensor, w: cute.Tensor, bias: cute.Tensor, z: cute.Tensor, g: cute.Tensor, cnt: cute.Tensor,
             ntile_max: cutlass.Int32, num_ctas: cutlass.Int32, stream: cuda.CUstream):
    Km = tcgen05.OperandMajorMode.K
    mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(F16, F32, (256, 128, 16), tcgen05.CtaGroup.TWO,
                                                   tcgen05.OperandSource.SMEM, Km, Km))
    mma1 = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(F16, F32, (128, 256, 16), tcgen05.CtaGroup.ONE,
                                                    tcgen05.OperandSource.SMEM, Km, Km))
    tiler = (256, 128, TILE_K)
    a_l = sm100_utils.make_smem_layout_a(mma, tiler, F16, A_STAGES)
    w_l = sm100_utils.make_smem_layout_b(mma, tiler, F16, 2 * W_STAGES)   # 2 half-tiles (N=128) per W stage
    vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma.thr_id,))
    a_op = utils.sm100.cluster_shape_to_tma_atom_A((2, 1), mma.thr_id)
    tma_a, tA = cute.nvgpu.make_tiled_tma_atom_A(a_op, a, cute.select(a_l, mode=[0, 1, 2]), tiler, mma, vmnk.shape)
    w_op = utils.sm100.cluster_shape_to_tma_atom_B((2, 1), mma.thr_id)
    tma_w, tW = cute.nvgpu.make_tiled_tma_atom_B(w_op, w, cute.select(w_l, mode=[0, 1, 2]), tiler, mma, vmnk.shape)
    k1c_kernel(mma, mma1, tma_a, tA, tma_w, tW, bias, z, g, cnt, ntile_max, a_l, w_l, vmnk).launch(
        grid=(num_ctas, 1, 1), block=(NTHREADS, 1, 1), cluster=(2, 1, 1), stream=stream)


# ------------------------------------------------------------------ host side
def arrange_w(W0, b0, W1, W2):
    """Wflat [NW_STAGES*2*128, 64] FP16: W stage i = 128-row tiles 2i, 2i+1 (each a 2-CTA N=128 B tile), in TMA
    consumption order; bias fp32 [768] (gate | z0a | z0b + pad)"""
    import torch
    dev = W0.device
    W0f, W1f, W2f = W0.float(), W1.float(), W2.float()
    Wa, Wb = W1f[:256, :512], W1f[256:, :512]
    assert torch.equal(W1f[256:, 512:], Wa) and torch.equal(W1f[:256, 512:], -Wb), "W1 is not SO(2) [[Wa,-Wb],[Wb,Wa]]"
    Va, Vb = W2f[:128, :256], W2f[128:, :256]
    assert torch.equal(W2f[128:, 256:], Va) and torch.equal(W2f[:128, 256:], -Vb), "W2 is not SO(2) [[Va,-Vb],[Vb,Va]]"
    ks = lambda k: slice(k * 64, (k + 1) * 64)
    t = []
    gen = {
        "gate": lambda: [x for k in range(12) for x in (W0f[0:128, ks(k)], W0f[128:256, ks(k)])],
        "z0a": lambda: [x for k in range(12) for x in (W0f[256:384, ks(k)], W0f[384:512, ks(k)])],
        "z0b": lambda: [x for j in range(6) for x in (W0f[512:640, ks(2 * j)], W0f[512:640, ks(2 * j + 1)])],
        "m1h0": lambda: [x for k in range(8) for x in (Wa[0:128, ks(k)], Wb[0:128, ks(k)])],
        "m1h1": lambda: [x for k in range(8) for x in (Wa[128:256, ks(k)], Wb[128:256, ks(k)])],
        "m2": lambda: [x for k in range(4) for x in (Va[:, ks(k)], Vb[:, ks(k)])],
    }
    for n in PASS_NAMES:
        t += gen[n]()
    assert len(t) == 2 * NW_STAGES
    b = torch.zeros(768, device=dev)
    b[:640] = b0.float()
    return torch.cat(t, 0).half().contiguous(), b.contiguous()


class K1:
    def __init__(self, W0, W1, W2, b0):
        import torch
        self.wflat, self.bias = arrange_w(W0, b0, W1, W2)
        nsm = torch.cuda.get_device_properties(W0.device).multi_processor_count
        try:
            ncl = utils.HardwareInfo().get_max_active_clusters(2)
        except Exception:
            ncl = nsm // 2
        self.num_ctas = 2 * min(ncl, nsm // 2)
        self._cnt = {}

    def __call__(self, A, save_z, save_g, cnt=None):
        import torch
        from cutlass.cute.runtime import from_dlpack
        nt = A.shape[0] // 128
        if cnt is None:
            cnt = self._cnt.get(nt)
            if cnt is None:
                cnt = self._cnt[nt] = torch.full((1,), nt, dtype=torch.int32, device=A.device)
        ts = [A, self.wflat, self.bias, save_z.view(-1, 8), save_g.view(-1, 8), cnt]
        cu = [from_dlpack(t, assumed_align=16) for t in ts]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = ("k1", self.num_ctas) + tuple(tuple(t.shape) for t in ts)
        if key not in KERNELS:      # one compile per shape for all blocks (weights are kernel arguments)
            KERNELS[key] = cute.compile(k1c_host, *cu, nt, self.num_ctas, stream)
        KERNELS[key](*cu, nt, self.num_ctas, stream)
