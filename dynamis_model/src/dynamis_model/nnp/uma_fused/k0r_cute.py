"""CuTe DSL (sm100) K0r: rotation into M-row order with the radial MLP's last GEMM (fc3) folded in.

Per 128-edge packed tile (lane = edge):
    radial = FP16(h[pc[e]] . W_last^T + b_last)          (tcgen05, FP16 operands, fp32 acc in TMEM)
    A[e, M*256 + side*128 + c] = FP16( (D[e] x[node_side])[L(M)][c] * radial[e, RAD_ROW[M]*256 + side*128 + c] )
    R[e, :] = radial (FP16 packed row-major [Ep,1536], what S3 reads in the backward)
Same A / R as a separate radial GEMM followed by the rotation, up to fp32 accumulation order in the radial GEMM.

MMA units: 8 per tile, unit u = (side = u // 4, cq = u % 4) covers channels cq*32 .. cq*32+31 of one side for all
6 radial rows q: N = 192 columns ordered (sb, q, c16) (sb = 16-channel half), i.e. W_last rows permuted on the host
(wperm[u*192 + sb*96 + q*16 + c] = W_last[q*256 + side*128 + cq*32 + sb*16 + c]). TMEM: buffer g at column 256*g
(g = u % 2), each owned by one epilogue warpgroup (1-stage acc pipeline each).

Warp roles: 0-3 / 4-7 epilogue groups (units with u % 2 == 0 / 1), 8-11 producers (h rows -> FP16 A ring),
12 MMA, 13 TMA (W_last tiles (192 x 64), from L2), 14-15 idle.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

import cuda.bindings.driver as cuda

from dynamis_model.nnp.uma_fused.common import CP_EVICT_LAST, KERNELS, set_regs

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
TILE_M, TILE_K, TILE_KB = 128, 64, 16
UNIT_N, NUNITS = 192, 8
NTHREADS = 512
MMA_WARP, TMA_WARP = 12, 13
W_KT = 64        # K per W TMA stage (24 KB); W_last tiles loaded with an L2 evict_last hint
W_STAGES = 2
A_STAGES = 2
SR_LD = 200   # sR row stride (halves): 192 + 8 pad -> conflict-free 16 B row writes
CPL = 8          # channels per lane in phase 2 (16 B A / R stores, streaming)
SUBW = 16        # channels per epilogue step
REGS_ENTRY = 512 * 128 // NTHREADS
REGS_EPI = 168
REGS_OTH = 2 * REGS_ENTRY - REGS_EPI                 # producers + ctrl: 256*(EPI-128) + 256*(OTH-128) = 0

# M-row -> (radial row q, Wigner row Lr, l columns)
_L0, _L1, _L2 = (0,), (1, 2, 3), (4, 5, 6, 7, 8)
M_INFO = ((0, 0, _L0), (1, 2, _L1), (2, 6, _L2), (3, 3, _L1), (4, 7, _L2), (3, 1, _L1), (4, 5, _L2),
          (5, 8, _L2), (5, 4, _L2))
# epilogue order: l-group -> M rows using it
DW_OFF = [sum(len(M_INFO[m][2]) for m in range(M)) for M in range(9)]
GROUPS = ((_L0, (0,)), (_L1, (1, 3, 5)), (_L2, (2, 4, 6, 7, 8)))


@cute.struct
class SharedStorage:
    w_mbar: cute.struct.MemRange[cutlass.Int64, W_STAGES * 2]
    a_mbar: cute.struct.MemRange[cutlass.Int64, A_STAGES * 2]
    acc0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    acc1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def k0r_kernel(
    mma: cute.TiledMma, tma_w: cute.CopyAtom, mW: cute.Tensor,
    mH: cute.Tensor, mPc: cute.Tensor, mSrc: cute.Tensor, mTgt: cute.Tensor, mD: cute.Tensor, mX: cute.Tensor,
    mBias: cute.Tensor, mA: cute.Tensor, mR: cute.Tensor, mCnt: cute.Tensor,
    w_smem_layout: cute.ComposedLayout, a_smem_layout: cute.ComposedLayout, cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    cl_coord = cta_layout_vmnk.get_flat_coord(0)
    mcast_mask = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cl_coord, mcast_mode=1)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sW = smem.allocate_tensor(io_dtype, w_smem_layout.outer, 128, w_smem_layout.inner)
    sA = smem.allocate_tensor(io_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
    sR = smem.allocate_tensor(io_dtype, cute.make_layout((2, 128, SR_LD), stride=(128 * SR_LD, SR_LD, 1)), 16)

    all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=all_bar)
    tmem.allocate(512)

    one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    g128 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
    w_bytes = cute.size_in_bytes(io_dtype, cute.select(w_smem_layout, mode=[0, 1, 2]))
    w_prod, w_cons = pipeline.PipelineTmaUmma.create(
        num_stages=W_STAGES, producer_group=one, consumer_group=one, tx_count=w_bytes,
        barrier_storage=storage.w_mbar.data_ptr(), cta_layout_vmnk=cta_layout_vmnk).make_participants()
    a_prod, a_cons = pipeline.PipelineAsyncUmma.create(
        num_stages=A_STAGES, producer_group=g128, consumer_group=one,
        barrier_storage=storage.a_mbar.data_ptr()).make_participants()
    acc0_prod, acc0_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=g128,
        barrier_storage=storage.acc0_mbar.data_ptr()).make_participants()
    acc1_prod, acc1_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=g128,
        barrier_storage=storage.acc1_mbar.data_ptr()).make_participants()

    thr = mma.get_slice(0)
    gW = cute.local_tile(mW, (TILE_M, UNIT_N, W_KT), (None, None, None), proj=(None, 1, 1))
    sW_, gW_ = cpasync.tma_partition(tma_w, cl_coord[1], cute.make_layout(1),
                                     cute.group_modes(sW, 0, 3), cute.group_modes(thr.partition_B(gW), 0, 3))
    tCrW = mma.make_fragment_B(sW)
    tCrA = mma.make_fragment_A(sA)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    lU = mma.make_fragment_C(mma.partition_shape_C((TILE_M, UNIT_N))).layout
    l512 = mma.make_fragment_C(mma.partition_shape_C((TILE_M, 3 * UNIT_N))).layout
    tAcc512 = cute.make_tensor(tmem_ptr, l512)
    tAccU = [cute.make_tensor(tmem_ptr, lU), cute.make_tensor(tmem_ptr + 256, lU)]

    epi_tiler = ((TILE_M, SUBW),)
    ld_copy = tcgen05.make_tmem_copy(
        cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(SUBW)), acc_dtype),
        cute.zipped_divide(tAcc512, epi_tiler)[None, 0])
    lane = tidx % 128
    ld_thr = ld_copy.get_slice(lane)
    tD512 = ld_thr.partition_S(cute.zipped_divide(tAcc512, epi_tiler))
    frag_shape = ld_thr.partition_D(cute.zipped_divide(tAcc512, epi_tiler))[None, None, 0].shape

    num_tiles = cutlass.min(mCnt[0], cute.size(mSrc, mode=[0]) // TILE_M)
    nctas, _, _ = cute.arch.grid_dim()

    # ======================= producers (warps 8-11): h[pc[e]] fp32 -> FP16 A ring =======================
    if warp_idx >= 8 and warp_idx < 12:
        set_regs(REGS_OTH, REGS_ENTRY)
        r16p = cute.make_rmem_tensor(cute.make_layout(32), io_dtype)
        for tile in cutlass.range(bidx, num_tiles, nctas, unroll=1):
            e = tile * TILE_M + lane
            pe = mPc[e]
            for kt in cutlass.range_constexpr(2):
                a_empty = a_prod.acquire_and_advance()
                for hh in cutlass.range_constexpr(2):
                    v = cute.local_tile(mH[pe, None], (32,), (kt * 2 + hh,)).load()
                    r16p.store(v.to(io_dtype))
                    for q16 in cutlass.range_constexpr(2):
                        cute.autovec_copy(cute.local_tile(r16p, (16,), (q16,)),
                                          sA[((lane, None), 0, 2 * hh + q16, a_empty.index)])
                cute.arch.fence_proxy("async.shared", space="cta")
                a_empty.commit()

    # ======================= TMA (13) / MMA (12) =======================
    if warp_idx >= 12:
        set_regs(REGS_OTH, REGS_ENTRY)
    if warp_idx == TMA_WARP:
        for tile in cutlass.range(bidx, num_tiles, nctas, unroll=1):
            for u in cutlass.range_constexpr(NUNITS):
                for kt in cutlass.range_constexpr(128 // W_KT):
                    w_empty = w_prod.acquire_and_advance()
                    cute.copy(tma_w, gW_[(None, u, kt)], sW_[(None, w_empty.index)],
                              tma_bar_ptr=w_empty.barrier, mcast_mask=mcast_mask,
                              cache_policy=cutlass.Int64(CP_EVICT_LAST))
        w_prod.tail()
    if warp_idx == MMA_WARP:
        for tile in cutlass.range(bidx, num_tiles, nctas, unroll=1):
            ha0 = a_cons.wait_and_advance()
            ha1 = a_cons.wait_and_advance()
            for u in cutlass.range_constexpr(NUNITS):
                if cutlass.const_expr(u % 2 == 0):
                    acc_empty = acc0_prod.acquire_and_advance()
                else:
                    acc_empty = acc1_prod.acquire_and_advance()
                tacc = tAccU[u % 2]
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range_constexpr(128 // W_KT):
                    w_full = w_cons.wait_and_advance()
                    aidx = ha0.index if (kt * W_KT) // TILE_K == 0 else ha1.index
                    for kb in cutlass.range_constexpr(W_KT // TILE_KB):
                        kba = ((kt * W_KT) % TILE_K) // TILE_KB + kb
                        cute.gemm(mma, tacc, tCrA[(None, None, kba, aidx)], tCrW[(None, None, kb, w_full.index)], tacc)
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    w_full.release()
                acc_empty.commit()
            ha0.release()
            ha1.release()

    # ======================= epilogue (warps 0-3: even units, 4-7: odd units) =======================
    # phase 1 (lane = edge): TMEM radial + bias -> FP16 -> sR rows (warp-local rows)
    # phase 2 (2 edges per warp instruction, 16 lanes x 2 channels per edge): x gathers, rotation, A / R stores
    # (coalesced 64 B row segments). A warp only touches the sR rows of its own 32 edges -> __syncwarp only.
    rAcc = cute.make_rmem_tensor(frag_shape, acc_dtype)
    rAccf = cute.make_tensor(rAcc.iterator, cute.make_layout(SUBW))
    o16 = cute.make_rmem_tensor(cute.make_layout(SUBW), io_dtype)
    o8 = cute.make_rmem_tensor(cute.make_layout(CPL), io_dtype)
    st_atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), io_dtype, num_bits_per_copy=CPL * 16,
                                  store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING)    # A stores
    st_atom_r = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), io_dtype, num_bits_per_copy=CPL * 16,
                                    store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING)  # R stores
    for grp in cutlass.range_constexpr(2):
        if warp_idx >= 4 * grp and warp_idx < 4 * grp + 4:
            set_regs(REGS_EPI, REGS_ENTRY)
            wi = warp_idx % 4
            l32 = tidx % 32
            c8 = l32 % (32 // CPL)
            for tile in cutlass.range(bidx, num_tiles, nctas, unroll=1):
                e = tile * TILE_M + lane
                ns = mSrc[e]
                nt = mTgt[e]
                for ui in cutlass.range_constexpr(NUNITS // 2):
                    u = 2 * ui + grp
                    side, cq = u // 4, u % 4
                    if cutlass.const_expr(grp == 0):
                        acc_full = acc0_cons.wait_and_advance()
                    else:
                        acc_full = acc1_cons.wait_and_advance()
                    for j in cutlass.range_constexpr(UNIT_N // SUBW):
                        col = grp * 256 + j * SUBW
                        cute.copy(ld_copy, tD512[None, None, col // SUBW], rAcc)
                        b = cute.local_tile(mBias, (SUBW,), (u * (UNIT_N // SUBW) + j,)).load()
                        o16.store((rAccf.load() + b).to(io_dtype))
                        cute.autovec_copy(o16, cute.local_tile(sR[grp, lane, None], (SUBW,), (j,)))
                    acc_full.release()
                    cute.arch.sync_warp()
                    nside = ns if side == 0 else nt
                    for p in cutlass.range(32 // CPL, unroll=1):
                        srcl = CPL * p + l32 // (32 // CPL)
                        el = wi * 32 + srcl
                        ee = cutlass.Int64(tile) * TILE_M + wi * 32 + srcl   # int64 row (A row = 2304 > 2^31 at Ep > 932k)
                        node = cute.arch.shuffle_sync(nside, srcl)
                        xr = [cute.local_tile(mX[node, l, None], (CPL,), ((cq * 32) // CPL + c8,)).load()
                              for l in range(9)]
                        dwv = [mD[ee, M_INFO[M][1] * 9 + l] for M in range(9) for l in M_INFO[M][2]]
                        rq = []
                        for q in cutlass.range_constexpr(6):
                            cc = c8 * CPL
                            rv = cute.local_tile(sR[grp, el, None], (CPL,), (((cc // 16) * 96 + q * 16 + cc % 16) // CPL,)).load()
                            rq.append(rv)
                            o8.store(rv)
                            cute.copy(st_atom_r, o8, cute.local_tile(
                                mR[ee, None], (CPL,), ((q * 256 + side * 128 + cq * 32) // CPL + c8,)))
                        for lgrp, Ms in GROUPS:
                            xs = [xr[l].to(acc_dtype) for l in lgrp]
                            for M in Ms:
                                q = M_INFO[M][0]
                                k0 = DW_OFF[M]
                                y = None
                                for jj, xv in enumerate(xs):
                                    y = dwv[k0 + jj] * xv if y is None else y + dwv[k0 + jj] * xv
                                o8.store((y * rq[q].to(acc_dtype)).to(io_dtype))
                                cute.copy(st_atom, o8, cute.local_tile(
                                    mA[ee, None], (CPL,), ((M * 256 + side * 128 + cq * 32) // CPL + c8,)))
                    cute.arch.sync_warp()

    tmem.relinquish_alloc_permit()
    all_bar.arrive_and_wait()
    tmem.free(tmem_ptr, 512)


@cute.jit
def k0r_host(w: cute.Tensor, h: cute.Tensor, pc: cute.Tensor, src: cute.Tensor, tgt: cute.Tensor, d: cute.Tensor,
             x: cute.Tensor, bias: cute.Tensor, a: cute.Tensor, r: cute.Tensor, cnt: cute.Tensor,
             num_ctas: cutlass.Int32, stream: cuda.CUstream):
    """w = permuted W_last [1536, 128] FP16 (K-major B), h fp32 [E, 128], x FP16 [N, 9, 128]"""
    K = tcgen05.OperandMajorMode.K
    mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        io_dtype, acc_dtype, (128, UNIT_N, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM, K, K))
    w_l = sm100_utils.make_smem_layout_b(mma, (TILE_M, UNIT_N, W_KT), io_dtype, W_STAGES)
    a_l = sm100_utils.make_smem_layout_a(mma, (TILE_M, UNIT_N, TILE_K), io_dtype, A_STAGES)
    cta_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id,))
    op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    atom, tw = cute.nvgpu.make_tiled_tma_atom_B(op, w, cute.select(w_l, mode=[0, 1, 2]),
                                                (TILE_M, UNIT_N, W_KT), mma, cta_layout_vmnk.shape)
    k0r_kernel(mma, atom, tw, h, pc, src, tgt, d, x, bias, a, r, cnt, w_l, a_l, cta_layout_vmnk).launch(
        grid=(num_ctas, 1, 1), block=(NTHREADS, 1, 1), min_blocks_per_mp=1, stream=stream)


# ------------------------------------------------------------------ host side
def permute_w(W_last, b_last):
    """rows of W_last [1536,128] / b_last [1536] in unit order (see module doc)"""
    import torch
    idx = []
    for u in range(NUNITS):
        side, cq = u // 4, u % 4
        for j in range(UNIT_N):
            sb, q, c = j // 96, (j % 96) // 16, j % 16
            idx.append(q * 256 + side * 128 + cq * 32 + sb * 16 + c)
    idx = torch.tensor(idx, device=W_last.device)
    return W_last.index_select(0, idx).half().contiguous(), b_last.index_select(0, idx).float().contiguous()


def k0r_cute(wperm16, bperm, h, pc, src, tgt, d, x16, A, R, cnt, num_ctas):
    """A [Ep,2304] FP16 and R [Ep,1536] FP16 on torch's current stream"""
    import torch
    from cutlass.cute.runtime import from_dlpack
    x3 = x16.view(x16.shape[0], 9, 128)
    ts = [wperm16, h, pc, src, tgt, d, x3, bperm, A, R, cnt]
    cu = [from_dlpack(t, assumed_align=16) for t in ts] + [num_ctas]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = ("k0r",) + tuple(tuple(t.shape) for t in ts)
    if key not in KERNELS:
        KERNELS[key] = cute.compile(k0r_host, *cu, stream)
    KERNELS[key](*cu, stream)
