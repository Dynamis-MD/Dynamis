"""CuTe DSL (sm100) K2 of the fused forward: node messages from K1's saved pre-activations.

Per 128-edge packed tile (lane = edge, <= MAX_SLOTS target nodes per tile):
    z    = act(zp) * g                    zp / g = K1 output (FP16, tile-blocked ilv, staged to SMEM by TMA)
    q_R  = z_m(R) . V_m(R)^T  (+ c0)       conv2 of the 9 M-rows R (tcgen05 TS MMA: z in TMEM, V streamed by TMA)
    Y[(slot, i), c] += sum_e Dinv_env[e, i, L(R)] * q_R[e, c]          (inverse rotation + per-target sum, as an MMA)
    node_out[tile_nodes[tile, slot], i, c] = Y[(slot, i), c]
Matches ``reference.edge_block_reference`` (FP16 operands, fp32 accumulation). No atomics: every target node is
owned by exactly one tile, so the result is bitwise reproducible.

Per tile the M-rows run in three phases (m=1: M3..M6, m=2: M7, M8, m=0: M0..M2 + bias c0). conv2(R+1) is issued
before the reduction of row R, so the V stream and the MMAs overlap the epilogue's G^T / coefficient SMEM writes.
TMEM: z [0, 256) (FP16, packed), Q (conv2 row) [256, 384), Y^T [448, 512) (lane = channel).

Warps (384 threads, 3 warpgroups; setmaxnreg needs min_blocks_per_mp=1):
    0-3  epilogue WG0: channels 0-63 of z; reads Q, writes G^T = Q^T (A operand of the reduction MMA); Y rows 0-31
    4-7  epilogue WG1: channels 64-127 of z; writes the reduction coefficients (B operand); Y rows 32-63
    8 MMA, 9 V TMA, 10 zp / g TMA (evict-first), 11 Dinv TMA (the tile's [128, 81] fp32 block into SMEM)
Registers: entry 168, epilogue 176, control 152: 256 * (176 - 168) + 128 * (152 - 168) = 0.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from dynamis_model.nnp.uma_fused.common import CP_EVICT_FIRST, KERNELS, MAX_SLOTS, W_KB, set_regs

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
TILE_M, TILE_KB = 128, 16
NTHREADS = 384
W_STAGES = 2      # V ring = W_STAGES x 32 KB (the 512 x W_KB layout only sizes it)
SUB_COLS = 32
V_KB = 64         # K per V stage (128 x V_KB FP16)
NVST = W_STAGES * 128 // V_KB   # V ring stages
Q_COL, Y_COL = 256, 448         # conv2 row accumulator, node accumulator Y^T (64 cols)
DINV_WARP = 11
# consumption order of 32-col zp / g sub-tiles per tile: g (8), zp m1 rows M3..M6 (16), m2 (8), m0 (12)
K2_ITEMS = [(1, cb) for cb in range(8)] + [(0, cb) for cb in range(12, 28)] + [(0, cb) for cb in range(28, 36)] + [(0, cb) for cb in range(12)]
NEPI = 2          # epilogue warpgroups
ZS = 4            # zp / g TMA ring stages (8 KB) per epilogue WG
MMA_WARP, TMA_WARP = 8, 9
CTRL_WARP0 = 8    # first warp of the control warpgroup
REGS_EPI, REGS_CTRL = 176, 152
REGS_ENTRY = 65536 // NTHREADS // 8 * 8
assert 256 * (REGS_EPI - REGS_ENTRY) + 128 * (REGS_CTRL - REGS_ENTRY) <= 0, 'setmaxnreg budget'

# M-row bookkeeping: (L row of the M-row, l-block rows i it rotates into)
L1_ROWS, L2_ROWS = (1, 2, 3), (4, 5, 6, 7, 8)
M1_ROWS = ((3, L1_ROWS), (7, L2_ROWS), (1, L1_ROWS), (5, L2_ROWS))  # M3..M6
M2_ROWS = ((8, L2_ROWS), (4, L2_ROWS))                              # M7, M8
M0_ROWS = ((0, (0,)), (2, L1_ROWS), (6, L2_ROWS))                   # M0..M2


@cute.struct
class SharedStorage:
    w_mbar: cute.struct.MemRange[cutlass.Int64, NVST * 2]
    z0_mbar: cute.struct.MemRange[cutlass.Int64, ZS * 2]
    z1_mbar: cute.struct.MemRange[cutlass.Int64, ZS * 2]
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    ab3_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    y_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    d_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def k2_kernel(
    mma: cute.TiledMma, mma2: cute.TiledMma, mma3: cute.TiledMma,
    tma_v1: cute.CopyAtom, mV1: cute.Tensor,
    tma_v2: cute.CopyAtom, mV2: cute.Tensor,
    tma_v0: cute.CopyAtom, mV0: cute.Tensor,
    mB0: cute.Tensor, mC0: cute.Tensor,
    mSlot: cute.Tensor, mTileNodes: cute.Tensor, mNodeOut: cute.Tensor, mCnt: cute.Tensor,
    tma_zt: cute.CopyAtom, mZt: cute.Tensor, tma_gt: cute.CopyAtom, mGt: cute.Tensor,
    tma_dv: cute.CopyAtom, mDv: cute.Tensor,
    w_smem_layout: cute.ComposedLayout, v_smem_layout: cute.ComposedLayout, z_tmem_layout: cute.ComposedLayout,
    r_smem_layout: cute.ComposedLayout, rb_smem_layout: cute.ComposedLayout, cta_layout_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, _, _ = cute.arch.block_idx()
    cta_rank = cute.arch.block_idx_in_cluster()
    cl_coord = cta_layout_vmnk.get_flat_coord(cta_rank)
    n_mcast = cute.size(cta_layout_vmnk, mode=[1])
    mcast_mask = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cl_coord, mcast_mode=1)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sW = smem.allocate_tensor(io_dtype, w_smem_layout.outer, 128, w_smem_layout.inner)   # V ring (64 KB)
    sV = cute.make_tensor(cute.recast_ptr(cute.recast_ptr(sW.iterator, dtype=io_dtype), v_smem_layout.inner),
                          v_smem_layout.outer)
    sZ = smem.allocate_tensor(io_dtype, cute.make_layout(NEPI * ZS * 4096), 1024)       # zp / g sub-tile rings
    # reduction Y^T[c, (slot,i)] += G^T[c, e] coef^T[e, (slot,i)]:  A = G^T (MN-major), B = coef^T (N = 64)
    sB3 = smem.allocate_tensor(io_dtype, r_smem_layout.outer, 128, r_smem_layout.inner)    # G^T  (A operand)
    sA3 = smem.allocate_tensor(io_dtype, rb_smem_layout.outer, 128, rb_smem_layout.inner)  # coef (B operand)
    sBias = smem.allocate_tensor(acc_dtype, cute.make_layout(1024), 16)  # b0 [0,640) | c0 [640,1024)
    sDv = smem.allocate_tensor(acc_dtype, cute.make_layout(128 * 81), 128)   # one tile's Dinv rows (fp32, row-major)
    for i_ in cutlass.range(tidx, 640, NTHREADS):
        sBias[i_] = mB0[i_]
    for i_ in cutlass.range(tidx, 384, NTHREADS):
        sBias[640 + i_] = mC0[i_]  # visible after the TMEM-allocation barrier below

    all_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=all_bar)
    tmem.allocate(512)

    one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mc_cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, n_mcast)
    w_prod, w_cons = pipeline.PipelineTmaUmma.create(
        num_stages=NVST, producer_group=one, consumer_group=mc_cons, tx_count=128 * V_KB * 2,
        barrier_storage=storage.w_mbar.data_ptr(), cta_layout_vmnk=cta_layout_vmnk).make_participants()
    # PipelineTmaAsync.consumer_release arrives only from lane 0 of each consumer warp, so the empty-barrier count is
    # the number of consumer WARPS (4), not 128 threads
    g4w = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    z0_prod, z0_cons = pipeline.PipelineTmaAsync.create(
        num_stages=ZS, producer_group=one, consumer_group=g4w, tx_count=8192,
        barrier_storage=storage.z0_mbar.data_ptr()).make_participants()
    z1_prod, z1_cons = pipeline.PipelineTmaAsync.create(
        num_stages=ZS, producer_group=one, consumer_group=g4w, tx_count=8192,
        barrier_storage=storage.z1_mbar.data_ptr()).make_participants()
    gepi = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128 * NEPI)
    d_prod, d_cons = pipeline.PipelineTmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=g4w, tx_count=128 * 81 * 4,
        barrier_storage=storage.d_mbar.data_ptr()).make_participants()
    ab3_prod, ab3_cons = pipeline.PipelineAsyncUmma.create(
        num_stages=1, producer_group=gepi, consumer_group=one,
        barrier_storage=storage.ab3_mbar.data_ptr()).make_participants()
    y_prod, y_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=gepi,
        barrier_storage=storage.y_mbar.data_ptr()).make_participants()
    # acc: per phase "z written by the epilogue" (the empty acquire/commit pair before each conv2 phase keeps the
    # epilogue from overwriting z while the previous phase's conv2 still reads it) and per row "Q full / free"
    acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1, producer_group=one, consumer_group=gepi,
        barrier_storage=storage.acc_mbar.data_ptr()).make_participants()

    # ---------------- V TMA partitions ----------------
    thr2 = mma2.get_slice(0)

    def v_part(tma, mV):
        g = cute.local_tile(mV, (TILE_M, 128, V_KB), (None, None, None), proj=(None, 1, 1))
        s_, g_ = cpasync.tma_partition(tma, cl_coord[1], cute.make_layout(n_mcast),
                                       cute.group_modes(sV, 0, 3), cute.group_modes(thr2.partition_B(g), 0, 3))
        return s_, g_, cute.size(g, mode=[3]), cute.size(g, mode=[2])

    tV1 = v_part(tma_v1, mV1)
    tV2 = v_part(tma_v2, mV2)
    tV0 = v_part(tma_v0, mV0)
    tCrV = mma2.make_fragment_B(sV)
    tCrA3 = mma3.make_fragment_A(sB3)  # G^T
    tCrB3 = mma3.make_fragment_B(sA3)  # coefficients

    # ---------------- TMEM ----------------
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    l256 = mma.make_fragment_C(mma.partition_shape_C((TILE_M, 256))).layout
    l128 = mma2.make_fragment_C(mma2.partition_shape_C((TILE_M, 128))).layout
    tAcc256 = cute.make_tensor(tmem_ptr, l256)          # [0, 256)
    tAccQ = cute.make_tensor(tmem_ptr + Q_COL, l128)    # [256, 384)
    l64 = mma3.make_fragment_C(mma3.partition_shape_C((TILE_M, 64))).layout
    tY = cute.make_tensor(tmem_ptr + Y_COL, l64)        # Y^T [448, 512): lane = channel
    fz = mma2.make_fragment_A(cute.make_tensor(tmem_ptr, z_tmem_layout.outer))[None, None, None, 0]
    tCrZ = cute.make_tensor(cute.recast_ptr(tmem_ptr, dtype=fz.element_type), fz.layout)

    epi_tiler = ((TILE_M, SUB_COLS),)
    ld_copy = tcgen05.make_tmem_copy(
        cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(SUB_COLS)), acc_dtype),
        cute.zipped_divide(tAcc256, epi_tiler)[None, 0])
    p32 = SUB_COLS // 2
    zf32 = cute.zipped_divide(tAcc256, ((TILE_M, p32),))   # fp32 view of [0,256) in 16-col pieces
    st_copy = tcgen05.make_tmem_copy(
        cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(p32)), acc_dtype), zf32[None, 0])
    lane = tidx % 128
    ld_thr = ld_copy.get_slice(lane)
    st_thr = st_copy.get_slice(lane)
    tDQ = ld_thr.partition_S(cute.zipped_divide(tAccQ, epi_tiler))      # 4 sub-tiles
    tDY = ld_thr.partition_S(cute.zipped_divide(tY, epi_tiler))         # 2 sub-tiles
    tDZ = st_thr.partition_D(zf32)                                     # 16 packed z pieces
    frag_shape = ld_thr.partition_D(cute.zipped_divide(tAcc256, epi_tiler))[None, None, 0].shape
    rAcc = cute.make_rmem_tensor(frag_shape, acc_dtype)
    r16 = cute.make_rmem_tensor(frag_shape, io_dtype)
    r16_32 = cute.recast_tensor(r16, acc_dtype)
    r16_flat = cute.make_tensor(r16.iterator, cute.make_layout(SUB_COLS))
    rS = cute.make_rmem_tensor(cute.make_layout(SUB_COLS), io_dtype)  # one zp sub-tile of this edge
    rAccf = cute.make_tensor(rAcc.iterator, cute.make_layout(SUB_COLS))
    rQ = cute.make_rmem_tensor(cute.make_layout(4 * SUB_COLS), io_dtype)   # one conv2 row (FP16)

    # tiles: CTA c processes a contiguous block (same per-CTA counts as round-robin); real tile count from device memory
    num_pairs = cutlass.min(mCnt[0], cute.size(mSlot, mode=[0]) // TILE_M)
    nctas, _, _ = cute.arch.grid_dim()
    nclusters = nctas
    cluster_id = bidx
    q_ = num_pairs // nclusters
    r_ = num_pairs - q_ * nclusters
    blk0 = cluster_id * q_ + r_
    if cluster_id < r_:
        blk0 = cluster_id * q_ + cluster_id

    # ======================= helpers (trace-time inlined; all staged values passed in) =======================
    def conv2_only(c, acc_prod, v_cons, mma2, n_kv):
        acc_empty = acc_prod.acquire_and_advance()
        mma2.set(tcgen05.Field.ACCUMULATE, False)
        tCrZ_ = c.tCrZ
        tCrV_ = c.tCrV
        tAccQ_ = c.tAccQ
        for kv in cutlass.range(n_kv, unroll=1):    # runtime loop: the MMA warp spilled when fully unrolled
            v_full = v_cons.wait_and_advance()
            mma2.set(tcgen05.Field.ACCUMULATE, kv != 0)
            for kb in cutlass.range_constexpr(V_KB // TILE_KB):
                cute.gemm(mma2, tAccQ_, tCrZ_[(None, None, kv * (V_KB // TILE_KB) + kb)],
                          tCrV_[(None, None, kb, v_full.index)], tAccQ_)
                mma2.set(tcgen05.Field.ACCUMULATE, True)
            v_full.release()
        acc_empty.commit()
        return acc_prod, v_cons, mma2

    def red_only(c, ab3_cons, mma3, first_red):
        ab3_full = ab3_cons.wait_and_advance()
        mma3.set(tcgen05.Field.ACCUMULATE, not first_red)
        tY_ = c.tY
        tA3_ = c.tCrA3
        tB3_ = c.tCrB3
        cute.gemm(mma3, tY_, tA3_[(None, None, 0, ab3_full.index)], tB3_[(None, None, 0, ab3_full.index)], tY_)
        mma3.set(tcgen05.Field.ACCUMULATE, True)
        for kb in cutlass.range(1, cute.size(c.tCrA3, mode=[2]), unroll=1):
            cute.gemm(mma3, tY_, tA3_[(None, None, kb, ab3_full.index)], tB3_[(None, None, kb, ab3_full.index)], tY_)
        ab3_full.release()
        return ab3_cons, mma3

    def conv2_rows(c, acc_prod, v_cons, ab3_cons, mma2, mma3, n_kv, n_rows, first_chain, defer_last=False):
        """conv2(R+1) is issued before reduction(R) so its weight stream and MMAs
        overlap the epilogue's coefficient writes for row R"""
        acc_prod, v_cons, mma2 = c.conv2_only(c, acc_prod, v_cons, mma2, n_kv)
        for r in cutlass.range_constexpr(n_rows):
            if cutlass.const_expr(r + 1 < n_rows):
                acc_prod, v_cons, mma2 = c.conv2_only(c, acc_prod, v_cons, mma2, n_kv)
            if cutlass.const_expr(not (defer_last and r == n_rows - 1)):
                ab3_cons, mma3 = c.red_only(c, ab3_cons, mma3, first_chain and r == 0)
        return acc_prod, v_cons, ab3_cons, mma2, mma3

    def z_ready(acc_prod):
        """empty acc handshake per phase: the epilogue has written z (it waits for the previous phase's conv2)"""
        acc_prod.acquire_and_advance().commit()
        return acc_prod

    kk = lane % 16
    kb_ = lane // 16
    r16c = cute.make_rmem_tensor(cute.make_layout(16), io_dtype)
    rGate = cute.make_rmem_tensor(cute.make_layout(256 // NEPI), io_dtype)
    zero16 = cutlass.Float16(0.0)

    def bias_vec(c, mb, col0):
        """mb: 0 -> b0, 1 -> c0 (both staged in SMEM)"""
        return cute.local_tile(c.sBias, (SUB_COLS,), ((col0 + 640 * mb) // SUB_COLS,)).load()

    def load_v(c, v_prod, tv, tma):
        sV_, gV_, n, nrow = tv
        for q in range(nrow):
            for kv in cutlass.range(n):
                v_empty = v_prod.acquire_and_advance()
                cute.copy(tma, gV_[(None, q, kv)], sV_[(None, v_empty.index)],
                          tma_bar_ptr=v_empty.barrier, mcast_mask=c.mcast_mask)
        return v_prod

    import types as _types
    c = _types.SimpleNamespace()
    c.rS = rS
    c.rQ = rQ
    c.rQ4 = [cute.make_rmem_tensor(frag_shape, acc_dtype) for _ in range(4)]
    c.sZ = sZ
    c.sBias = sBias
    c.r16_flat = r16_flat
    c.lane = lane
    c.tCrA3 = tCrA3
    c.tCrB3 = tCrB3
    c.tAccQ = tAccQ
    c.tCrZ = tCrZ
    c.tCrV = tCrV
    c.tY = tY
    c.ld_copy = ld_copy
    c.tDQ = tDQ
    c.rAcc = rAcc
    c.rAccf = rAccf
    c.rGate = rGate
    c.st_copy = st_copy
    c.r16_32 = r16_32
    c.tDZ = tDZ
    c.r16c = r16c
    c.zero16 = zero16
    c.sA3 = sA3
    c.sB3 = sB3
    c.kk = kk
    c.kb_ = kb_
    c.mcast_mask = mcast_mask
    c.conv2_only = conv2_only
    c.red_only = red_only
    c.bias_vec = bias_vec

    # ======================= control warpgroup: V TMA (warp 9), zp / g TMA (10), Dinv TMA (11), MMA (8) =========
    if warp_idx >= CTRL_WARP0:
        set_regs(REGS_CTRL, REGS_ENTRY)
    if warp_idx == TMA_WARP:
        for it in cutlass.range(cluster_id, num_pairs, nclusters, unroll=1):
            w_prod = load_v(c, w_prod, tV1, tma_v1)
            w_prod = load_v(c, w_prod, tV2, tma_v2)
            w_prod = load_v(c, w_prod, tV0, tma_v0)
        w_prod.tail()
    if warp_idx == TMA_WARP + 1:   # whole warp (acquire elects one lane via elect.sync: a single active thread would hang)
        zl = cute.make_layout((16, 256, ZS), stride=(256, 1, 4096))
        gZv = cute.group_modes(cute.local_tile(mZt, (16, 256), (None, 0)), 0, 2)
        gGv = cute.group_modes(cute.local_tile(mGt, (16, 256), (None, 0)), 0, 2)
        sZ0_, gZ_ = cpasync.tma_partition(tma_zt, 0, cute.make_layout(1), cute.group_modes(cute.make_tensor(sZ.iterator, zl), 0, 2), gZv)
        sG0_, gG_ = cpasync.tma_partition(tma_gt, 0, cute.make_layout(1), cute.group_modes(cute.make_tensor(sZ.iterator, zl), 0, 2), gGv)
        sZ1_, _ = cpasync.tma_partition(tma_zt, 0, cute.make_layout(1), cute.group_modes(cute.make_tensor(sZ.iterator + ZS * 4096, zl), 0, 2), gZv)
        sG1_, _ = cpasync.tma_partition(tma_gt, 0, cute.make_layout(1), cute.group_modes(cute.make_tensor(sZ.iterator + ZS * 4096, zl), 0, 2), gGv)
        cpol = cutlass.Int64(CP_EVICT_FIRST)
        for it in cutlass.range(cluster_id, num_pairs, nclusters, unroll=1):
            tile = blk0 + (it - cluster_id) // nclusters
            for kind, cb in K2_ITEMS:
                ring = (cb % 4) // 2   # channel half -> epilogue WG
                if cutlass.const_expr(ring == 0):
                    empty = z0_prod.acquire_and_advance()
                    if cutlass.const_expr(kind == 1):
                        cute.copy(tma_gt, gG_[(None, tile * 8 + cb)], sG0_[(None, empty.index)], tma_bar_ptr=empty.barrier, cache_policy=cpol)
                    else:
                        cute.copy(tma_zt, gZ_[(None, tile * 36 + cb)], sZ0_[(None, empty.index)], tma_bar_ptr=empty.barrier, cache_policy=cpol)
                else:
                    empty = z1_prod.acquire_and_advance()
                    if cutlass.const_expr(kind == 1):
                        cute.copy(tma_gt, gG_[(None, tile * 8 + cb)], sG1_[(None, empty.index)], tma_bar_ptr=empty.barrier, cache_policy=cpol)
                    else:
                        cute.copy(tma_zt, gZ_[(None, tile * 36 + cb)], sZ1_[(None, empty.index)], tma_bar_ptr=empty.barrier, cache_policy=cpol)
        z1_prod.tail()
        z0_prod.tail()

    if warp_idx == DINV_WARP:   # whole warp (elect.sync inside acquire)
        dl = cute.make_layout((144, 72), stride=(72, 1))
        gDv = cute.group_modes(cute.local_tile(mDv, (144, 72), (None, 0)), 0, 2)
        sDv_, gDv_ = cpasync.tma_partition(tma_dv, 0, cute.make_layout(1), cute.group_modes(cute.make_tensor(sDv.iterator, dl), 0, 2), gDv)
        for it in cutlass.range(cluster_id, num_pairs, nclusters, unroll=1):
            tile = blk0 + (it - cluster_id) // nclusters
            d_empty = d_prod.acquire_and_advance()
            cute.copy(tma_dv, gDv_[(None, tile)], sDv_, tma_bar_ptr=d_empty.barrier)
        d_prod.tail()

    if warp_idx == MMA_WARP:
        for it in cutlass.range(cluster_id, num_pairs, nclusters, unroll=1):
            y_empty = y_prod.acquire_and_advance()
            acc_prod = z_ready(acc_prod)                                                                    # m=1
            acc_prod, w_cons, ab3_cons, mma2, mma3 = conv2_rows(c, acc_prod, w_cons, ab3_cons, mma2, mma3, tV1[2], 4, True, True)
            acc_prod = z_ready(acc_prod)                                                                    # m=2
            ab3_cons, mma3 = red_only(c, ab3_cons, mma3, False)   # deferred last m1 reduction
            acc_prod, w_cons, ab3_cons, mma2, mma3 = conv2_rows(c, acc_prod, w_cons, ab3_cons, mma2, mma3, tV2[2], 2, False, True)
            acc_prod = z_ready(acc_prod)                                                                    # m=0
            ab3_cons, mma3 = red_only(c, ab3_cons, mma3, False)   # deferred last m2 reduction
            acc_prod, w_cons, ab3_cons, mma2, mma3 = conv2_rows(c, acc_prod, w_cons, ab3_cons, mma2, mma3, tV0[2], 3, False)
            y_empty.commit()

    # ======================= epilogue (WG w = warps 4w..4w+3, z channels [64w, 64w+64)) =======================
    def mine(w, q):
        """channel quarter q (32 cols) belongs to WG w"""
        return q // 2 == w

    def gidx(gs):
        """compact register index of gate sub-tile gs (0..7) inside this WG's rGate"""
        return (gs // 4) * 2 + (gs % 4) % 2

    def zread2(c, zc, zbase, dst):
        full = zc.wait_and_advance()
        sv = cute.make_tensor(c.sZ.iterator + (full.index + zbase) * 4096 + c.lane * 8,
                              cute.make_layout((8, 4), stride=(1, 1024)))
        cute.autovec_copy(sv, cute.make_tensor(dst.iterator, cute.make_layout((8, 4))))
        full.release()
        return zc

    def gate_zw(c, w, acc_cons, zc, n_sub, mode):
        """z = act(zp) * g (FP16) into TMEM [0, n_sub * 16); mode 'm1' / 'm2' gate, 's0': SiLU (rows M0) | gate"""
        acc_full = acc_cons.wait_and_advance()
        for sb in cutlass.range_constexpr(n_sub):
            if cutlass.const_expr(mine(w, sb % 4)):
                zc = zread2(c, zc, w * ZS, c.rS)
                c.rAccf.store(c.rS.load().to(acc_dtype))   # zp includes the conv1 bias
                if cutlass.const_expr(mode == "s0" and sb < 4):
                    vf = c.rAccf.load()
                    c.r16_flat.store((vf * (0.5 + 0.5 * cute.math.tanh(0.5 * vf, fastmath=True))).to(io_dtype))  # SiLU
                else:
                    if cutlass.const_expr(mode == "m1"):
                        go = ((sb // 4) % 2) * 128 + (sb % 4) * SUB_COLS
                    elif cutlass.const_expr(mode == "m2"):
                        go = 128 + (sb % 4) * SUB_COLS
                    else:
                        go = ((sb // 4) - 1) * 128 + (sb % 4) * SUB_COLS
                    gv = cute.local_tile(c.rGate, (SUB_COLS,), (gidx(go // SUB_COLS),)).load().to(acc_dtype)
                    c.r16_flat.store((c.rAccf.load() * gv).to(io_dtype))
                cute.copy(c.st_copy, c.r16_32, c.tDZ[None, None, sb])
        cute.arch.fence_view_async_tmem_store()
        acc_full.release()
        return acc_cons, zc

    def w_zero(c, st):
        for j in cutlass.range_constexpr(16):
            c.r16c[j] = c.zero16
        for h in cutlass.range_constexpr(4):
            cute.autovec_copy(c.r16c, cute.local_tile(c.sA3[((None, c.kk), 0, c.kb_, st)], (16,), (h,)))

    def w_gt(c, st):
        for sb in cutlass.range_constexpr(4):
            qv = cute.local_tile(c.rQ, (SUB_COLS,), (sb,))
            for h in cutlass.range_constexpr(2):
                c0 = sb * SUB_COLS + h * 16
                cute.autovec_copy(cute.local_tile(qv, (16,), (h,)),
                                  cute.local_tile(c.sB3[(((None, c0 // 64), c.kk), 0, c.kb_, st)], (16,), ((c0 % 64) // 16,)))

    def w_coef(c, st, slot_base, coef, rows):
        for hh in cutlass.range_constexpr(2):
            h = slot_base // 16 + hh * ((slot_base + 8) // 16 - slot_base // 16)
            for j in cutlass.range_constexpr(16):
                v16 = c.zero16
                for i_ in rows:
                    v16 = cutlass.select_(slot_base + i_ == h * 16 + j, coef[i_], v16)
                c.r16c[j] = v16
            cute.autovec_copy(c.r16c, cute.local_tile(c.sA3[((None, c.kk), 0, c.kb_, st)], (16,), (h,)))

    def reduce_roww(c, w, acc_cons, ab3_prod, slot_base, lr, rows, bias_col0, zero_first, cpf):
        """one M-row: WG0 reads Q (TMEM) and writes G^T, WG1 writes the coefficients (from cpf)"""
        if cutlass.const_expr(w == 1):
            coef = {i_: cpf[(lr, i_)] for i_ in rows}
        acc_full = acc_cons.wait_and_advance()
        for sb in cutlass.range_constexpr(4):
            if cutlass.const_expr(w == 0):
                cute.copy(c.ld_copy, c.tDQ[None, None, sb], c.rQ4[sb])
        cute.arch.fence_view_async_tmem_load()   # one tcgen05.wait::ld for the whole row
        acc_full.release()
        for sb in cutlass.range_constexpr(4):
            if cutlass.const_expr(w == 0):
                qf = cute.make_tensor(c.rQ4[sb].iterator, cute.make_layout(SUB_COLS))
                if cutlass.const_expr(bias_col0 >= 0):
                    qf.store(qf.load() + c.bias_vec(c, 1, bias_col0 + sb * SUB_COLS))
                cute.local_tile(c.rQ, (SUB_COLS,), (sb,)).store(qf.load().to(io_dtype))
        ab3_empty = ab3_prod.acquire_and_advance()
        if cutlass.const_expr(w == 1):
            if cutlass.const_expr(zero_first):
                w_zero(c, 0)
        else:
            w_gt(c, 0)
        if cutlass.const_expr(w == 1):
            w_coef(c, 0, slot_base, coef, rows)
        cute.arch.fence_proxy("async.shared", space="cta")
        ab3_empty.commit()
        return acc_cons, ab3_prod

    def epi_role(c, w, acc_cons, zc, ab3_prod, y_cons, d_c):
        cluster_id = c.cluster_id
        num_pairs = c.num_pairs
        nclusters = c.nclusters
        blk0 = c.blk0
        mSlot = c.mSlot
        rGate = c.rGate
        lane = c.lane
        ld_copy = c.ld_copy
        tDY = c.tDY
        rAcc = c.rAcc
        mTileNodes = c.mTileNodes
        mNodeOut = c.mNodeOut
        for it in cutlass.range(cluster_id, num_pairs, nclusters, unroll=1):
            tile = blk0 + (it - cluster_id) // nclusters
            e = tile * TILE_M + lane
            slot_base = mSlot[e] * 9
            cpf = None
            if cutlass.const_expr(w == 1):   # the coefficient WG: all 35 Dinv coefficients of this edge from SMEM
                d_full = d_c.wait_and_advance()
                cpf = {}
                for lr_, rows_ in M1_ROWS + M2_ROWS + M0_ROWS:
                    for i_ in rows_:
                        cpf[(lr_, i_)] = c.sDv[lane * 81 + i_ * 9 + lr_].to(io_dtype)
                d_full.release()
            # 1. gate (staged sigmoid) -> registers (FP16), this WG's channel half only
            for sb in cutlass.range_constexpr(8):
                if cutlass.const_expr(mine(w, sb % 4)):
                    zc = zread2(c, zc, w * ZS, cute.local_tile(rGate, (SUB_COLS,), (gidx(sb),)))
            # 2. m=1 (rows M3..M6)
            acc_cons, zc = gate_zw(c, w, acc_cons, zc, 16, "m1")
            for q in cutlass.range_constexpr(4):
                acc_cons, ab3_prod = reduce_roww(c, w, acc_cons, ab3_prod, slot_base, M1_ROWS[q][0], M1_ROWS[q][1], -1, q == 0, cpf)
            # 3. m=2 (rows M7, M8)
            acc_cons, zc = gate_zw(c, w, acc_cons, zc, 8, "m2")
            for q in cutlass.range_constexpr(2):
                acc_cons, ab3_prod = reduce_roww(c, w, acc_cons, ab3_prod, slot_base, M2_ROWS[q][0], M2_ROWS[q][1], -1, False, cpf)
            # 4. m=0 (rows M0..M2), conv2 bias c0
            acc_cons, zc = gate_zw(c, w, acc_cons, zc, 12, "s0")
            for q in cutlass.range_constexpr(3):
                acc_cons, ab3_prod = reduce_roww(c, w, acc_cons, ab3_prod, slot_base, M0_ROWS[q][0], M0_ROWS[q][1], q * 128, False, cpf)
            # 5. node messages (Y^T sub-tile sb = rows 32 sb .. 32 sb + 31, read by WG sb)
            y_full = y_cons.wait_and_advance()
            for sb in cutlass.range_constexpr(2):
                if cutlass.const_expr(sb == w):
                    cute.copy(ld_copy, tDY[None, None, sb], rAcc)
                    for k in cutlass.range_constexpr(SUB_COLS):
                        n = sb * SUB_COLS + k
                        if cutlass.const_expr(n < 9 * MAX_SLOTS):
                            node = mTileNodes[tile, n // 9]
                            if node >= 0:
                                mNodeOut[node, n % 9, lane] = rAcc[k]
            y_full.release()

    c.cluster_id = cluster_id
    c.num_pairs = num_pairs
    c.nclusters = nclusters
    c.blk0 = blk0
    c.mSlot = mSlot
    c.tDY = tDY
    c.mTileNodes = mTileNodes
    c.mNodeOut = mNodeOut
    c.sDv = sDv
    if warp_idx < 4:
        set_regs(REGS_EPI, REGS_ENTRY)
        epi_role(c, 0, acc_cons, z0_cons, ab3_prod, y_cons, d_cons)
    if warp_idx >= 4 and warp_idx < 8:
        set_regs(REGS_EPI, REGS_ENTRY)
        epi_role(c, 1, acc_cons, z1_cons, ab3_prod, y_cons, d_cons)

    tmem.relinquish_alloc_permit()
    all_bar.arrive_and_wait()
    tmem.free(tmem_ptr)


@cute.jit
def k2_host(v1: cute.Tensor, v2: cute.Tensor, v0: cute.Tensor, b0: cute.Tensor, c0: cute.Tensor,
            dinv: cute.Tensor, slot: cute.Tensor, tile_nodes: cute.Tensor, node_out: cute.Tensor, cnt: cute.Tensor,
            save_z: cute.Tensor, save_g: cute.Tensor, num_ctas: cutlass.Int32, stream: cuda.CUstream):
    K, MN = tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN

    def make(a_src, n, am=K, bm=K):
        return cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            io_dtype, acc_dtype, (128, n, 16), tcgen05.CtaGroup.ONE, a_src, am, bm))

    mma = make(tcgen05.OperandSource.SMEM, 256)     # only for the 256-column TMEM layout of z and the V ring size
    mma2 = make(tcgen05.OperandSource.TMEM, 128)    # conv2: z (TMEM) x V
    mma3 = make(tcgen05.OperandSource.SMEM, 64, MN, MN)   # reduction: G^T x coef
    w_l = sm100_utils.make_smem_layout_b(mma, (TILE_M, 512, W_KB), io_dtype, W_STAGES)
    v_l = sm100_utils.make_smem_layout_b(mma2, (TILE_M, 128, V_KB), io_dtype, NVST)
    z_l = sm100_utils.make_smem_layout_a(mma2, (TILE_M, 128, 512), io_dtype, 1)
    r_l = sm100_utils.make_smem_layout_a(mma3, (TILE_M, 64, 128), io_dtype, 1)
    rb_l = sm100_utils.make_smem_layout_b(mma3, (TILE_M, 64, 128), io_dtype, 1)
    cta_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id,))
    op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)

    def tma_v(t):
        return cute.nvgpu.make_tiled_tma_atom_B(op, t, cute.select(v_l, mode=[0, 1, 2]),
                                                (TILE_M, 128, V_KB), mma2, cta_layout_vmnk.shape)

    a_v1, t_v1 = tma_v(v1)
    a_v2, t_v2 = tma_v(v2)
    a_v0, t_v0 = tma_v(v0)

    def flat2d(t):
        n = cute.size(t)
        return cute.make_tensor(t.iterator, cute.make_layout((n // 256, 256), stride=(256, 1)))
    st_l = cute.make_layout((16, 256), stride=(256, 1))
    a_zt, t_zt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), flat2d(save_z), st_l, (16, 256))
    a_gt, t_gt = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), flat2d(save_g), st_l, (16, 256))
    nd = cute.size(dinv)
    dv2 = cute.make_tensor(dinv.iterator, cute.make_layout((nd // 72, 72), stride=(72, 1)))   # [Ep*9/8, 72]: tile = 144 rows
    a_dv, t_dv = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), dv2, cute.make_layout((144, 72), stride=(72, 1)), (144, 72))
    k2_kernel(mma, mma2, mma3, a_v1, t_v1, a_v2, t_v2, a_v0, t_v0, b0, c0, slot, tile_nodes, node_out, cnt,
              a_zt, t_zt, a_gt, t_gt, a_dv, t_dv, w_l, v_l, z_l, r_l, rb_l, cta_layout_vmnk).launch(
        grid=(num_ctas, 1, 1), block=(NTHREADS, 1, 1), cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)


def k2_cute(v1, v2, v0, b0, c0, dinv, slot, tile_nodes, node_out, cnt, save_z, save_g, num_ctas):
    """K2 launch on torch's current stream (CUDA-graph capturable). v1 / v2 / v0: conv2 weights [4 / 2 / 3 * 128, 512]
    FP16 (M-row blocks), b0 [640] / c0 [384] fp32, dinv = envelope * D^-1 [Ep, 81] fp32 (zero padding rows),
    slot [Ep] int32, tile_nodes [Ep/128, MAX_SLOTS] int32 (-1 = none), node_out [N, 9, C] fp32 (zero-initialised:
    rows of nodes without edges stay 0), cnt = device tile count, save_z / save_g = K1's ilv FP16 activations."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    tensors = (v1, v2, v0, b0, c0, dinv, slot, tile_nodes, node_out, cnt, save_z, save_g)
    cu = [from_dlpack(t, assumed_align=16) for t in tensors] + [num_ctas]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = ("k2",) + tuple(tuple(t.shape) for t in tensors)
    if key not in KERNELS:
        KERNELS[key] = cute.compile(k2_host, *cu, stream)
    KERNELS[key](*cu, stream)
