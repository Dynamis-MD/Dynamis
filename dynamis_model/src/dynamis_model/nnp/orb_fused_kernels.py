"""Blackwell (sm_100a) CuTeDSL + Triton fused kernels for Orb-v3 Direct inference.

Implements four fused stages while keeping coordinates, displacements, cutoffs,
node states, scatter accumulators (sent_agg/recv_agg), heads, and ZBL in FP32/FP64
(BF16 only for edge/node MLP GEMMs and stored edge embeddings — NO FP8):

  1. Stage 4 (Triton): `orb_geom_zbl_featurize_kernel`
     Fuses PBC displacement, distance, polynomial attention & encoder cutoffs,
     exact analytic ZBL energy (FP64) & forces (FP32), Bessel(8), lmax=3
     component-normalized SphericalHarmonics(16), and outer product -> [E, 128] BF16.

  2. Stage 2 (CuTeDSL sm_100a): `FusedOrbMlp3Layer`
     Fuses 3-layer MLP (D_in -> 1024 -> 1024 -> 256 + FP32 RMSNorm) into a single
     persistent warp-specialized 2-CTA tcgen05.mma kernel with zero HBM spills:
     B_stg=4, resident X0 (D_in <= 256 in sXH2), 4-stage h1_pipe, double-buffered
     acc_out, vectorized LDG.128 bias/gamma, MUFU.TANH SiLU, Register RMSNorm,
     and async hardware 2D TMA Store (UTMASTG.2D). Used for node_enc (256),
     edge_enc (128), and node_mlp (768).

  3. Stage 3 (CuTeDSL sm_100a): `FusedOrbEdgeLayer`
     Fuses the full AttentionInteractionNetwork edge pass into ONE persistent
     2-CTA tcgen05.mma kernel with zero intermediate [N, 1024] HBM buffers:
       - Coalesced 4x-MLP fused-gather of `nodes_bf16 [N, 256]` directly into
         SMEM (`K_SW128` `sH1[8..15]`), overlapping 4 back-to-back receiver
         `LDG.128` loads with sender-deduplicated loads and Warp 5 `We[0] @ E`
       - Resident single-load E [64, 256] in SMEM + contiguous `n0`-outer W0 +
         B_stg=4 + 4-stage h1_pipe + double-buffered `acc_out` (`tCtAccOut0/1`)
       - Inline FP32 RMSNorm + sigmoid attention gates (recv_attn, send_attn) * cutoff
       - In-place edge residual (edges += upd_e via coalesced 128-bit STG.128)
       - 16-row register-accumulated contiguous-sender scatter (~16x fewer atomics)
         and coalesced receiver scatter using Blackwell hardware 128-bit FP32x4
         atomic `red.relaxed.gpu.global.add.v4.f32` (REDG.E.ADD.F32x4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

try:
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute
    import cutlass.memory as memory
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.tensor_utils import LayoutEnum
    from cutlass.cute.nvgpu import cpasync, tcgen05
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import dsl_user_op, Float32, Int64, T
    from cutlass._mlir.dialects import llvm

    _HAS_CUTEDSL = True
except ImportError:
    _HAS_CUTEDSL = False


def is_sm100_fused_orb_available(device: torch.device | str | None = None) -> bool:
    """Return True if Blackwell (sm_100+) CUDA + CuTeDSL + Triton are available."""
    if not (_HAS_CUTEDSL and _HAS_TRITON):
        return False
    if not torch.cuda.is_available():
        return False
    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(dev)
    return major == 10


# =============================================================================
# Stage 4: Fused Triton Kernel (Geometry + ZBL + Bessel(8) + lmax=3 SH(16))
# =============================================================================

if _HAS_TRITON:

    @triton.jit
    def orb_geom_zbl_featurize_kernel(
        pos_ptr,            # [N_total, 3] fp32
        atom_types_ptr,     # [N_total] int64
        snd_ptr,            # [E] int64
        rcv_ptr,            # [E] int64
        off_ptr,            # [E, 3] int32 or fp32
        cell_ptr,           # [3, 3] fp32
        cov_radii_ptr,      # [119] fp32
        bessel_w_ptr,       # [8] fp32
        cutoff_out_ptr,     # [E] fp32
        ef_out_ptr,         # [E, 128] bf16
        zbl_e_blk_ptr,      # [num_blocks] fp64
        zbl_f_ptr,          # [n_local, 3] fp32
        E,
        n_local,
        model_radius,       # float (e.g. 6.0)
        bessel_prefactor,   # float (sqrt(2 / 6.0))
        zbl_a_prefactor,    # float (0.4543)
        zbl_a_exp,          # float (0.3)
        zbl_p,              # float (6.0)
        zbl_c0, zbl_c1, zbl_c2, zbl_c3,
        zbl_d0, zbl_d1, zbl_d2, zbl_d3,
        HAS_ZBL: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        pid = tl.program_id(0)
        e_idx = pid.to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E).to(tl.int64)
        mask = e_idx < E

        c00 = tl.load(cell_ptr + 0); c01 = tl.load(cell_ptr + 1); c02 = tl.load(cell_ptr + 2)
        c10 = tl.load(cell_ptr + 3); c11 = tl.load(cell_ptr + 4); c12 = tl.load(cell_ptr + 5)
        c20 = tl.load(cell_ptr + 6); c21 = tl.load(cell_ptr + 7); c22 = tl.load(cell_ptr + 8)

        snd = tl.load(snd_ptr + e_idx, mask=mask, other=0)
        rcv = tl.load(rcv_ptr + e_idx, mask=mask, other=0)

        off0 = tl.load(off_ptr + e_idx * 3 + 0, mask=mask, other=0).to(tl.float32)
        off1 = tl.load(off_ptr + e_idx * 3 + 1, mask=mask, other=0).to(tl.float32)
        off2 = tl.load(off_ptr + e_idx * 3 + 2, mask=mask, other=0).to(tl.float32)

        shx = off0 * c00 + off1 * c10 + off2 * c20
        shy = off0 * c01 + off1 * c11 + off2 * c21
        shz = off0 * c02 + off1 * c12 + off2 * c22

        px_s = tl.load(pos_ptr + snd * 3 + 0, mask=mask, other=0.0)
        py_s = tl.load(pos_ptr + snd * 3 + 1, mask=mask, other=0.0)
        pz_s = tl.load(pos_ptr + snd * 3 + 2, mask=mask, other=0.0)

        px_r = tl.load(pos_ptr + rcv * 3 + 0, mask=mask, other=0.0)
        py_r = tl.load(pos_ptr + rcv * 3 + 1, mask=mask, other=0.0)
        pz_r = tl.load(pos_ptr + rcv * 3 + 2, mask=mask, other=0.0)

        vx = px_r - px_s - shx
        vy = py_r - py_s - shy
        vz = pz_r - pz_s - shz

        r2 = vx * vx + vy * vy + vz * vz
        r = tl.sqrt(r2)
        inv_r = tl.where(r > 1e-12, 1.0 / r, 0.0)
        x = vx * inv_r
        y = vy * inv_r
        z = vz * inv_r

        # 1. Attention cutoff (p=4, r_max=model_radius)
        u = r / model_radius
        u2 = u * u
        u4 = u2 * u2
        env_attn = 1.0 - 15.0 * u4 + 24.0 * u4 * u - 10.0 * u4 * u2
        cut_attn = tl.where(r < model_radius, env_attn, 0.0)
        tl.store(cutoff_out_ptr + e_idx, cut_attn, mask=mask)

        # Encoder cutoff (p=4, r_max=6.0 hardcoded in orb_models.nn_util.get_cutoff)
        u6 = r / 6.0
        u6_2 = u6 * u6
        u6_4 = u6_2 * u6_2
        env_enc = 1.0 - 15.0 * u6_4 + 24.0 * u6_4 * u6 - 10.0 * u6_4 * u6_2
        cut_enc = tl.where(r < 6.0, env_enc, 0.0)

        # 2. Exact analytic ZBL energy (receiver-local, FP64) & forces (union -> local, FP32)
        if HAS_ZBL:
            zu_i = tl.load(atom_types_ptr + snd, mask=mask, other=0) + 1
            zv_i = tl.load(atom_types_ptr + rcv, mask=mask, other=0) + 1
            zu = zu_i.to(tl.float32)
            zv = zv_i.to(tl.float32)

            r_max_zbl = (
                tl.load(cov_radii_ptr + zu_i, mask=mask, other=1.0)
                + tl.load(cov_radii_ptr + zv_i, mask=mask, other=1.0)
            )
            in_zbl = mask & (r < r_max_zbl)
            r_safe = tl.where(in_zbl, r, 1.0)

            a = (zbl_a_prefactor * 0.529) / (
                tl.exp(zbl_a_exp * tl.log(zu)) + tl.exp(zbl_a_exp * tl.log(zv))
            )
            r_over_a = r_safe / a
            e0 = tl.exp(-zbl_d0 * r_over_a)
            e1 = tl.exp(-zbl_d1 * r_over_a)
            e2 = tl.exp(-zbl_d2 * r_over_a)
            e3 = tl.exp(-zbl_d3 * r_over_a)

            phi = zbl_c0 * e0 + zbl_c1 * e1 + zbl_c2 * e2 + zbl_c3 * e3
            dphi_dr = (
                -zbl_d0 * zbl_c0 * e0
                - zbl_d1 * zbl_c1 * e1
                - zbl_d2 * zbl_c2 * e2
                - zbl_d3 * zbl_c3 * e3
            ) / a

            coulomb = 14.3996 * zu * zv / r_safe
            v_raw = coulomb * phi

            rr = r_safe / r_max_zbl
            p = zbl_p
            rr_pm1 = tl.exp((p - 1.0) * tl.log(tl.maximum(rr, 1e-30)))
            rr_p = rr_pm1 * rr
            rr_p1 = rr_p * rr
            rr_p2 = rr_p1 * rr

            c_p0 = 0.5 * (p + 1.0) * (p + 2.0)
            c_p1 = p * (p + 2.0)
            c_p2 = 0.5 * p * (p + 1.0)

            env_z = tl.where(in_zbl, 1.0 - c_p0 * rr_p + c_p1 * rr_p1 - c_p2 * rr_p2, 0.0)
            inv_rmax = 1.0 / r_max_zbl
            denv_dr = tl.where(
                in_zbl,
                (-c_p0 * p * rr_pm1 + c_p1 * (p + 1.0) * rr_p - c_p2 * (p + 2.0) * rr_p1) * inv_rmax,
                0.0,
            )

            v_edge = 0.5 * v_raw * env_z
            rcv_loc = in_zbl & (rcv < n_local)
            blk_e = tl.sum(tl.where(rcv_loc, v_edge.to(tl.float64), 0.0))
            tl.store(zbl_e_blk_ptr + pid, blk_e)

            dv_dr_raw = (-coulomb / r_safe) * phi + coulomb * dphi_dr
            f_mag = -0.5 * (dv_dr_raw * env_z + v_raw * denv_dr)
            fx = tl.where(in_zbl, f_mag * x, 0.0)
            fy = tl.where(in_zbl, f_mag * y, 0.0)
            fz = tl.where(in_zbl, f_mag * z, 0.0)

            snd_loc = in_zbl & (snd < n_local)
            tl.atomic_add(zbl_f_ptr + snd * 3 + 0, -fx, mask=snd_loc)
            tl.atomic_add(zbl_f_ptr + snd * 3 + 1, -fy, mask=snd_loc)
            tl.atomic_add(zbl_f_ptr + snd * 3 + 2, -fz, mask=snd_loc)

            tl.atomic_add(zbl_f_ptr + rcv * 3 + 0, fx, mask=rcv_loc)
            tl.atomic_add(zbl_f_ptr + rcv * 3 + 1, fy, mask=rcv_loc)
            tl.atomic_add(zbl_f_ptr + rcv * 3 + 2, fz, mask=rcv_loc)

        # 3. Spherical Harmonics lmax=3, normalize=True, normalization="component"
        SQRT3 = 1.7320508075688772
        SQRT5 = 2.23606797749979
        SQRT7 = 2.6457513110645907
        SQRT5_6 = 0.9128709291752769
        SQRT3_8 = 0.6123724356957945

        y2 = y * y
        x2 = x * x
        z2 = z * z
        x2z2 = x2 + z2

        sh20 = SQRT3 * x * z
        sh21 = SQRT3 * x * y
        sh22 = y2 - 0.5 * x2z2
        sh23 = SQRT3 * y * z
        sh24 = (0.5 * SQRT3) * (z2 - x2)

        t_4y2 = 4.0 * y2 - x2z2
        sh30 = SQRT5_6 * (sh20 * z + sh24 * x)
        sh31 = SQRT5 * sh20 * y
        sh32 = SQRT3_8 * t_4y2 * x
        sh33 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
        sh34 = SQRT3_8 * z * t_4y2
        sh35 = SQRT5 * sh24 * y
        sh36 = SQRT5_6 * (sh24 * z - sh20 * x)

        s0 = tl.full([BLOCK_E], 1.0, dtype=tl.float32)
        s1 = SQRT3 * x
        s2 = SQRT3 * y
        s3 = SQRT3 * z
        s4 = SQRT5 * sh20
        s5 = SQRT5 * sh21
        s6 = SQRT5 * sh22
        s7 = SQRT5 * sh23
        s8 = SQRT5 * sh24
        s9 = SQRT7 * sh30
        s10 = SQRT7 * sh31
        s11 = SQRT7 * sh32
        s12 = SQRT7 * sh33
        s13 = SQRT7 * sh34
        s14 = SQRT7 * sh35
        s15 = SQRT7 * sh36

        # 4. Bessel(8) * cut_enc * outer(SH(16)) -> [E, 128] BF16
        inv_r_bessel = bessel_prefactor * inv_r * cut_enc
        base_out = ef_out_ptr + e_idx * 128

        for b in tl.static_range(8):
            bw = tl.load(bessel_w_ptr + b)
            rb = tl.sin(bw * r) * inv_r_bessel
            off_b = base_out + b * 16
            tl.store(off_b + 0, (rb * s0).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 1, (rb * s1).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 2, (rb * s2).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 3, (rb * s3).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 4, (rb * s4).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 5, (rb * s5).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 6, (rb * s6).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 7, (rb * s7).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 8, (rb * s8).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 9, (rb * s9).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 10, (rb * s10).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 11, (rb * s11).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 12, (rb * s12).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 13, (rb * s13).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 14, (rb * s14).to(tl.bfloat16), mask=mask)
            tl.store(off_b + 15, (rb * s15).to(tl.bfloat16), mask=mask)


# =============================================================================
# Stage 2 & Stage 3: Blackwell sm_100a CuTeDSL Kernels
# =============================================================================

if _HAS_CUTEDSL:

    @dsl_user_op
    def _tanh_approx_f32(a: Float32, *, loc=None, ip=None) -> Float32:
        """Single-instruction Blackwell SFU tanh.approx.f32 (MUFU.TANH)."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip)],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _red_global_add_f32x4(
        addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None
    ) -> None:
        """Blackwell sm_100a hardware 128-bit FP32x4 atomic global add (REDG.E.ADD.F32x4)."""
        llvm.inline_asm(
            None,
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
                Float32(d).ir_value(loc=loc, ip=ip),
            ],
            "red.relaxed.gpu.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
            "l,f,f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    class FusedOrbMlp3Layer:
        """Persistent 2-CTA Blackwell tcgen05 fused 3-layer MLP (D_in -> 1024 -> 1024 -> 256 + RMSNorm)."""

        def __init__(self, d_in: int, num_sms: int = 152, num_b_stages: int = 4):
            assert d_in % 64 == 0
            self.d_in = d_in
            self.k0_tiles = d_in // 64
            self.resident_x0 = self.k0_tiles <= 4
            self.h_dim = 1024
            self.d_out = 256
            self.num_b_stages = num_b_stages
            self.num_xh2_stages = 4
            self.num_h1_stages = 16
            self.max_clusters = max(1, num_sms // 2)

            self.ab_dtype = cutlass.BFloat16
            self.acc_dtype = cutlass.Float32
            self.c_dtype = cutlass.BFloat16

            self.use_2cta_instrs = True
            self.cta_group = tcgen05.CtaGroup.TWO
            self.cluster_shape_mn = (2, 1)
            self.mma_tiler = (128, 256, 64)
            self.cta_tile_shape_mnk = (64, 256, 64)
            self.epi_threads = 128
            self.threads_per_cta = 192
            self.a_major_mode = cute.nvgpu.OperandMajorMode.K
            self.b_major_mode = cute.nvgpu.OperandMajorMode.K
            self.c_layout = LayoutEnum.ROW_MAJOR

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mW0: cute.Tensor,
            mB0: cute.Tensor,
            mW1: cute.Tensor,
            mB1: cute.Tensor,
            mW2: cute.Tensor,
            mB2: cute.Tensor,
            mGamma: cute.Tensor,
            mY: cute.Tensor,
            eps: cutlass.Float32,
            M_orig: cutlass.Int32,
            stream: cuda_drv.CUstream,
        ):
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.ab_dtype, self.ab_dtype,
                self.a_major_mode, self.b_major_mode,
                self.acc_dtype, self.cta_group, self.mma_tiler[:2],
            )
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,),
            )

            xh2_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.ab_dtype, self.num_xh2_stages
            )
            h1_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.ab_dtype, self.num_h1_stages
            )
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.ab_dtype, self.num_b_stages
            )

            a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
            x0_smem_layout = cute.select(xh2_smem_layout_staged, mode=[0, 1, 2])
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])

            tma_atom_x, tma_tensor_x = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mX, x0_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
            )
            tma_atom_w0, tma_tensor_w0 = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mW0, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
            )
            tma_atom_w1, tma_tensor_w1 = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mW1, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
            )
            tma_atom_w2, tma_tensor_w2 = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mW2, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
            )

            y_epi_smem_layout = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, (64, 64), 1)
            y_epi_smem_2d = cute.select(y_epi_smem_layout, mode=[0, 1])
            tma_atom_y, tma_tensor_y = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mY, y_epi_smem_2d, (64, 64)
            )

            tx_x_single = cute.size_in_bytes(self.ab_dtype, x0_smem_layout) * atom_thr_size
            tx_x_bytes = tx_x_single * self.k0_tiles if self.resident_x0 else tx_x_single
            tx_b_bytes = cute.size_in_bytes(self.ab_dtype, b_smem_layout) * atom_thr_size

            epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
            cols_per_acc = memory.get_num_tmem_alloc_cols(tCtAcc_fake)
            num_tmem_alloc_cols = cols_per_acc * 4

            total_clusters_m = cute.ceil_div(mX.shape[0], 128)
            num_launch_clusters = cutlass.min(total_clusters_m, self.max_clusters)
            grid = (num_launch_clusters * 2, 1, 1)

            self.kernel(
                tiled_mma,
                tma_atom_x, tma_tensor_x,
                tma_atom_w0, tma_tensor_w0, mB0,
                tma_atom_w1, tma_tensor_w1, mB1,
                tma_atom_w2, tma_tensor_w2, mB2,
                mGamma, tma_atom_y, tma_tensor_y,
                eps, M_orig, total_clusters_m, num_launch_clusters,
                cluster_layout_vmnk,
                xh2_smem_layout_staged, h1_smem_layout_staged, b_smem_layout_staged,
                epi_tile, tx_x_bytes, tx_b_bytes, cols_per_acc, num_tmem_alloc_cols,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_x: cute.CopyAtom, mX_mkl: cute.Tensor,
            tma_atom_w0: cute.CopyAtom, mW0_nkl: cute.Tensor, mB0: cute.Tensor,
            tma_atom_w1: cute.CopyAtom, mW1_nkl: cute.Tensor, mB1: cute.Tensor,
            tma_atom_w2: cute.CopyAtom, mW2_nkl: cute.Tensor, mB2: cute.Tensor,
            mGamma: cute.Tensor, tma_atom_y: cute.CopyAtom, mY_tma: cute.Tensor,
            eps: cutlass.Float32, M_orig: cutlass.Int32,
            total_clusters_m: cutlass.Int32, num_launch_clusters: cutlass.Int32,
            cluster_layout_vmnk: cute.Layout,
            xh2_smem_layout_staged: cute.ComposedLayout,
            h1_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_tile: cute.Tile,
            tx_x_bytes: cutlass.Constexpr, tx_b_bytes: cutlass.Constexpr,
            cols_per_acc: cutlass.Constexpr, num_tmem_alloc_cols: cutlass.Constexpr,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 4:
                cpasync.prefetch_descriptor(tma_atom_x)
                cpasync.prefetch_descriptor(tma_atom_w0)
                cpasync.prefetch_descriptor(tma_atom_w1)
                cpasync.prefetch_descriptor(tma_atom_w2)
                cpasync.prefetch_descriptor(tma_atom_y)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            mma_tile_coord_v = bidx % 2
            is_leader_cta = mma_tile_coord_v == 0
            cluster_id = bidx // 2
            x0_pipe_stages = 1 if self.resident_x0 else self.num_xh2_stages

            @cute.struct
            class SharedStorage:
                x0_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 8]
                b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stages * 2]
                acc_tmp_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
                acc_out_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
                h1_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 8]
                h2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                x0_free_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                h2_free_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                rms_half: cute.struct.MemRange[cutlass.Float32, 128]

            smem = memory.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            x0_prod, x0_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.x0_mbar_ptr.data_ptr(), num_stages=x0_pipe_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=tx_x_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            ).make_participants()
            b_prod, b_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_mbar_ptr.data_ptr(), num_stages=self.num_b_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=tx_b_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            ).make_participants()
            acc_tmp_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_tmp_mbar_ptr.data_ptr(), num_stages=2,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            acc_tmp_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_tmp_cmt = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_tmp_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            acc_out_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_out_mbar_ptr.data_ptr(), num_stages=2,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            acc_out_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_out_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            h1_pipe = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.h1_mbar_ptr.data_ptr(), num_stages=4,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h1_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 4)
            h1_cwait_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 4)
            h1_crel_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 4)
            h2_pipe = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.h2_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h2_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            h2_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            x0_free_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.x0_free_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            x0_free_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            x0_free_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h2_free_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.h2_free_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h2_free_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            h2_free_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.threads_per_cta)
            tmem = memory.TmemAllocator(
                storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
                allocator_warp_id=0, is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            )
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sH1 = smem.allocate_tensor(self.ab_dtype, h1_smem_layout_staged.outer, 128, h1_smem_layout_staged.inner)
            sXH2 = smem.allocate_tensor(self.ab_dtype, xh2_smem_layout_staged.outer, 128, xh2_smem_layout_staged.inner)
            sB = smem.allocate_tensor(self.ab_dtype, b_smem_layout_staged.outer, 128, b_smem_layout_staged.inner)

            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

            gX_mkl = cute.local_tile(mX_mkl, (128, 64), (None, None, None))
            gW0_nkl = cute.local_tile(mW0_nkl, (256, 64), (None, None, None))
            gW1_nkl = cute.local_tile(mW1_nkl, (256, 64), (None, None, None))
            gW2_nkl = cute.local_tile(mW2_nkl, (256, 64), (None, None, None))

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgX = thr_mma.partition_A(gX_mkl)
            tCgW0 = thr_mma.partition_B(gW0_nkl)
            tCgW1 = thr_mma.partition_B(gW1_nkl)
            tCgW2 = thr_mma.partition_B(gW2_nkl)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            tAsX, tAgX_all = cpasync.tma_partition(tma_atom_x, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sXH2, 0, 3), cute.group_modes(tCgX, 0, 3))
            tBsW, tBgW0_all = cpasync.tma_partition(tma_atom_w0, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW0, 0, 3))
            _, tBgW1_all = cpasync.tma_partition(tma_atom_w1, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW1, 0, 3))
            _, tBgW2_all = cpasync.tma_partition(tma_atom_w2, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW2, 0, 3))

            tCrA_XH2 = tiled_mma.make_fragment_A(sXH2)
            tCrA_H1 = tiled_mma.make_fragment_A(sH1)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()

            tCtAccTmp0 = cute.make_tensor(tmem_ptr_base, tCtAcc_fake.layout)
            tCtAccTmp1 = cute.make_tensor(tmem_ptr_base + cols_per_acc, tCtAcc_fake.layout)
            tCtAccOut0 = cute.make_tensor(tmem_ptr_base + cols_per_acc * 2, tCtAcc_fake.layout)
            tCtAccOut1 = cute.make_tensor(tmem_ptr_base + cols_per_acc * 3, tCtAcc_fake.layout)

            k0_tiles = self.k0_tiles
            k1_tiles = 16
            num_kblks = cute.size(tCrA_XH2, mode=[2])
            tBgW2_0 = tBgW2_all[(None, 0, None, 0)]

            if warp_idx == 4:
                cm = cluster_id
                tile_i = cutlass.Int32(0)
                while cm < total_clusters_m:
                    if tile_i > 0:
                        x0_free_pipe.consumer_wait(x0_free_cstate)
                        x0_free_pipe.consumer_release(x0_free_cstate)
                        x0_free_cstate.advance()

                    tAgX = tAgX_all[(None, cm, None, 0)]
                    if cutlass.const_expr(self.resident_x0):
                        xh = x0_prod.acquire_and_advance()
                        for k_e in cutlass.range(k0_tiles, unroll_full=True):
                            cute.copy(tma_atom_x, tAgX[(None, k_e)], tAsX[(None, k_e)],
                                      tma_bar_ptr=xh.barrier, mcast_mask=a_full_mcast_mask)
                        for n0 in cutlass.range(4, unroll=1):
                            tBgW0 = tBgW0_all[(None, n0, None, 0)]
                            for k_idx in cutlass.range(k0_tiles, unroll=1):
                                bh = b_prod.acquire_and_advance()
                                cute.copy(tma_atom_w0, tBgW0[(None, k_idx)], tBsW[(None, bh.index)],
                                          tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    else:
                        for n0 in cutlass.range(4, unroll=1):
                            tBgW0 = tBgW0_all[(None, n0, None, 0)]
                            for k_idx in cutlass.range(k0_tiles, unroll=1):
                                xh = x0_prod.acquire_and_advance()
                                cute.copy(tma_atom_x, tAgX[(None, k_idx)], tAsX[(None, xh.index)],
                                          tma_bar_ptr=xh.barrier, mcast_mask=a_full_mcast_mask)
                                bh = b_prod.acquire_and_advance()
                                cute.copy(tma_atom_w0, tBgW0[(None, k_idx)], tBsW[(None, bh.index)],
                                          tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)

                    tBgW1_0 = tBgW1_all[(None, 0, None, 0)]
                    for k_idx in cutlass.range(k1_tiles, unroll=1):
                        bh = b_prod.acquire_and_advance()
                        cute.copy(tma_atom_w1, tBgW1_0[(None, k_idx)], tBsW[(None, bh.index)],
                                  tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    for j_next in cutlass.range(1, 4, unroll=1):
                        tBgW1_j = tBgW1_all[(None, j_next, None, 0)]
                        for k_idx in cutlass.range(k1_tiles, unroll=1):
                            bh = b_prod.acquire_and_advance()
                            cute.copy(tma_atom_w1, tBgW1_j[(None, k_idx)], tBsW[(None, bh.index)],
                                      tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                        j_prev = j_next - 1
                        for k_sub in cutlass.range(4, unroll=1):
                            bh = b_prod.acquire_and_advance()
                            cute.copy(tma_atom_w2, tBgW2_0[(None, j_prev * 4 + k_sub)], tBsW[(None, bh.index)],
                                      tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    for k_sub in cutlass.range(4, unroll=1):
                        bh = b_prod.acquire_and_advance()
                        cute.copy(tma_atom_w2, tBgW2_0[(None, 12 + k_sub)], tBsW[(None, bh.index)],
                                  tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    cm = cm + num_launch_clusters
                    tile_i = tile_i + 1
                x0_prod.tail()
                b_prod.tail()

            if warp_idx == 5:
                if is_leader_cta:
                    cm = cluster_id
                    while cm < total_clusters_m:
                        if cutlass.const_expr(self.resident_x0):
                            xh = x0_cons.wait_and_advance()
                            for n0 in cutlass.range(4, unroll=1):
                                acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                                tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                                acc_tmp_acq.advance()
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for k_idx in cutlass.range(k0_tiles, unroll=1):
                                    bh = b_cons.wait_and_advance()
                                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                                        cute.gemm(tiled_mma, tCtAcc_cur, tCrA_XH2[(None, None, kblk, k_idx)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    bh.release()
                                acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                                acc_tmp_cmt.advance()
                            xh.release()
                        else:
                            for n0 in cutlass.range(4, unroll=1):
                                acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                                tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                                acc_tmp_acq.advance()
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for k_idx in cutlass.range(k0_tiles, unroll=1):
                                    xh = x0_cons.wait_and_advance()
                                    bh = b_cons.wait_and_advance()
                                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                                        cute.gemm(tiled_mma, tCtAcc_cur, tCrA_XH2[(None, None, kblk, xh.index)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    xh.release()
                                    bh.release()
                                acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                                acc_tmp_cmt.advance()

                        h2_free_pipe.producer_acquire(h2_free_pstate)
                        h2_free_pipe.producer_commit(h2_free_pstate)
                        h2_free_pstate.advance()

                        acc_out_pipe.producer_acquire(acc_out_pstate)
                        tCtAcc_out = cute.make_tensor(tmem_ptr_base + cols_per_acc * (2 + acc_out_pstate.index), tCtAcc_fake.layout)

                        acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                        tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                        acc_tmp_acq.advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for sl in cutlass.range(4, unroll=1):
                            h1_pipe.consumer_wait(h1_cwait_state)
                            h1_cwait_state.advance()
                            for k_sub in cutlass.range(4, unroll=1):
                                bh = b_cons.wait_and_advance()
                                k_h1 = sl * 4 + k_sub
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_H1[(None, None, kblk, k_h1)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                        acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                        acc_tmp_cmt.advance()

                        for j_next in cutlass.range(1, 4, unroll=1):
                            acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                            tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                            acc_tmp_acq.advance()
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for k_idx in cutlass.range(k1_tiles, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_H1[(None, None, kblk, k_idx)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                            acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                            acc_tmp_cmt.advance()

                            if j_next == 3:
                                for _sl in cutlass.range(4, unroll_full=True):
                                    h1_pipe.consumer_release(h1_crel_state)
                                    h1_crel_state.advance()

                            h2_pipe.consumer_wait(h2_cstate)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, j_next > 1)
                            for k_sub in cutlass.range(4, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_out, tCrA_XH2[(None, None, kblk, k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_out)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                            h2_pipe.consumer_release(h2_cstate)
                            h2_cstate.advance()

                        h2_pipe.consumer_wait(h2_cstate)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        for k_sub in cutlass.range(4, unroll=1):
                            bh = b_cons.wait_and_advance()
                            for kblk in cutlass.range(num_kblks, unroll_full=True):
                                cute.gemm(tiled_mma, tCtAcc_out, tCrA_XH2[(None, None, kblk, k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_out)
                            bh.release()
                        h2_pipe.consumer_release(h2_cstate)
                        h2_cstate.advance()

                        acc_out_pipe.producer_commit(acc_out_pstate)
                        acc_out_pstate.advance()

                        if cm + num_launch_clusters < total_clusters_m:
                            x0_free_pipe.producer_acquire(x0_free_pstate)
                            x0_free_pipe.producer_commit(x0_free_pstate)
                            x0_free_pstate.advance()

                        cm = cm + num_launch_clusters

                    acc_tmp_pipe.producer_tail(acc_tmp_acq)
                    acc_out_pipe.producer_tail(acc_out_pstate)
                    h2_free_pipe.producer_tail(h2_free_pstate)
                    x0_free_pipe.producer_tail(x0_free_pstate)

            if warp_idx < 4:
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, True
                )
                tAccTmp0_epi = cute.flat_divide(tCtAccTmp0[((None, None), 0, 0)], epi_tile)
                tAccTmp1_epi = cute.flat_divide(tCtAccTmp1[((None, None), 0, 0)], epi_tile)
                tAccOut0_epi = cute.flat_divide(tCtAccOut0[((None, None), 0, 0)], epi_tile)
                tAccOut1_epi = cute.flat_divide(tCtAccOut1[((None, None), 0, 0)], epi_tile)

                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAccTmp0_epi[(None, None, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAccTmp0 = cute.group_modes(thr_copy_t2r.partition_S(tAccTmp0_epi), 3, 5)
                tTR_tAccTmp1 = cute.group_modes(thr_copy_t2r.partition_S(tAccTmp1_epi), 3, 5)
                tTR_tAccOut0 = cute.group_modes(thr_copy_t2r.partition_S(tAccOut0_epi), 3, 5)
                tTR_tAccOut1 = cute.group_modes(thr_copy_t2r.partition_S(tAccOut1_epi), 3, 5)

                cAcc = cute.make_identity_tensor((64, 256))
                tTR_cAcc = cute.group_modes(thr_copy_t2r.partition_D(cute.flat_divide(cAcc, epi_tile)), 3, 5)
                tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0)].shape, self.acc_dtype)
                tTR_rC = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0)].shape, self.c_dtype)
                subtile_cnt = cute.size(tTR_tAccTmp0.shape, mode=[3])

                copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_rC = tiled_copy_r2s.retile(tTR_rC)

                k_sw128_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, self.ab_dtype)
                sH1_4slices_layout = cute.tile_to_shape(k_sw128_atom, (64, 256, 4), order=(0, 1, 2))
                sH1_4slices = cute.make_tensor(
                    cute.recast_ptr(sH1.iterator, sH1_4slices_layout.inner, dtype=self.ab_dtype), sH1_4slices_layout.outer)
                sH2_2d_layout = cute.tile_to_shape(k_sw128_atom, (64, 256), order=(0, 1))
                sH2_2d = cute.make_tensor(
                    cute.recast_ptr(sXH2.iterator, sH2_2d_layout.inner, dtype=self.ab_dtype), sH2_2d_layout.outer)

                tRS_sH2 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH2_2d, epi_tile)), 3, 5)
                tRS_sH1_0 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 0)], epi_tile)), 3, 5)
                tRS_sH1_1 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 1)], epi_tile)), 3, 5)
                tRS_sH1_2 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 2)], epi_tile)), 3, 5)
                tRS_sH1_3 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 3)], epi_tile)), 3, 5)

                sY_tma_layout = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, (64, 64), 4)
                sY_tma = cute.make_tensor(
                    cute.recast_ptr(sH1.iterator, sY_tma_layout.inner, dtype=self.c_dtype), sY_tma_layout.outer)
                gY_tiled = cute.local_tile(mY_tma, (64, 64), (None, None, None))
                bSG_sY, bSG_gY_all = cpasync.tma_partition(
                    tma_atom_y, 0, cute.make_layout(1),
                    cute.group_modes(sY_tma, 0, 2), cute.group_modes(gY_tiled, 0, 2),
                )
                y_tma_pipe = pipeline.PipelineTmaStore.create(
                    num_stages=4, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads),
                )
                sRmsHalf = cute.make_tensor(storage.rms_half.data_ptr(), cute.make_layout((64, 2), stride=(2, 1)))
                epi_sync = pipeline.NamedBarrier(barrier_id=3, num_threads=self.epi_threads)
                half_f32 = cutlass.Float32(0.5); one_f32 = cutlass.Float32(1.0); inv_256 = cutlass.Float32(1.0 / 256.0)

                thr_row_local = tTR_cAcc[(0, 0, 0, 0)][0]
                c_s0 = cute.arch.make_warp_uniform(tTR_cAcc[(0, 0, 0, 0)][1])
                c_s1 = cute.arch.make_warp_uniform(tTR_cAcc[(0, 0, 0, 1)][1])
                c_s2 = cute.arch.make_warp_uniform(tTR_cAcc[(0, 0, 0, 2)][1])
                c_s3 = cute.arch.make_warp_uniform(tTR_cAcc[(0, 0, 0, 3)][1])
                warp_half = cutlass.select_(c_s0 == 0, cutlass.Int32(0), cutlass.Int32(1))

                r_y0 = cute.make_rmem_tensor((32,), cutlass.Float32)
                r_y1 = cute.make_rmem_tensor((32,), cutlass.Float32)
                r_y2 = cute.make_rmem_tensor((32,), cutlass.Float32)
                r_y3 = cute.make_rmem_tensor((32,), cutlass.Float32)
                r_b32 = cute.make_rmem_tensor((32,), cutlass.Float32)

                cm = cluster_id
                tile_i = cutlass.Int32(0)
                while cm < total_clusters_m:
                    cta_m_tile = cm * 2 + mma_tile_coord_v
                    if tile_i > 0:
                        if warp_idx == 0:
                            y_tma_pipe.producer_tail()
                        epi_sync.arrive_and_wait()

                    for n0 in cutlass.range(4, unroll=1):
                        h1_pipe.producer_acquire(h1_pstate)
                        acc_tmp_pipe.consumer_wait(acc_tmp_cstate)
                        n0_col_base = n0 * 256
                        for st_idx in cutlass.range(subtile_cnt, unroll=1):
                            if acc_tmp_cstate.index == 0:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp0[(None, None, None, st_idx)], tTR_rAcc)
                            else:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp1[(None, None, None, st_idx)], tTR_rAcc)
                            c_base = n0_col_base + cute.arch.make_warp_uniform(tTR_cAcc[(None, None, None, st_idx)][0][1])
                            cute.autovec_copy(cute.local_tile(mB0, (32,), (c_base // 32,)), r_b32)
                            for elem_i in cutlass.range(32, unroll_full=True):
                                x = tTR_rAcc[elem_i] + r_b32[elem_i]
                                hx = x * half_f32
                                tTR_rC[elem_i] = cutlass.BFloat16(hx * (one_f32 + _tanh_approx_f32(hx)))
                            if n0 == 0: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_0[(None, None, None, st_idx)])
                            elif n0 == 1: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_1[(None, None, None, st_idx)])
                            elif n0 == 2: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_2[(None, None, None, st_idx)])
                            else: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_3[(None, None, None, st_idx)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        acc_tmp_pipe.consumer_release(acc_tmp_cstate)
                        acc_tmp_cstate.advance()
                        h1_pipe.producer_commit(h1_pstate)
                        h1_pstate.advance()

                    h2_free_pipe.consumer_wait(h2_free_cstate)
                    h2_free_pipe.consumer_release(h2_free_cstate)
                    h2_free_cstate.advance()

                    for j in cutlass.range(4, unroll=1):
                        acc_tmp_pipe.consumer_wait(acc_tmp_cstate)
                        h2_pipe.producer_acquire(h2_pstate)
                        n1_col_base = j * 256
                        for st_idx in cutlass.range(subtile_cnt, unroll=1):
                            if acc_tmp_cstate.index == 0:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp0[(None, None, None, st_idx)], tTR_rAcc)
                            else:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp1[(None, None, None, st_idx)], tTR_rAcc)
                            c_base = n1_col_base + cute.arch.make_warp_uniform(tTR_cAcc[(None, None, None, st_idx)][0][1])
                            cute.autovec_copy(cute.local_tile(mB1, (32,), (c_base // 32,)), r_b32)
                            for elem_i in cutlass.range(32, unroll_full=True):
                                x = tTR_rAcc[elem_i] + r_b32[elem_i]
                                hx = x * half_f32
                                tTR_rC[elem_i] = cutlass.BFloat16(hx * (one_f32 + _tanh_approx_f32(hx)))
                            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH2[(None, None, None, st_idx)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        acc_tmp_pipe.consumer_release(acc_tmp_cstate)
                        acc_tmp_cstate.advance()
                        h2_pipe.producer_commit(h2_pstate)
                        h2_pstate.advance()

                    acc_out_pipe.consumer_wait(acc_out_cstate)
                    ss_half = cutlass.Float32(0.0)
                    for st_i in cutlass.range(4, unroll_full=True):
                        if acc_out_cstate.index == 0:
                            cute.copy(tiled_copy_t2r, tTR_tAccOut0[(None, None, None, st_i)], tTR_rAcc)
                        else:
                            cute.copy(tiled_copy_t2r, tTR_tAccOut1[(None, None, None, st_i)], tTR_rAcc)
                        c_st = c_s0 if st_i == 0 else (c_s1 if st_i == 1 else (c_s2 if st_i == 2 else c_s3))
                        cute.autovec_copy(cute.local_tile(mB2, (32,), (c_st // 32,)), r_b32)
                        for ei in cutlass.range(32, unroll_full=True):
                            v = tTR_rAcc[ei] + r_b32[ei]
                            if st_i == 0: r_y0[ei] = v
                            elif st_i == 1: r_y1[ei] = v
                            elif st_i == 2: r_y2[ei] = v
                            else: r_y3[ei] = v
                            ss_half = ss_half + v * v
                    acc_out_pipe.consumer_release(acc_out_cstate)
                    acc_out_cstate.advance()

                    sRmsHalf[(thr_row_local, warp_half)] = ss_half
                    epi_sync.arrive_and_wait()
                    inv_rms = cute.math.rsqrt(
                        (sRmsHalf[(thr_row_local, 0)] + sRmsHalf[(thr_row_local, 1)]) * inv_256 + eps,
                        fastmath=False,
                    )

                    for st_i in cutlass.range(4, unroll_full=True):
                        c_st = c_s0 if st_i == 0 else (c_s1 if st_i == 1 else (c_s2 if st_i == 2 else c_s3))
                        cute.autovec_copy(cute.local_tile(mGamma, (32,), (c_st // 32,)), r_b32)
                        for ei in cutlass.range(32, unroll_full=True):
                            yv = r_y0[ei] if st_i == 0 else (r_y1[ei] if st_i == 1 else (r_y2[ei] if st_i == 2 else r_y3[ei]))
                            tTR_rC[ei] = cutlass.BFloat16(yv * inv_rms * r_b32[ei])
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_0[(None, None, None, st_i)])

                    cute.arch.fence_proxy("async.shared", space="cta")
                    epi_sync.arrive_and_wait()
                    if warp_idx == 0:
                        for n_blk in cutlass.range(4, unroll_full=True):
                            cute.copy(tma_atom_y, bSG_sY[(None, n_blk)], bSG_gY_all[(None, cta_m_tile, n_blk, 0)])
                        y_tma_pipe.producer_commit()

                    cm = cm + num_launch_clusters
                    tile_i = tile_i + 1

                if warp_idx == 0:
                    y_tma_pipe.producer_tail()
                epi_sync.arrive_and_wait()

            cute.arch.barrier()
            tmem.free(tmem_ptr_base)
            return

    class FusedOrbEdgeLayer:
        """Persistent 2-CTA Blackwell tcgen05 fused Orb MP Edge Layer (4x-MLP Coalesced Gather + n0-outer + Dbl-Buf acc_out + Gates + Residual + REDG.F32x4)."""

        def __init__(self, num_sms: int = 152, num_b_stages: int = 4, use_distance_cutoff: bool = True):
            self.h_dim = 1024
            self.d_out = 256
            self.num_b_stages = num_b_stages
            self.num_xh2_stages = 4
            self.num_h1_stages = 16
            self.max_clusters = max(1, num_sms // 2)
            self.use_distance_cutoff = use_distance_cutoff

            self.ab_dtype = cutlass.BFloat16
            self.acc_dtype = cutlass.Float32
            self.c_dtype = cutlass.BFloat16

            self.use_2cta_instrs = True
            self.cta_group = tcgen05.CtaGroup.TWO
            self.cluster_shape_mn = (2, 1)
            self.mma_tiler = (128, 256, 64)
            self.cta_tile_shape_mnk = (64, 256, 64)
            self.epi_threads = 128
            self.threads_per_cta = 192
            self.a_major_mode = cute.nvgpu.OperandMajorMode.K
            self.b_major_mode = cute.nvgpu.OperandMajorMode.K
            self.c_layout = LayoutEnum.ROW_MAJOR

        @cute.jit
        def __call__(
            self,
            mE: cute.Tensor, mNodes: cute.Tensor,
            mSenders: cute.Tensor, mReceivers: cute.Tensor, mCutoff: cute.Tensor,
            mW0: cute.Tensor, mB0: cute.Tensor,
            mW1: cute.Tensor, mB1: cute.Tensor,
            mW2: cute.Tensor, mB2: cute.Tensor, mGamma: cute.Tensor,
            mWra: cute.Tensor, mWsa: cute.Tensor,
            b_ra: cutlass.Float32, b_sa: cutlass.Float32,
            sent_agg_ptr: cutlass.Int64, recv_agg_ptr: cutlass.Int64,
            eps: cutlass.Float32, E_orig: cutlass.Int32, stream: cuda_drv.CUstream,
        ):
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.ab_dtype, self.ab_dtype,
                self.a_major_mode, self.b_major_mode,
                self.acc_dtype, self.cta_group, self.mma_tiler[:2],
            )
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,),
            )

            xh2_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_xh2_stages)
            h1_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_h1_stages)
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_b_stages)

            a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
            x0_smem_layout = cute.select(xh2_smem_layout_staged, mode=[0, 1, 2])
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])

            tma_atom_e, tma_tensor_e = cute.nvgpu.make_tiled_tma_atom_A(a_op, mE, x0_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
            tma_atom_w0, tma_tensor_w0 = cute.nvgpu.make_tiled_tma_atom_B(b_op, mW0, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
            tma_atom_w1, tma_tensor_w1 = cute.nvgpu.make_tiled_tma_atom_B(b_op, mW1, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
            tma_atom_w2, tma_tensor_w2 = cute.nvgpu.make_tiled_tma_atom_B(b_op, mW2, b_smem_layout, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)

            tx_e_all_bytes = cute.size_in_bytes(self.ab_dtype, x0_smem_layout) * atom_thr_size * 4
            tx_b_bytes = cute.size_in_bytes(self.ab_dtype, b_smem_layout) * atom_thr_size

            epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
            cols_per_acc = memory.get_num_tmem_alloc_cols(tCtAcc_fake)
            num_tmem_alloc_cols = cols_per_acc * 4

            total_clusters_m = cute.ceil_div(mE.shape[0], 128)
            num_launch_clusters = cutlass.min(total_clusters_m, self.max_clusters)
            grid = (num_launch_clusters * 2, 1, 1)

            self.kernel(
                tiled_mma,
                tma_atom_e, tma_tensor_e, mE,
                mNodes, mSenders, mReceivers, mCutoff,
                tma_atom_w0, tma_tensor_w0, mB0,
                tma_atom_w1, tma_tensor_w1, mB1,
                tma_atom_w2, tma_tensor_w2, mB2,
                mGamma, mWra, mWsa, b_ra, b_sa,
                sent_agg_ptr, recv_agg_ptr,
                eps, E_orig, total_clusters_m, num_launch_clusters,
                cluster_layout_vmnk,
                xh2_smem_layout_staged, h1_smem_layout_staged, b_smem_layout_staged,
                epi_tile, tx_e_all_bytes, tx_b_bytes, cols_per_acc, num_tmem_alloc_cols,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_e: cute.CopyAtom, mE_tma: cute.Tensor, mE_gmem: cute.Tensor,
            mNodes: cute.Tensor,
            mSenders: cute.Tensor, mReceivers: cute.Tensor, mCutoff: cute.Tensor,
            tma_atom_w0: cute.CopyAtom, mW0_nkl: cute.Tensor, mB0: cute.Tensor,
            tma_atom_w1: cute.CopyAtom, mW1_nkl: cute.Tensor, mB1: cute.Tensor,
            tma_atom_w2: cute.CopyAtom, mW2_nkl: cute.Tensor, mB2: cute.Tensor,
            mGamma: cute.Tensor, mWra: cute.Tensor, mWsa: cute.Tensor,
            b_ra: cutlass.Float32, b_sa: cutlass.Float32,
            sent_agg_ptr: cutlass.Int64, recv_agg_ptr: cutlass.Int64,
            eps: cutlass.Float32, E_orig: cutlass.Int32,
            total_clusters_m: cutlass.Int32, num_launch_clusters: cutlass.Int32,
            cluster_layout_vmnk: cute.Layout,
            xh2_smem_layout_staged: cute.ComposedLayout,
            h1_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_tile: cute.Tile,
            tx_e_all_bytes: cutlass.Constexpr, tx_b_bytes: cutlass.Constexpr,
            cols_per_acc: cutlass.Constexpr, num_tmem_alloc_cols: cutlass.Constexpr,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 4:
                cpasync.prefetch_descriptor(tma_atom_e)
                cpasync.prefetch_descriptor(tma_atom_w0)
                cpasync.prefetch_descriptor(tma_atom_w1)
                cpasync.prefetch_descriptor(tma_atom_w2)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            mma_tile_coord_v = bidx % 2
            is_leader_cta = mma_tile_coord_v == 0
            cluster_id = bidx // 2

            @cute.struct
            class SharedStorage:
                e_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stages * 2]
                gather_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                acc_tmp_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
                acc_out_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
                h1_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 8]
                h2_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                x0_free_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                h2_free_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            smem = memory.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            e_prod, e_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.e_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=tx_e_all_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            ).make_participants()
            b_prod, b_cons = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_mbar_ptr.data_ptr(), num_stages=self.num_b_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=tx_b_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            ).make_participants()

            gather_pipe = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.gather_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            gather_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            gather_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            acc_tmp_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_tmp_mbar_ptr.data_ptr(), num_stages=2,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            acc_tmp_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_tmp_cmt = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_tmp_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)

            acc_out_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_out_mbar_ptr.data_ptr(), num_stages=2,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            acc_out_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            acc_out_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)

            h1_pipe = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.h1_mbar_ptr.data_ptr(), num_stages=4,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h1_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 4)
            h1_cwait_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 4)
            h1_crel_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 4)

            h2_pipe = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.h2_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h2_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            h2_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            x0_free_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.x0_free_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            x0_free_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            x0_free_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            h2_free_pipe = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.h2_free_mbar_ptr.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epi_threads * 2),
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            h2_free_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            h2_free_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.threads_per_cta)
            tmem = memory.TmemAllocator(
                storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
                allocator_warp_id=0, is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            )
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sH1 = smem.allocate_tensor(self.ab_dtype, h1_smem_layout_staged.outer, 128, h1_smem_layout_staged.inner)
            sXH2 = smem.allocate_tensor(self.ab_dtype, xh2_smem_layout_staged.outer, 128, xh2_smem_layout_staged.inner)
            sB = smem.allocate_tensor(self.ab_dtype, b_smem_layout_staged.outer, 128, b_smem_layout_staged.inner)

            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

            gE_mkl = cute.local_tile(mE_tma, (128, 64), (None, None, None))
            gW0_nkl = cute.local_tile(mW0_nkl, (256, 64), (None, None, None))
            gW1_nkl = cute.local_tile(mW1_nkl, (256, 64), (None, None, None))
            gW2_nkl = cute.local_tile(mW2_nkl, (256, 64), (None, None, None))

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            tCgE = thr_mma.partition_A(gE_mkl)
            tCgW0 = thr_mma.partition_B(gW0_nkl)
            tCgW1 = thr_mma.partition_B(gW1_nkl)
            tCgW2 = thr_mma.partition_B(gW2_nkl)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)

            tAsE, tAgE_all = cpasync.tma_partition(tma_atom_e, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sXH2, 0, 3), cute.group_modes(tCgE, 0, 3))
            tBsW, tBgW0_all = cpasync.tma_partition(tma_atom_w0, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW0, 0, 3))
            _, tBgW1_all = cpasync.tma_partition(tma_atom_w1, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW1, 0, 3))
            _, tBgW2_all = cpasync.tma_partition(tma_atom_w2, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgW2, 0, 3))

            tCrA_XH2 = tiled_mma.make_fragment_A(sXH2)
            tCrA_H1 = tiled_mma.make_fragment_A(sH1)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tmem.relinquish_alloc_permit()

            tCtAccTmp0 = cute.make_tensor(tmem_ptr_base, tCtAcc_fake.layout)
            tCtAccTmp1 = cute.make_tensor(tmem_ptr_base + cols_per_acc, tCtAcc_fake.layout)
            tCtAccOut0 = cute.make_tensor(tmem_ptr_base + cols_per_acc * 2, tCtAcc_fake.layout)
            tCtAccOut1 = cute.make_tensor(tmem_ptr_base + cols_per_acc * 3, tCtAcc_fake.layout)

            k1_tiles = 16
            num_kblks = cute.size(tCrA_XH2, mode=[2])
            tBgW2_0 = tBgW2_all[(None, 0, None, 0)]

            if warp_idx == 4:
                cm = cluster_id
                tile_i = cutlass.Int32(0)
                while cm < total_clusters_m:
                    if tile_i > 0:
                        x0_free_pipe.consumer_wait(x0_free_cstate)
                        x0_free_pipe.consumer_release(x0_free_cstate)
                        x0_free_cstate.advance()

                    tAgE = tAgE_all[(None, cm, None, 0)]
                    eh = e_prod.acquire_and_advance()
                    for k_e in cutlass.range(4, unroll_full=True):
                        cute.copy(tma_atom_e, tAgE[(None, k_e)], tAsE[(None, k_e)],
                                  tma_bar_ptr=eh.barrier, mcast_mask=a_full_mcast_mask)

                    for n0 in cutlass.range(4, unroll=1):
                        tBgW0 = tBgW0_all[(None, n0, None, 0)]
                        for k_idx in cutlass.range(12, unroll=1):
                            bh = b_prod.acquire_and_advance()
                            cute.copy(tma_atom_w0, tBgW0[(None, k_idx)], tBsW[(None, bh.index)],
                                      tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)

                    tBgW1_0 = tBgW1_all[(None, 0, None, 0)]
                    for k_idx in cutlass.range(k1_tiles, unroll=1):
                        bh = b_prod.acquire_and_advance()
                        cute.copy(tma_atom_w1, tBgW1_0[(None, k_idx)], tBsW[(None, bh.index)],
                                  tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    for j_next in cutlass.range(1, 4, unroll=1):
                        tBgW1_j = tBgW1_all[(None, j_next, None, 0)]
                        for k_idx in cutlass.range(k1_tiles, unroll=1):
                            bh = b_prod.acquire_and_advance()
                            cute.copy(tma_atom_w1, tBgW1_j[(None, k_idx)], tBsW[(None, bh.index)],
                                      tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                        j_prev = j_next - 1
                        for k_sub in cutlass.range(4, unroll=1):
                            bh = b_prod.acquire_and_advance()
                            cute.copy(tma_atom_w2, tBgW2_0[(None, j_prev * 4 + k_sub)], tBsW[(None, bh.index)],
                                      tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    for k_sub in cutlass.range(4, unroll=1):
                        bh = b_prod.acquire_and_advance()
                        cute.copy(tma_atom_w2, tBgW2_0[(None, 12 + k_sub)], tBsW[(None, bh.index)],
                                  tma_bar_ptr=bh.barrier, mcast_mask=b_full_mcast_mask)
                    cm = cm + num_launch_clusters
                    tile_i = tile_i + 1
                e_prod.tail()
                b_prod.tail()

            if warp_idx == 5:
                if is_leader_cta:
                    cm = cluster_id
                    while cm < total_clusters_m:
                        eh = e_cons.wait_and_advance()

                        for n0 in cutlass.range(4, unroll=1):
                            acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                            tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                            acc_tmp_acq.advance()
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for k_sub in cutlass.range(4, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_XH2[(None, None, kblk, k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                            if n0 == 0:
                                gather_pipe.consumer_wait(gather_cstate)
                            for k_sub in cutlass.range(8, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_H1[(None, None, kblk, 8 + k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                bh.release()
                            if n0 < 2:
                                acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                                acc_tmp_cmt.advance()

                        eh.release()
                        gather_pipe.consumer_release(gather_cstate)
                        gather_cstate.advance()

                        acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                        acc_tmp_cmt.advance()
                        acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                        acc_tmp_cmt.advance()

                        h2_free_pipe.producer_acquire(h2_free_pstate)
                        h2_free_pipe.producer_commit(h2_free_pstate)
                        h2_free_pstate.advance()

                        acc_out_pipe.producer_acquire(acc_out_pstate)
                        tCtAcc_out = cute.make_tensor(tmem_ptr_base + cols_per_acc * (2 + acc_out_pstate.index), tCtAcc_fake.layout)

                        acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                        tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                        acc_tmp_acq.advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for sl in cutlass.range(4, unroll=1):
                            h1_pipe.consumer_wait(h1_cwait_state)
                            h1_cwait_state.advance()
                            for k_sub in cutlass.range(4, unroll=1):
                                bh = b_cons.wait_and_advance()
                                k_h1 = sl * 4 + k_sub
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_H1[(None, None, kblk, k_h1)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                        acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                        acc_tmp_cmt.advance()

                        for j_next in cutlass.range(1, 4, unroll=1):
                            acc_tmp_pipe.producer_acquire(acc_tmp_acq)
                            tCtAcc_cur = cute.make_tensor(tmem_ptr_base + cols_per_acc * acc_tmp_acq.index, tCtAcc_fake.layout)
                            acc_tmp_acq.advance()
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for k_idx in cutlass.range(k1_tiles, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_cur, tCrA_H1[(None, None, kblk, k_idx)], tCrB[(None, None, kblk, bh.index)], tCtAcc_cur)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                            acc_tmp_pipe.producer_commit(acc_tmp_cmt)
                            acc_tmp_cmt.advance()

                            if j_next == 3:
                                for _sl in cutlass.range(4, unroll_full=True):
                                    h1_pipe.consumer_release(h1_crel_state)
                                    h1_crel_state.advance()

                            h2_pipe.consumer_wait(h2_cstate)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, j_next > 1)
                            for k_sub in cutlass.range(4, unroll=1):
                                bh = b_cons.wait_and_advance()
                                for kblk in cutlass.range(num_kblks, unroll_full=True):
                                    cute.gemm(tiled_mma, tCtAcc_out, tCrA_XH2[(None, None, kblk, k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_out)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                bh.release()
                            h2_pipe.consumer_release(h2_cstate)
                            h2_cstate.advance()

                        h2_pipe.consumer_wait(h2_cstate)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        for k_sub in cutlass.range(4, unroll=1):
                            bh = b_cons.wait_and_advance()
                            for kblk in cutlass.range(num_kblks, unroll_full=True):
                                cute.gemm(tiled_mma, tCtAcc_out, tCrA_XH2[(None, None, kblk, k_sub)], tCrB[(None, None, kblk, bh.index)], tCtAcc_out)
                            bh.release()
                        h2_pipe.consumer_release(h2_cstate)
                        h2_cstate.advance()

                        acc_out_pipe.producer_commit(acc_out_pstate)
                        acc_out_pstate.advance()

                        if cm + num_launch_clusters < total_clusters_m:
                            x0_free_pipe.producer_acquire(x0_free_pstate)
                            x0_free_pipe.producer_commit(x0_free_pstate)
                            x0_free_pstate.advance()

                        cm = cm + num_launch_clusters

                    acc_tmp_pipe.producer_tail(acc_tmp_acq)
                    acc_out_pipe.producer_tail(acc_out_pstate)
                    h2_free_pipe.producer_tail(h2_free_pstate)
                    x0_free_pipe.producer_tail(x0_free_pstate)

            if warp_idx < 4:
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, True
                )
                tAccTmp0_epi = cute.flat_divide(tCtAccTmp0[((None, None), 0, 0)], epi_tile)
                tAccTmp1_epi = cute.flat_divide(tCtAccTmp1[((None, None), 0, 0)], epi_tile)
                tAccOut0_epi = cute.flat_divide(tCtAccOut0[((None, None), 0, 0)], epi_tile)
                tAccOut1_epi = cute.flat_divide(tCtAccOut1[((None, None), 0, 0)], epi_tile)

                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAccTmp0_epi[(None, None, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAccTmp0 = cute.group_modes(thr_copy_t2r.partition_S(tAccTmp0_epi), 3, 5)
                tTR_tAccTmp1 = cute.group_modes(thr_copy_t2r.partition_S(tAccTmp1_epi), 3, 5)
                tTR_tAccOut0 = cute.group_modes(thr_copy_t2r.partition_S(tAccOut0_epi), 3, 5)
                tTR_tAccOut1 = cute.group_modes(thr_copy_t2r.partition_S(tAccOut1_epi), 3, 5)

                cAcc = cute.make_identity_tensor((64, 256))
                tTR_cAcc = cute.group_modes(thr_copy_t2r.partition_D(cute.flat_divide(cAcc, epi_tile)), 3, 5)
                tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0)].shape, self.acc_dtype)
                tTR_rC = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0)].shape, self.c_dtype)
                subtile_cnt = cute.size(tTR_tAccTmp0.shape, mode=[3])

                copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_rC = tiled_copy_r2s.retile(tTR_rC)

                k_sw128_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, self.ab_dtype)
                sH1_4slices_layout = cute.tile_to_shape(k_sw128_atom, (64, 256, 4), order=(0, 1, 2))
                sH1_4slices = cute.make_tensor(
                    cute.recast_ptr(sH1.iterator, sH1_4slices_layout.inner, dtype=self.ab_dtype), sH1_4slices_layout.outer)
                sH2_2d_layout = cute.tile_to_shape(k_sw128_atom, (64, 256), order=(0, 1))
                sH2_2d = cute.make_tensor(
                    cute.recast_ptr(sXH2.iterator, sH2_2d_layout.inner, dtype=self.ab_dtype), sH2_2d_layout.outer)

                tRS_sH2 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH2_2d, epi_tile)), 3, 5)
                tRS_sH1_0 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 0)], epi_tile)), 3, 5)
                tRS_sH1_1 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 1)], epi_tile)), 3, 5)
                tRS_sH1_2 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 2)], epi_tile)), 3, 5)
                tRS_sH1_3 = cute.group_modes(thr_copy_r2s.partition_D(cute.flat_divide(sH1_4slices[(None, None, 3)], epi_tile)), 3, 5)

                sY_f32 = cute.make_tensor(
                    cute.recast_ptr(sH1.iterator, dtype=cutlass.Float32),
                    cute.make_layout((64, 256), stride=(260, 1)),
                )

                epi_sync = pipeline.NamedBarrier(barrier_id=3, num_threads=self.epi_threads)
                half_f32 = cutlass.Float32(0.5)
                one_f32 = cutlass.Float32(1.0)
                inv_256 = cutlass.Float32(1.0 / 256.0)
                lane = tidx % 32
                col_base_8 = lane * 8

                r_b2 = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_gamma = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_wra = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_wsa = cute.make_rmem_tensor((8,), cutlass.Float32)
                cute.autovec_copy(cute.local_tile(mB2, (8,), (lane,)), r_b2)
                cute.autovec_copy(cute.local_tile(mGamma, (8,), (lane,)), r_gamma)
                cute.autovec_copy(cute.local_tile(mWra, (8,), (lane,)), r_wra)
                cute.autovec_copy(cute.local_tile(mWsa, (8,), (lane,)), r_wsa)

                r_nr0 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_nr1 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_nr2 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_nr3 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_ns = cute.make_rmem_tensor((8,), cutlass.BFloat16)

                r_e_old = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_e_new = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                r_y8 = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_x = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_upd = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_sacc = cute.make_rmem_tensor((8,), cutlass.Float32)
                r_b32 = cute.make_rmem_tensor((32,), cutlass.Float32)

                thr_row_local = tTR_cAcc[(0, 0, 0, 0)][0]
                col_byte_off_0 = cutlass.Int64(col_base_8 * 4)
                col_byte_off_4 = cutlass.Int64((col_base_8 + 4) * 4)
                row_stride_bytes = cutlass.Int64(256 * 4)

                sNs_2d = sH1_4slices[(None, None, 2)]
                sNr_2d = sH1_4slices[(None, None, 3)]

                cm = cluster_id
                while cm < total_clusters_m:
                    cta_row_base = (cm * 2 + mma_tile_coord_v) * 64

                    gather_pipe.producer_acquire(gather_pstate)
                    prev_snd_g = cutlass.Int64(-1)
                    for r_grp in cutlass.range(4, unroll=1):
                        r0 = warp_idx * 16 + r_grp * 4
                        g0 = cutlass.min(cta_row_base + r0 + 0, E_orig - 1)
                        g1 = cutlass.min(cta_row_base + r0 + 1, E_orig - 1)
                        g2 = cutlass.min(cta_row_base + r0 + 2, E_orig - 1)
                        g3 = cutlass.min(cta_row_base + r0 + 3, E_orig - 1)
                        rcv0 = mReceivers[g0]; rcv1 = mReceivers[g1]; rcv2 = mReceivers[g2]; rcv3 = mReceivers[g3]
                        cute.autovec_copy(cute.local_tile(mNodes[(rcv0, None)], (8,), (lane,)), r_nr0)
                        cute.autovec_copy(cute.local_tile(mNodes[(rcv1, None)], (8,), (lane,)), r_nr1)
                        cute.autovec_copy(cute.local_tile(mNodes[(rcv2, None)], (8,), (lane,)), r_nr2)
                        cute.autovec_copy(cute.local_tile(mNodes[(rcv3, None)], (8,), (lane,)), r_nr3)
                        snd0 = mSenders[g0]
                        if snd0 != prev_snd_g:
                            cute.autovec_copy(cute.local_tile(mNodes[(snd0, None)], (8,), (lane,)), r_ns)
                            prev_snd_g = snd0
                        cute.autovec_copy(r_ns, cute.local_tile(sNs_2d[(r0 + 0, None)], (8,), (lane,)))
                        snd1 = mSenders[g1]
                        if snd1 != prev_snd_g:
                            cute.autovec_copy(cute.local_tile(mNodes[(snd1, None)], (8,), (lane,)), r_ns)
                            prev_snd_g = snd1
                        cute.autovec_copy(r_ns, cute.local_tile(sNs_2d[(r0 + 1, None)], (8,), (lane,)))
                        snd2 = mSenders[g2]
                        if snd2 != prev_snd_g:
                            cute.autovec_copy(cute.local_tile(mNodes[(snd2, None)], (8,), (lane,)), r_ns)
                            prev_snd_g = snd2
                        cute.autovec_copy(r_ns, cute.local_tile(sNs_2d[(r0 + 2, None)], (8,), (lane,)))
                        snd3 = mSenders[g3]
                        if snd3 != prev_snd_g:
                            cute.autovec_copy(cute.local_tile(mNodes[(snd3, None)], (8,), (lane,)), r_ns)
                            prev_snd_g = snd3
                        cute.autovec_copy(r_ns, cute.local_tile(sNs_2d[(r0 + 3, None)], (8,), (lane,)))
                        cute.autovec_copy(r_nr0, cute.local_tile(sNr_2d[(r0 + 0, None)], (8,), (lane,)))
                        cute.autovec_copy(r_nr1, cute.local_tile(sNr_2d[(r0 + 1, None)], (8,), (lane,)))
                        cute.autovec_copy(r_nr2, cute.local_tile(sNr_2d[(r0 + 2, None)], (8,), (lane,)))
                        cute.autovec_copy(r_nr3, cute.local_tile(sNr_2d[(r0 + 3, None)], (8,), (lane,)))
                    cute.arch.fence_proxy("async.shared", space="cta")
                    gather_pipe.producer_commit(gather_pstate)
                    gather_pstate.advance()

                    for n0 in cutlass.range(4, unroll=1):
                        h1_pipe.producer_acquire(h1_pstate)
                        acc_tmp_pipe.consumer_wait(acc_tmp_cstate)
                        n0_col_base = n0 * 256
                        for st_idx in cutlass.range(subtile_cnt, unroll=1):
                            if acc_tmp_cstate.index == 0:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp0[(None, None, None, st_idx)], tTR_rAcc)
                            else:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp1[(None, None, None, st_idx)], tTR_rAcc)
                            c_start = n0_col_base + cute.arch.make_warp_uniform(tTR_cAcc[(None, None, None, st_idx)][0][1])
                            cute.autovec_copy(cute.local_tile(mB0, (32,), (c_start // 32,)), r_b32)
                            for elem_i in cutlass.range(32, unroll_full=True):
                                x = tTR_rAcc[elem_i] + r_b32[elem_i]
                                hx = x * half_f32
                                tTR_rC[elem_i] = cutlass.BFloat16(hx * (one_f32 + _tanh_approx_f32(hx)))
                            if n0 == 0: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_0[(None, None, None, st_idx)])
                            elif n0 == 1: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_1[(None, None, None, st_idx)])
                            elif n0 == 2: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_2[(None, None, None, st_idx)])
                            else: cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH1_3[(None, None, None, st_idx)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        acc_tmp_pipe.consumer_release(acc_tmp_cstate)
                        acc_tmp_cstate.advance()
                        h1_pipe.producer_commit(h1_pstate)
                        h1_pstate.advance()

                    h2_free_pipe.consumer_wait(h2_free_cstate)
                    h2_free_pipe.consumer_release(h2_free_cstate)
                    h2_free_cstate.advance()

                    for j in cutlass.range(4, unroll=1):
                        acc_tmp_pipe.consumer_wait(acc_tmp_cstate)
                        h2_pipe.producer_acquire(h2_pstate)
                        n1_col_base = j * 256
                        for st_idx in cutlass.range(subtile_cnt, unroll=1):
                            if acc_tmp_cstate.index == 0:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp0[(None, None, None, st_idx)], tTR_rAcc)
                            else:
                                cute.copy(tiled_copy_t2r, tTR_tAccTmp1[(None, None, None, st_idx)], tTR_rAcc)
                            c_start = n1_col_base + cute.arch.make_warp_uniform(tTR_cAcc[(None, None, None, st_idx)][0][1])
                            cute.autovec_copy(cute.local_tile(mB1, (32,), (c_start // 32,)), r_b32)
                            for elem_i in cutlass.range(32, unroll_full=True):
                                x = tTR_rAcc[elem_i] + r_b32[elem_i]
                                hx = x * half_f32
                                tTR_rC[elem_i] = cutlass.BFloat16(hx * (one_f32 + _tanh_approx_f32(hx)))
                            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sH2[(None, None, None, st_idx)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        acc_tmp_pipe.consumer_release(acc_tmp_cstate)
                        acc_tmp_cstate.advance()
                        h2_pipe.producer_commit(h2_pstate)
                        h2_pstate.advance()

                    acc_out_pipe.consumer_wait(acc_out_cstate)
                    for st_idx in cutlass.range(subtile_cnt, unroll=1):
                        if acc_out_cstate.index == 0:
                            cute.copy(tiled_copy_t2r, tTR_tAccOut0[(None, None, None, st_idx)], tTR_rAcc)
                        else:
                            cute.copy(tiled_copy_t2r, tTR_tAccOut1[(None, None, None, st_idx)], tTR_rAcc)
                        c_start = cute.arch.make_warp_uniform(tTR_cAcc[(None, None, None, st_idx)][0][1])
                        cute.autovec_copy(tTR_rAcc, cute.local_tile(sY_f32[(thr_row_local, None)], (32,), (c_start // 32,)))
                    acc_out_pipe.consumer_release(acc_out_cstate)
                    acc_out_cstate.advance()
                    epi_sync.arrive_and_wait()

                    cur_snd = cutlass.Int64(-1)
                    for v_i in cutlass.range(8, unroll_full=True):
                        r_sacc[v_i] = cutlass.Float32(0.0)

                    for r_step in cutlass.range(16, unroll=1):
                        r_loc = warp_idx * 16 + r_step
                        row_global = cta_row_base + r_loc
                        if row_global < E_orig:
                            cute.autovec_copy(cute.local_tile(mE_gmem[(row_global, None, 0)], (8,), (lane,)), r_e_old)
                            cute.autovec_copy(cute.local_tile(sY_f32[(r_loc, None)], (8,), (lane,)), r_y8)

                            ss = cutlass.Float32(0.0)
                            dot_r = cutlass.Float32(0.0)
                            dot_s = cutlass.Float32(0.0)
                            for v_i in cutlass.range(8, unroll_full=True):
                                xv = r_y8[v_i] + r_b2[v_i]
                                r_x[v_i] = xv
                                ss = ss + xv * xv
                                ev = cutlass.Float32(r_e_old[v_i])
                                dot_r = dot_r + ev * r_wra[v_i]
                                dot_s = dot_s + ev * r_wsa[v_i]

                            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=16)
                            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=8)
                            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=4)
                            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=2)
                            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=1)

                            dot_r = dot_r + cute.arch.shuffle_sync_bfly(dot_r, offset=16)
                            dot_r = dot_r + cute.arch.shuffle_sync_bfly(dot_r, offset=8)
                            dot_r = dot_r + cute.arch.shuffle_sync_bfly(dot_r, offset=4)
                            dot_r = dot_r + cute.arch.shuffle_sync_bfly(dot_r, offset=2)
                            dot_r = dot_r + cute.arch.shuffle_sync_bfly(dot_r, offset=1)

                            dot_s = dot_s + cute.arch.shuffle_sync_bfly(dot_s, offset=16)
                            dot_s = dot_s + cute.arch.shuffle_sync_bfly(dot_s, offset=8)
                            dot_s = dot_s + cute.arch.shuffle_sync_bfly(dot_s, offset=4)
                            dot_s = dot_s + cute.arch.shuffle_sync_bfly(dot_s, offset=2)
                            dot_s = dot_s + cute.arch.shuffle_sync_bfly(dot_s, offset=1)

                            inv_rms = cute.math.rsqrt(ss * inv_256 + eps, fastmath=False)
                            cut_v = mCutoff[row_global] if cutlass.const_expr(self.use_distance_cutoff) else one_f32
                            r_attn = half_f32 * (one_f32 + _tanh_approx_f32((dot_r + b_ra) * half_f32)) * cut_v
                            s_attn = half_f32 * (one_f32 + _tanh_approx_f32((dot_s + b_sa) * half_f32)) * cut_v

                            for v_i in cutlass.range(8, unroll_full=True):
                                u = r_x[v_i] * inv_rms * r_gamma[v_i]
                                r_upd[v_i] = u
                                r_e_new[v_i] = cutlass.BFloat16(cutlass.Float32(r_e_old[v_i]) + u)
                            cute.autovec_copy(r_e_new, cute.local_tile(mE_gmem[(row_global, None, 0)], (8,), (lane,)))

                            rcv = mReceivers[row_global]
                            rcv_row_addr = recv_agg_ptr + rcv * row_stride_bytes
                            _red_global_add_f32x4(
                                rcv_row_addr + col_byte_off_0,
                                r_upd[0] * r_attn, r_upd[1] * r_attn, r_upd[2] * r_attn, r_upd[3] * r_attn,
                            )
                            _red_global_add_f32x4(
                                rcv_row_addr + col_byte_off_4,
                                r_upd[4] * r_attn, r_upd[5] * r_attn, r_upd[6] * r_attn, r_upd[7] * r_attn,
                            )

                            snd = mSenders[row_global]
                            if cur_snd >= 0 and snd != cur_snd:
                                snd_row_addr = sent_agg_ptr + cur_snd * row_stride_bytes
                                _red_global_add_f32x4(snd_row_addr + col_byte_off_0, r_sacc[0], r_sacc[1], r_sacc[2], r_sacc[3])
                                _red_global_add_f32x4(snd_row_addr + col_byte_off_4, r_sacc[4], r_sacc[5], r_sacc[6], r_sacc[7])
                                for v_i in cutlass.range(8, unroll_full=True):
                                    r_sacc[v_i] = r_upd[v_i] * s_attn
                                cur_snd = snd
                            else:
                                for v_i in cutlass.range(8, unroll_full=True):
                                    r_sacc[v_i] = r_sacc[v_i] + r_upd[v_i] * s_attn
                                cur_snd = snd

                    if cur_snd >= 0:
                        snd_row_addr = sent_agg_ptr + cur_snd * row_stride_bytes
                        _red_global_add_f32x4(snd_row_addr + col_byte_off_0, r_sacc[0], r_sacc[1], r_sacc[2], r_sacc[3])
                        _red_global_add_f32x4(snd_row_addr + col_byte_off_4, r_sacc[4], r_sacc[5], r_sacc[6], r_sacc[7])

                    epi_sync.arrive_and_wait()
                    cm = cm + num_launch_clusters

            cute.arch.barrier()
            tmem.free(tmem_ptr_base)
            return


# =============================================================================
# Weight Packing, Kernel Caching & High-Level Launch Wrappers
# =============================================================================

_MLP_KERNEL_CACHE: dict[tuple[int, int], Any] = {}
_EDGE_KERNEL_CACHE: dict[tuple[bool, int], Any] = {}


@dataclass
class PreparedMlp3Weights:
    d_in: int
    w0: torch.Tensor      # [1024, d_in] bf16 contiguous
    b0: torch.Tensor      # [1024] fp32 contiguous
    w1: torch.Tensor      # [1024, 1024] bf16 contiguous
    b1: torch.Tensor      # [1024] fp32 contiguous
    w2: torch.Tensor      # [256, 1024] bf16 contiguous
    b2: torch.Tensor      # [256] fp32 contiguous
    gamma: torch.Tensor   # [256] fp32 contiguous
    eps: float


@dataclass
class PreparedEdgeLayerWeights:
    w0: torch.Tensor      # [1024, 768] bf16 contiguous
    b0: torch.Tensor      # [1024] fp32 contiguous
    w1: torch.Tensor      # [1024, 1024] bf16 contiguous
    b1: torch.Tensor      # [1024] fp32 contiguous
    w2: torch.Tensor      # [256, 1024] bf16 contiguous
    b2: torch.Tensor      # [256] fp32 contiguous
    gamma: torch.Tensor   # [256] fp32 contiguous
    w_ra: torch.Tensor    # [256] fp32 contiguous
    w_sa: torch.Tensor    # [256] fp32 contiguous
    b_ra: float
    b_sa: float
    eps: float
    use_distance_cutoff: bool


@dataclass
class PreparedOrbFusedState:
    node_enc: PreparedMlp3Weights
    edge_enc: PreparedMlp3Weights
    edge_layers: list[PreparedEdgeLayerWeights]
    node_layers: list[PreparedMlp3Weights]
    bessel_weights: torch.Tensor
    bessel_prefactor: float
    has_zbl: bool
    zbl_cov_radii: torch.Tensor | None
    zbl_a_prefactor: float
    zbl_a_exp: float
    zbl_p: float
    zbl_c: tuple[float, float, float, float]
    zbl_d: tuple[float, float, float, float]
    num_sms: int


def is_compatible_3layer_mlp(module: nn.Module, expected_d_in: int | None = None) -> bool:
    """Validate module matches Orb's Sequential(mlp=3xLinear+2xSiLU+Identity, layer_norm=RMSNorm(256))."""
    if not (hasattr(module, "mlp") and hasattr(module, "layer_norm")):
        return False
    mlp = module.mlp
    ln = module.layer_norm
    if not isinstance(ln, nn.RMSNorm) or tuple(ln.normalized_shape) != (256,) or ln.weight is None:
        return False
    children = list(mlp.children())
    if len(children) != 6:
        return False
    l0, a0, l1, a1, l2, a2 = children
    if not (
        isinstance(l0, nn.Linear)
        and isinstance(a0, nn.SiLU)
        and isinstance(l1, nn.Linear)
        and isinstance(a1, nn.SiLU)
        and isinstance(l2, nn.Linear)
        and isinstance(a2, nn.Identity)
    ):
        return False
    if l0.bias is None or l1.bias is None or l2.bias is None:
        return False
    if l0.out_features != 1024 or l1.in_features != 1024 or l1.out_features != 1024:
        return False
    if l2.in_features != 1024 or l2.out_features != 256:
        return False
    if l0.in_features % 64 != 0:
        return False
    if expected_d_in is not None and l0.in_features != expected_d_in:
        return False
    return True


def _extract_mlp3_weights(module: nn.Module) -> PreparedMlp3Weights:
    children = list(module.mlp.children())
    l0, _, l1, _, l2, _ = children
    ln = module.layer_norm
    eps = float(ln.eps) if ln.eps is not None else float(torch.finfo(torch.float32).eps)
    return PreparedMlp3Weights(
        d_in=l0.in_features,
        w0=l0.weight.detach().to(torch.bfloat16).contiguous(),
        b0=l0.bias.detach().to(torch.float32).contiguous(),
        w1=l1.weight.detach().to(torch.bfloat16).contiguous(),
        b1=l1.bias.detach().to(torch.float32).contiguous(),
        w2=l2.weight.detach().to(torch.bfloat16).contiguous(),
        b2=l2.bias.detach().to(torch.float32).contiguous(),
        gamma=ln.weight.detach().to(torch.float32).contiguous(),
        eps=eps,
    )


def _extract_edge_layer_weights(gnn: nn.Module) -> PreparedEdgeLayerWeights:
    mlp = _extract_mlp3_weights(gnn._edge_mlp)
    assert mlp.d_in == 768
    ra = gnn._receive_attn
    sa = gnn._send_attn
    w_ra = ra.weight.detach().reshape(-1).to(torch.float32).contiguous()
    w_sa = sa.weight.detach().reshape(-1).to(torch.float32).contiguous()
    b_ra = float(ra.bias.detach().item()) if ra.bias is not None else 0.0
    b_sa = float(sa.bias.detach().item()) if sa.bias is not None else 0.0
    return PreparedEdgeLayerWeights(
        w0=mlp.w0,
        b0=mlp.b0,
        w1=mlp.w1,
        b1=mlp.b1,
        w2=mlp.w2,
        b2=mlp.b2,
        gamma=mlp.gamma,
        w_ra=w_ra,
        w_sa=w_sa,
        b_ra=b_ra,
        b_sa=b_sa,
        eps=mlp.eps,
        use_distance_cutoff=bool(gnn._distance_cutoff),
    )


def prepare_orb_fused_state(orbff: nn.Module, device: torch.device) -> PreparedOrbFusedState:
    """Pre-extract and pack all BF16/FP32 weights once per ORBModel instance."""
    gns = orbff.model
    has_zbl = bool(getattr(orbff, "pair_repulsion", False))
    if has_zbl:
        zbl = orbff.pair_repulsion_fn
        c_list = [float(x) for x in zbl.c.detach().reshape(-1).tolist()]
        d_list = [float(x) for x in zbl.d.detach().reshape(-1).tolist()]
        zbl_cov_radii = zbl.covalent_radii.detach().to(device=device, dtype=torch.float32).contiguous()
        zbl_a_pref = float(zbl.a_prefactor.detach().item())
        zbl_a_exp = float(zbl.a_exp.detach().item())
        zbl_p = float(zbl.p.detach().item())
        zbl_c = (c_list[0], c_list[1], c_list[2], c_list[3])
        zbl_d = (d_list[0], d_list[1], d_list[2], d_list[3])
    else:
        zbl_cov_radii = None
        zbl_a_pref, zbl_a_exp, zbl_p = 1.0, 1.0, 6.0
        zbl_c = (0.0, 0.0, 0.0, 0.0)
        zbl_d = (0.0, 0.0, 0.0, 0.0)

    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return PreparedOrbFusedState(
        node_enc=_extract_mlp3_weights(gns._encoder._node_fn),
        edge_enc=_extract_mlp3_weights(gns._encoder._edge_fn),
        edge_layers=[_extract_edge_layer_weights(gnn) for gnn in gns.gnn_stacks],
        node_layers=[_extract_mlp3_weights(gnn._node_mlp) for gnn in gns.gnn_stacks],
        bessel_weights=gns.rbf_transform.bessel_weights.detach().to(device=device, dtype=torch.float32).contiguous(),
        bessel_prefactor=float(gns.rbf_transform.prefactor.detach().item()),
        has_zbl=has_zbl,
        zbl_cov_radii=zbl_cov_radii,
        zbl_a_prefactor=zbl_a_pref,
        zbl_a_exp=zbl_a_exp,
        zbl_p=zbl_p,
        zbl_c=zbl_c,
        zbl_d=zbl_d,
        num_sms=num_sms,
    )


def _to_cute_3d(t: torch.Tensor):
    return from_dlpack(t.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _to_cute_2d(t: torch.Tensor):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _to_cute_1d_dyn(t: torch.Tensor):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _to_cute_1d_static(t: torch.Tensor):
    return from_dlpack(t, assumed_align=16)


def _pad_rows_128(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad 2D tensor rows to a multiple of 128 if needed (zero-copy when already multiple of 128)."""
    m = x.shape[0]
    m_pad = ((m + 127) // 128) * 128
    if m_pad == m and x.is_contiguous():
        return x, m
    out = torch.zeros((m_pad, x.shape[1]), device=x.device, dtype=x.dtype)
    out[:m].copy_(x)
    return out, m


def run_fused_mlp_3layer(
    x_bf16: torch.Tensor,
    weights: PreparedMlp3Weights,
    num_sms: int,
    out_bf16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run CuTeDSL fused 3-layer MLP + RMSNorm on [M, D_in] BF16 -> [M, 256] BF16.

    If `out_bf16` is provided, it must be contiguous BF16 with shape `(M_pad, 256)`
    where `M_pad = ((M + 127) // 128) * 128`, and results are written directly into it.
    """
    m_orig = x_bf16.shape[0]
    if m_orig == 0:
        return torch.empty((0, 256), device=x_bf16.device, dtype=torch.bfloat16)

    x_pad, _ = _pad_rows_128(x_bf16)
    m_pad = x_pad.shape[0]
    if out_bf16 is None:
        y_pad = torch.empty((m_pad, 256), device=x_bf16.device, dtype=torch.bfloat16)
    else:
        y_pad = out_bf16

    key = (weights.d_in, num_sms)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(x_bf16.device).cuda_stream)
    args = (
        _to_cute_3d(x_pad),
        _to_cute_3d(weights.w0), _to_cute_1d_static(weights.b0),
        _to_cute_3d(weights.w1), _to_cute_1d_static(weights.b1),
        _to_cute_3d(weights.w2), _to_cute_1d_static(weights.b2),
        _to_cute_1d_static(weights.gamma),
        _to_cute_3d(y_pad),
        cutlass.Float32(weights.eps),
        cutlass.Int32(m_orig),
        stream,
    )
    comp = _MLP_KERNEL_CACHE.get(key)
    if comp is None:
        k = FusedOrbMlp3Layer(d_in=weights.d_in, num_sms=num_sms, num_b_stages=4)
        comp = cute.compile(k, *args, options="--opt-level 3")
        _MLP_KERNEL_CACHE[key] = comp
    comp(*args)
    return y_pad[:m_orig]


def run_fused_edge_layer(
    edges_pad: torch.Tensor,          # [E_pad, 256] BF16, updated IN-PLACE!
    nodes_f32: torch.Tensor,          # [N, 256] FP32
    senders_pad: torch.Tensor,        # [E_pad] Int64
    receivers_pad: torch.Tensor,      # [E_pad] Int64
    cutoff_pad: torch.Tensor,         # [E_pad] FP32
    sent_agg: torch.Tensor,           # [N, 256] FP32 (zeroed by caller)
    recv_agg: torch.Tensor,           # [N, 256] FP32 (zeroed by caller)
    weights: PreparedEdgeLayerWeights,
    e_orig: int,
    num_sms: int,
) -> None:
    """Run CuTeDSL fused Stage 3 Orb MP Edge Layer in-place on `edges_pad`."""
    if e_orig == 0:
        return

    nodes_bf16 = nodes_f32.to(torch.bfloat16).contiguous()

    key = (weights.use_distance_cutoff, num_sms)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(edges_pad.device).cuda_stream)
    args = (
        _to_cute_3d(edges_pad),
        _to_cute_2d(nodes_bf16),
        _to_cute_1d_dyn(senders_pad),
        _to_cute_1d_dyn(receivers_pad),
        _to_cute_1d_dyn(cutoff_pad),
        _to_cute_3d(weights.w0), _to_cute_1d_static(weights.b0),
        _to_cute_3d(weights.w1), _to_cute_1d_static(weights.b1),
        _to_cute_3d(weights.w2), _to_cute_1d_static(weights.b2),
        _to_cute_1d_static(weights.gamma),
        _to_cute_1d_static(weights.w_ra),
        _to_cute_1d_static(weights.w_sa),
        cutlass.Float32(weights.b_ra),
        cutlass.Float32(weights.b_sa),
        cutlass.Int64(sent_agg.data_ptr()),
        cutlass.Int64(recv_agg.data_ptr()),
        cutlass.Float32(weights.eps),
        cutlass.Int32(e_orig),
        stream,
    )
    comp = _EDGE_KERNEL_CACHE.get(key)
    if comp is None:
        k = FusedOrbEdgeLayer(
            num_sms=num_sms,
            num_b_stages=4,
            use_distance_cutoff=weights.use_distance_cutoff,
        )
        comp = cute.compile(k, *args, options="--opt-level 3")
        _EDGE_KERNEL_CACHE[key] = comp
    comp(*args)


def run_fused_geom_zbl_featurize(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    senders_c: torch.Tensor,
    receivers_c: torch.Tensor,
    offsets_c: torch.Tensor,
    cell_2d: torch.Tensor,
    n_local: int,
    model_radius: float,
    state: PreparedOrbFusedState,
    cutoff_out: torch.Tensor,
    ef_out: torch.Tensor,
    zbl_energy_local: torch.Tensor,
    zbl_forces_local: torch.Tensor | None,
) -> None:
    """Launch fused Stage 4 Triton kernel on one edge chunk `[0:E_c]`."""
    e_c = senders_c.shape[0]
    if e_c == 0:
        return
    block_e = 128
    num_blocks = triton.cdiv(e_c, block_e)
    device = positions.device

    if state.has_zbl:
        zbl_e_blk = torch.empty((num_blocks,), device=device, dtype=torch.float64)
        zbl_f_arg = zbl_forces_local
        cov_radii = state.zbl_cov_radii
    else:
        zbl_e_blk = torch.empty((1,), device=device, dtype=torch.float64)
        zbl_f_arg = state.bessel_weights  # dummy non-null ptr
        cov_radii = state.bessel_weights

    orb_geom_zbl_featurize_kernel[(num_blocks,)](
        positions,
        atom_types,
        senders_c,
        receivers_c,
        offsets_c,
        cell_2d,
        cov_radii,
        state.bessel_weights,
        cutoff_out,
        ef_out,
        zbl_e_blk,
        zbl_f_arg,
        e_c,
        n_local,
        float(model_radius),
        state.bessel_prefactor,
        state.zbl_a_prefactor,
        state.zbl_a_exp,
        state.zbl_p,
        state.zbl_c[0], state.zbl_c[1], state.zbl_c[2], state.zbl_c[3],
        state.zbl_d[0], state.zbl_d[1], state.zbl_d[2], state.zbl_d[3],
        HAS_ZBL=state.has_zbl,
        BLOCK_E=block_e,
        num_warps=4,
    )
    if state.has_zbl:
        zbl_energy_local.add_(zbl_e_blk.sum())


def can_use_fused_orb_direct(
    orbff: nn.Module,
    positions: torch.Tensor,
    *,
    mixed_precision: str | None,
    use_fused_kernels: bool | str,
) -> bool:
    """Return True when Blackwell (sm_100) BF16 fused CuTeDSL + Triton kernels can run identically."""
    if use_fused_kernels is False:
        return False
    if mixed_precision != "bf16" or positions.dtype != torch.float32:
        if use_fused_kernels is True:
            raise RuntimeError(
                "use_fused_kernels=True requires mixed_precision='bf16' and float32 positions."
            )
        return False
    if not is_sm100_fused_orb_available(positions.device):
        if use_fused_kernels is True:
            raise RuntimeError(
                "use_fused_kernels=True requires an sm_100 CUDA GPU with CuTeDSL and Triton installed "
                "(pip install -e ./dynamis_model[orb-fused])."
            )
        return False
    gns = orbff.model
    if not getattr(gns, "outer_product_with_cutoff", False) or getattr(gns, "edge_embed_size", 0) != 128:
        return False
    ang = getattr(gns, "angular_transform", None)
    rbf = getattr(gns, "rbf_transform", None)
    if ang is None or rbf is None:
        return False
    if (
        getattr(ang, "_lmax", None) != 3
        or not getattr(ang, "normalize", False)
        or getattr(ang, "normalization", None) != "component"
    ):
        return False
    if getattr(rbf, "num_bases", None) != 8:
        return False
    if not is_compatible_3layer_mlp(gns._encoder._node_fn, expected_d_in=256):
        return False
    if not is_compatible_3layer_mlp(gns._encoder._edge_fn, expected_d_in=128):
        return False
    for gnn in gns.gnn_stacks:
        if not is_compatible_3layer_mlp(gnn._edge_mlp, expected_d_in=768):
            return False
        if not is_compatible_3layer_mlp(gnn._node_mlp, expected_d_in=768):
            return False
        if not (isinstance(gnn._receive_attn, nn.Linear) and isinstance(gnn._send_attn, nn.Linear)):
            return False
        if gnn._receive_attn.in_features != 256 or gnn._receive_attn.out_features != 1:
            return False
        if gnn._send_attn.in_features != 256 or gnn._send_attn.out_features != 1:
            return False
    if getattr(orbff, "pair_repulsion", False):
        zbl = orbff.pair_repulsion_fn
        if zbl.c.numel() != 4 or zbl.d.numel() != 4:
            return False
    return True


def run_fused_orb_backbone(
    *,
    orbff: nn.Module,
    fused: PreparedOrbFusedState,
    exchanger: Any,
    distributed_state: Any,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    mp_offsets: torch.Tensor,
    cell_2d: torch.Tensor,
    n_local_atoms: int,
    model_radius: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fused sm_100a CuTeDSL + Triton direct-family backbone.

    Returns `(nodes, zbl_energy_local, zbl_forces_local)` with coordinates,
    displacements, cutoffs, ZBL, node states, and scatter accumulators kept in FP32/FP64.
    """
    gns = orbff.model
    device = positions.device
    hidden_dim = int(gns.gnn_stacks[0].latent_dim)
    # Align chunk size to 128 so every chunk slice aligns with 2-CTA TMA tiles.
    fused_chunk = max(128, ((int(chunk_size) + 127) // 128) * 128)
    n_edges = int(senders.shape[0])
    e_pad = ((n_edges + 127) // 128) * 128
    n_atoms_total = int(atom_types.shape[0])

    # 1. Node embedding + fused 3-layer Node Encoder MLP (256 -> 1024 -> 1024 -> 256 + RMSNorm)
    nodes = torch.empty((n_atoms_total, hidden_dim), device=device, dtype=torch.float32)
    for s in range(0, n_atoms_total, fused_chunk):
        e = min(s + fused_chunk, n_atoms_total)
        emb_bf16 = gns.atom_emb.embeddings(atom_types[s:e]).to(torch.bfloat16)
        nodes[s:e] = run_fused_mlp_3layer(
            emb_bf16, fused.node_enc, fused.num_sms
        ).to(torch.float32)

    # 2. Fused Stage 4: Geometry + ZBL + Bessel(8) + SH(16) + Outer Product + Fused Edge Encoder MLP
    edges_pad = torch.empty((e_pad, hidden_dim), device=device, dtype=torch.bfloat16)
    cutoff_pad = torch.zeros((e_pad,), device=device, dtype=torch.float32)
    zbl_energy_local = torch.zeros((), device=device, dtype=torch.float64)
    zbl_forces_local = (
        torch.zeros((n_local_atoms, 3), device=device, dtype=torch.float32)
        if fused.has_zbl
        else None
    )

    if n_edges > 0:
        ef_buf = torch.empty(
            (min(fused_chunk, e_pad), 128), device=device, dtype=torch.bfloat16
        )
        cell_2d_f32 = cell_2d.to(torch.float32).contiguous()
        pos_f32 = positions.to(torch.float32).contiguous()
        atom_types_i64 = atom_types.to(torch.int64).contiguous()

        for s in range(0, n_edges, fused_chunk):
            e = min(s + fused_chunk, n_edges)
            c_len = e - s
            c_pad = ((c_len + 127) // 128) * 128
            ef_slice = ef_buf[:c_pad]
            run_fused_geom_zbl_featurize(
                pos_f32,
                atom_types_i64,
                senders[s:e],
                receivers[s:e],
                mp_offsets[s:e],
                cell_2d_f32,
                n_local_atoms,
                float(model_radius),
                fused,
                cutoff_pad[s:e],
                ef_slice[:c_len],
                zbl_energy_local,
                zbl_forces_local,
            )
            if c_pad > c_len:
                ef_slice[c_len:].zero_()
            run_fused_mlp_3layer(
                ef_slice,
                fused.edge_enc,
                fused.num_sms,
                out_bf16=edges_pad[s : s + c_pad],
            )
        del ef_buf

    del mp_offsets, positions, atom_types

    # Pad sender/receiver indices to e_pad (zero-copy when already multiple of 128)
    if e_pad == n_edges:
        senders_pad = senders
        receivers_pad = receivers
    else:
        senders_pad = torch.zeros((e_pad,), device=device, dtype=torch.int64)
        receivers_pad = torch.zeros((e_pad,), device=device, dtype=torch.int64)
        if n_edges > 0:
            senders_pad[:n_edges].copy_(senders)
            receivers_pad[:n_edges].copy_(receivers)
        del senders, receivers

    # 3. Fused Stage 3 Message-Passing Stack (in-place edges_pad + fused 3-layer node MLP)
    sent_agg = torch.empty((n_atoms_total, hidden_dim), device=device, dtype=torch.float32)
    recv_agg = torch.empty((n_atoms_total, hidden_dim), device=device, dtype=torch.float32)

    for i in range(len(fused.edge_layers)):
        if i > 0:
            nodes = exchanger.push(nodes[:n_local_atoms], distributed_state)
            if sent_agg.shape[0] != nodes.shape[0]:
                sent_agg = torch.empty_like(nodes)
                recv_agg = torch.empty_like(nodes)

        sent_agg.zero_()
        recv_agg.zero_()
        if n_edges > 0:
            run_fused_edge_layer(
                edges_pad=edges_pad,
                nodes_f32=nodes,
                senders_pad=senders_pad,
                receivers_pad=receivers_pad,
                cutoff_pad=cutoff_pad,
                sent_agg=sent_agg,
                recv_agg=recv_agg,
                weights=fused.edge_layers[i],
                e_orig=n_edges,
                num_sms=fused.num_sms,
            )

        n_nodes = int(nodes.shape[0])
        for s in range(0, n_nodes, fused_chunk):
            e = min(s + fused_chunk, n_nodes)
            nf_bf16 = torch.cat(
                [nodes[s:e], recv_agg[s:e], sent_agg[s:e]], dim=1
            ).to(torch.bfloat16)
            upd_n = run_fused_mlp_3layer(
                nf_bf16, fused.node_layers[i], fused.num_sms
            )
            nodes[s:e].add_(upd_n.to(torch.float32))

    del edges_pad, senders_pad, receivers_pad, cutoff_pad, sent_agg, recv_agg
    return nodes, zbl_energy_local, zbl_forces_local
