"""S0 + S1 of the fused edge-block backward: S1 with mma.sync tensor cores (generated CUDA).

One warp = 16 consecutive packed edges (block = 8 warps = one 128-edge tile). The 16 rows form runs of equal
target node (found with __match_any_sync; a node appearing twice is one "run" = row mask). Per 32-channel chunk q,
per z-row R (m-block m), per run (node nd):
    P_m [16 e x 16 pairs] += z_R[16 e x 32 h] . Wt[nd, pairs, R-slice]^T       (B = Wt rows: h contiguous, LDG.32)
    dz_R[16 e x 32 h]     += Coef_m[16 e x 16 pairs] . Wt[nd, pairs, R-slice]  (B = movmatrix.trans of the same regs)
A rows of other runs are zeroed (coef / z masked), so every row only sees its own target's Wt.
Epilogue per (q, R): dzp = dz * act'(zp) -> dh, dsg[gate] += dz * zp; per gate group dgate -> dh.
dDinv = P (+ ct for m0) after all chunks. FP16 operands, fp32 accumulate, fixed order -> deterministic.
"""
import ctypes
import os
import subprocess

import torch

from dynamis_model.nnp.uma_fused.common import nvcc_build

from dynamis_model.nnp.uma_fused.reference import GATE_IDX, M_PERM

# Build configuration (the fastest measured on GB200 among the tile / prefetch / unroll choices):
# 8 warps x 16 edges per block, __launch_bounds__(128, 4); per z-row the next CPR = 2 rows' zp / gate / first- and
# second-run Wt words are prefetched by 16 B cp.async into a 3-buffer per-warp SMEM ring (BUFB bytes per row);
# dh stored with columns permuted inside every 16-block (one 8 B store per row and block; S2 uses dh_perm_weights).
BT = 128         # threads per block (warps x 16 edges)
KK = 1           # 16-channel k-chunks per q iteration
NQ = 8 // KK     # q iterations (16 channels each)
QU = 1           # q-loop unroll
CPR = 2          # cp.async row prefetch distance
BUFB = 2048      # SMEM bytes per prefetched row (zp 512 | gate 512 | Wt run 1 512 | Wt run 2 512)
M_OF = [0, 0, 0, 1, 1, 1, 1, 2, 2]

# ---- S0 layout (l-blocked Wt): one GEMM per l-block of i:  dOut[:, l-block] [N*nb, C] . Vcat_l [C, sumK_l], where
# Vcat_l concatenates V_m[R rows] of every M-row R whose L(R) is in block l (M-row order).
# region l = [N, nb, sumK_l]; pair (i, R) -> start_l + n*nb*sumK_l + (i-lo)*sumK_l + ROFF[R] + rr*128 + h.
LB = [0, 1, 1, 1, 2, 2, 2, 2, 2]               # l-block of L-coefficient i
BLK = {0: (0, 1), 1: (1, 4), 2: (4, 9)}        # L-coefficient range of l-block
M_ROWS = ((0, 1, 2), (3, 4, 5, 6), (7, 8))     # M-rows of m-block m
K_M = (384, 512, 256)                          # conv2 input width of m-block m (3H, 4H, 2H)


def _pairs():
    """per m-block: list of (i, R) in storage order (R-major, i ascending)"""
    out = []
    for rows in M_ROWS:
        ps = []
        for R in rows:
            lo, hi = BLK[LB[M_PERM[R]]]
            ps += [(i, R) for i in range(lo, hi)]
        out.append(ps)
    return out


PAIRS = _pairs()
LB_ROWS = {lb: [R for R in range(9) if LB[M_PERM[R]] == lb] for lb in range(3)}
LB_SUMK = {}
ROFF = {}
for _lb, _rs in LB_ROWS.items():
    _o = 0
    for _R in _rs:
        ROFF[_R] = _o
        _o += K_M[0 if _R < 3 else (1 if _R < 7 else 2)]
    LB_SUMK[_lb] = _o                                   # 384, 1408, 1920
LB_START = {0: 0, 1: 384, 2: 384 + 3 * 1408}            # x N


class S1Weights:
    """frozen conv2 weights in the S0 layout: vcat[lb] [C, sumK_lb] FP16, c0rows [9, C] fp32 (c0 slice per m0 pair)"""

    def __init__(self, w, dev):
        C = self.C = w.C
        V = [w.V0.to(dev).half(), w.V1.to(dev).half(), w.V2.to(dev).half()]   # [rows*C, K_m]
        c0 = w.c0.to(dev).float()
        self.vcat = {}
        for lb, rs in LB_ROWS.items():
            parts = []
            for R in rs:
                m = 0 if R < 3 else (1 if R < 7 else 2)
                r = R - M_ROWS[m][0]
                parts.append(V[m][r * C:(r + 1) * C])
            self.vcat[lb] = torch.cat(parts, 1).contiguous()          # [C, sumK]
        assert all(i == p for p, (i, _) in enumerate(PAIRS[0])) and len(PAIRS[0]) == 9
        self.c0rows = torch.stack([c0[R * C:(R + 1) * C] for _, R in PAIRS[0]]).contiguous()


def _tables():
    lbs, off, st, goff = [], [], [], []
    for m in range(3):
        for p in range(16):
            if p < len(PAIRS[m]):
                i, Rp = PAIRS[m][p]
                lb = LB[M_PERM[Rp]]
                lo, hi = BLK[lb]
                lbs.append(LB_START[lb])
                off.append((i - lo) * LB_SUMK[lb] + ROFF[Rp])
                st.append((hi - lo) * LB_SUMK[lb])
                goff.append(i * 9 + M_PERM[Rp])
            else:
                lbs.append(0), off.append(0), st.append(0), goff.append(-1)
    return lbs, off, st, goff


def _gen():
    lbs, off, st, goff = _tables()
    o = []
    w = o.append
    w("#include <cuda_fp16.h>\n#include <cstdint>\n")
    w("__constant__ int T_LBS[48] = {" + ",".join(map(str, lbs)) + "};\n")
    w("__constant__ int T_OFF[48] = {" + ",".join(map(str, off)) + "};\n")
    w("__constant__ int T_STR[48] = {" + ",".join(map(str, st)) + "};\n")
    w("__constant__ int T_GOFF[48] = {" + ",".join(map(str, goff)) + "};\n")
    offblk = [k for k in range(81) if k not in set(g for g in goff if g >= 0)]
    assert len(offblk) == 46, len(offblk)
    w("__constant__ int T_ZOFF[46] = {" + ",".join(map(str, offblk)) + "};\n")
    w(r"""
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t movT(uint32_t a) {
  uint32_t d; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a)); return d;
}
__device__ __forceinline__ uint32_t pk(float a, float b) {
  __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 upk(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
// Wt loads with an L2 evict-last policy (POL_EL, created once per thread)
#define ld32(p) ld32_(p, POL_EL)
__device__ __forceinline__ uint32_t ld32_(const __half* p, uint64_t pol) {
  uint32_t v; asm("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol)); return v;
}
__device__ __forceinline__ void cpz16(unsigned s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ void cpw16(unsigned s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ float ldcoef(const float* p) { return __ldg(p); }
__device__ __forceinline__ void sts64(__half* p, uint32_t a, uint32_t b) {
  __stcs(reinterpret_cast<uint2*>(p), make_uint2(a, b));
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
""")
    w(f"__global__ void __launch_bounds__({BT}, 4) s1m(const __half* __restrict__ sz, const __half* __restrict__ sg,\n"
      "  const float* __restrict__ dinv, const int* __restrict__ tgt, const int* __restrict__ orig,\n"
      "  const __half* __restrict__ wt, const float* __restrict__ ct, __half* __restrict__ dh,\n"
      "  float* __restrict__ ddinv, int Ep, int N, const int* __restrict__ cnt) {\n")
    w("  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;\n"
      "  const unsigned F = 0xffffffffu;\n"
      "  uint64_t POL_EL;\n"
      "  asm volatile(\"createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\" : \"=l\"(POL_EL));\n"
      f"  const long e0 = (long)blockIdx.x * {BT // 2} + warp * 16;\n"
      "  if (e0 >= Ep) return;\n"
      "  if (cnt != nullptr && e0 >= (long)cnt[0] * 128) return;   // capacity padding tiles\n"
      "  const int tL = tgt[e0 + (lane & 15)];\n  const int oL = orig[e0 + (lane & 15)];\n"
      "  const int tg = __shfl_sync(F, tL, g), tg8 = __shfl_sync(F, tL, g + 8);\n"
      "  const unsigned mm = __match_any_sync(F, tL);\n"
      "  unsigned leaders = __ballot_sync(F, (__ffs(mm) - 1) == lane) & 0xffffu;\n"
            "  const int nd0 = __shfl_sync(F, tL, __ffs(leaders) - 1);\n"
      "  const unsigned lead2 = leaders & (leaders - 1);\n  const bool has2 = lead2 != 0;\n"
      "  const int nd1 = __shfl_sync(F, tL, has2 ? __ffs(lead2) - 1 : __ffs(leaders) - 1);\n"
      "  const long eg = e0 + g, eg8 = e0 + g + 8;\n"
      "  __half* const dhp = dh + eg * 1408;\n")
    # coef A fragments per m-block: reg j: (row g | g+8) x pairs (2t,2t+1) + 8*(j>=2)
    for m in range(3):
        for j in range(4):
            row = "eg8" if j & 1 else "eg"
            k = f"{m * 16} + 2 * t + {8 if j >= 2 else 0}"
            w(f"  uint32_t cf{m}_{j};\n  {{ const int k0 = T_GOFF[{k}], k1 = T_GOFF[{k} + 1];\n"
              f"    cf{m}_{j} = " + f"pk(k0 >= 0 ? ldcoef(dinv + {row} * 81 + k0) : 0.f, k1 >= 0 ? ldcoef(dinv + {row} * 81 + k1) : 0.f)" + "; }\n")
    # per (m, pt) Wt pair base / node stride for pair p = 8 pt + g (stride 0 + base -1 marks padding pairs)
    for m in range(3):
        for pt in range(2):
            w(f"  const int wb{m}{pt} = T_GOFF[{m * 16 + pt * 8} + g] >= 0 ? N * T_LBS[{m * 16 + pt * 8} + g] + T_OFF[{m * 16 + pt * 8} + g] : N * 14208;   // padding pair -> zero tail\n"
              f"  const int ws{m}{pt} = T_STR[{m * 16 + pt * 8} + g];\n")
            w(f"  float P{m}{pt}[4] = {{0.f, 0.f, 0.f, 0.f}};\n")
    ROWSEQ = [0, 1, 3, 5, 2, 4, 6, 7, 8]

    def cpr_buf(i):
        """SMEM buffer of the row at sequence index i (step q*9 + i)"""
        return str(i % (CPR + 1))    # 9 rows per q, 3 buffers: static assignment

    def cpr_issue(Rn, nq, par):
        """cp.async of row Rn (q + nq) into buffer par: words 0-3 zp, 4-7 gate, 8-11 Wt B (pt, half)"""
        m = M_OF[Rn]
        rr = Rn - M_ROWS[m][0]
        gi = None if Rn == 0 else GATE_IDX[Rn - 1]
        w(f"    {{ const unsigned sb_ = cpa + ({par}) * {BUFB};\n")
        # lane L: edge e0 + (L & 15), column group (L >> 4): 16 B; smem [cg][edge][8 halves] (512 B per row)
        w(f"      cpz16(sb_ - lane * 4 + lane * 16, z16 + (q + {nq}) * 2048 + {Rn * 16 * 1024});\n")
        if Rn in (1, 2):
            w(f"      cpz16(sb_ - lane * 4 + 512 + lane * 16, g16 + (q + {nq}) * 2048 + {gi * 16 * 1024});\n")
        # lane L: pt = L >> 4, pair g' = (L >> 1) & 7, half (L & 1): 16 B; smem 1024 + L * 16
        w(f"      cpw16(sb_ - lane * 4 + 1024 + lane * 16, wt + 16 * (q + {nq}) + (lane & 1) * 8 + (nd0 * wsL{m} + wbL{m}) + {rr * 128});\n")
        w(f"      cpw16(sb_ - lane * 4 + 1536 + lane * 16, wt + 16 * (q + {nq}) + (lane & 1) * 8 + (nd1 * wsL{m} + wbL{m}) + {rr * 128});\n")
        w("    }\n")

    w("  extern __shared__ uint32_t s1smem[];\n"
      f"  const unsigned cpa = (unsigned)__cvta_generic_to_shared(s1smem + warp * {BUFB // 4 * (CPR + 1)}) + lane * 4;\n")
    for m in range(3):
        w(f"  int wbL{m}, wsL{m};\n  {{ const int ix = {m * 16} + (lane >> 4) * 8 + ((lane >> 1) & 7);\n"
          f"    wbL{m} = T_GOFF[ix] >= 0 ? N * T_LBS[ix] + T_OFF[ix] : N * 14208; wsL{m} = T_STR[ix]; }}\n")
    w("  const long e16 = e0 + (lane & 15);\n"
      "  const __half* const z16 = sz + (e16 >> 7) * (128 * 1152) + (e16 & 127) * 8 + (lane >> 4) * 1024;\n"
      "  const __half* const g16 = sg + (e16 >> 7) * (128 * 256) + (e16 & 127) * 8 + (lane >> 4) * 1024;\n"
      "  const unsigned char* const cpbb = reinterpret_cast<const unsigned char*>(s1smem + warp * " + str(BUFB // 4 * (CPR + 1)) + ");\n")
    w("  { const int q = 0;\n")
    for i0 in range(CPR):
        cpr_issue(ROWSEQ[i0], 0, cpr_buf(i0))
        w("    asm volatile(\"cp.async.commit_group;\" ::: \"memory\");\n")
    w("  }\n")

    def load(R):
        """wait for the prefetched z-row R (zp, gate, Wt of the first run) and issue the prefetch of row R + CPR"""
        w(f"    uint32_t zq{R}[{KK}][4], " + (f"gq{R}[{KK}][4], " if R else "") + f"Bf{R}[2][{KK}][2];\n")
        i = ROWSEQ.index(R)
        i2 = i + CPR
        w("    __syncwarp();\n")
        if i2 < 9:
            cpr_issue(ROWSEQ[i2], 0, cpr_buf(i2))
        else:
            w(f"    if (q + 1 < {NQ}) {{\n")
            cpr_issue(ROWSEQ[i2 - 9], 1, cpr_buf(i2))
            w("    }\n")
        w("    asm volatile(\"cp.async.commit_group;\" ::: \"memory\");\n"
          f"    asm volatile(\"cp.async.wait_group {CPR};\" ::: \"memory\");\n")
        w("    __syncwarp();\n")
        w(f"    {{ const unsigned char* b8 = cpbb + {cpr_buf(i)} * {BUFB} + 4 * t;\n")
        for j in range(4):
            off = (256 if j >= 2 else 0) + (8 * 16 if j & 1 else 0)
            w(f"      zq{R}[0][{j}] = *reinterpret_cast<const uint32_t*>(b8 + g * 16 + {off});\n")
            if R and R not in (1, 2):    # gate words of rows 1 / 2 kept in registers for their group
                w(f"      gq{R}[0][{j}] = gq{1 + GATE_IDX[R - 1]}[0][{j}];\n")
            elif R:
                w(f"      gq{R}[0][{j}] = *reinterpret_cast<const uint32_t*>(b8 + 512 + g * 16 + {off});\n")
        for pt in range(2):
            w(f"      Bf{R}[{pt}][0][0] = *reinterpret_cast<const uint32_t*>(b8 + 1024 + ({pt * 8} + g) * 32);"
              f" Bf{R}[{pt}][0][1] = *reinterpret_cast<const uint32_t*>(b8 + 1024 + ({pt * 8} + g) * 32 + 16);\n")
        w("    }\n")

    def row_body(R):
        m = M_OF[R]
        rr = R - M_ROWS[m][0]
        gi = None if R == 0 else GATE_IDX[R - 1]
        w(f"    {{ // z-row {R} (m{m}, rr {rr})\n")
        w(f"      uint32_t za[{KK}][4]; float dz[{2*KK}][4];\n      uint32_t (&zq)[{KK}][4] = zq{R};" + (f" uint32_t (&gq)[{KK}][4] = gq{R};" if R else "") + "\n")
        w("      #pragma unroll\n      " + f"for (int a = 0; a < {2*KK}; ++a)" + " { dz[a][0] = dz[a][1] = dz[a][2] = dz[a][3] = 0.f; }\n")
        w("      #pragma unroll\n      " + f"for (int kk = 0; kk < {KK}; ++kk)" + " {\n"
          "        #pragma unroll\n        for (int j = 0; j < 4; ++j) {\n")
        if R == 0:
            w("          const float2 zf = upk(zq[kk][j]); const float2 gf = make_float2(sigm(zf.x), sigm(zf.y));\n")
        else:
            w("          { __half2 zz = __hmul2(*reinterpret_cast<__half2*>(&zq[kk][j]), *reinterpret_cast<__half2*>(&gq[kk][j])); za[kk][j] = *reinterpret_cast<uint32_t*>(&zz); }\n")
        if R == 0:
            w("          za[kk][j] = pk(zf.x * gf.x, zf.y * gf.y);\n")
        w("        }\n      }\n")
        w("      unsigned lm = leaders;\n      while (lm) {\n"
          "        const int b = __ffs(lm) - 1; const bool firstrun = lm == leaders, secondrun = lm == lead2; lm &= lm - 1;\n"
          "        const int nd = __shfl_sync(F, tL, b);\n"
          "        const bool r0 = tg == nd, r1 = tg8 == nd;\n")
        w(
          f"        const uint32_t a0 = r0 ? cf{m}_0 : 0u, a1 = r1 ? cf{m}_1 : 0u, a2 = r0 ? cf{m}_2 : 0u, a3 = r1 ? cf{m}_3 : 0u;\n"
          "        " + f"uint32_t B[2][{KK}][2];" + "\n")
        w(f"        if (firstrun) {{\n          #pragma unroll\n          for (int kk = 0; kk < {KK}; ++kk) {{ B[0][kk][0] = Bf{R}[0][kk][0]; B[0][kk][1] = Bf{R}[0][kk][1]; B[1][kk][0] = Bf{R}[1][kk][0]; B[1][kk][1] = Bf{R}[1][kk][1]; }}\n        }} else ")
        i = ROWSEQ.index(R)
        w(f"if (secondrun) {{\n          const unsigned char* b2 = cpbb + {cpr_buf(i)} * {BUFB} + 4 * t + 1536;\n")
        for pt in range(2):
            w(f"          B[{pt}][0][0] = *reinterpret_cast<const uint32_t*>(b2 + ({pt * 8} + g) * 32);"
              f" B[{pt}][0][1] = *reinterpret_cast<const uint32_t*>(b2 + ({pt * 8} + g) * 32 + 16);\n")
        w("        } else ")
        w("{\n")
        for pt in range(2):
            w(f"        {{ const __half* wp = wt_b + (nd * ws{m}{pt} + wb{m}{pt}) + {rr * 128};\n"
              "          #pragma unroll\n          " + f"for (int kk = 0; kk < {KK}; ++kk)" + " {\n"
              f"            B[{pt}][kk][0] = " + "ld32(wp + 16 * kk)" + ";\n"
              f"            B[{pt}][kk][1] = " + "ld32(wp + 16 * kk + 8)" + ";\n"
              "          }\n        }\n")
        w("        }\n")
        w("        #pragma unroll\n        " + f"for (int kk = 0; kk < {KK}; ++kk)" + " {\n"
          "          const uint32_t z0 = r0 ? za[kk][0] : 0u, z1 = r1 ? za[kk][1] : 0u, z2 = r0 ? za[kk][2] : 0u, z3 = r1 ? za[kk][3] : 0u;\n"
          f"          mma(P{m}0, z0, z1, z2, z3, B[0][kk][0], B[0][kk][1]);\n"
          f"          mma(P{m}1, z0, z1, z2, z3, B[1][kk][0], B[1][kk][1]);\n"
          "          #pragma unroll\n          for (int hf = 0; hf < 2; ++hf)\n"
          "            mma(dz[2 * kk + hf], a0, a1, a2, a3, movT(B[0][kk][hf]), movT(B[1][kk][hf]));\n"
          "        }\n      }\n")
        # epilogue: dz[2kk+hf][0..1] row g <-> zp[kk][2hf], [2..3] row g8 <-> zp[kk][2hf+1]
        w("      #pragma unroll\n      " + f"for (int kk = 0; kk < {KK}; ++kk)" + " {\n"
          "        uint32_t ow[2][2];\n"
          "        #pragma unroll\n        for (int hf = 0; hf < 2; ++hf) {\n"
          "          #pragma unroll\n          for (int rw = 0; rw < 2; ++rw) {\n"
          "            const float2 z = upk(zq[kk][2 * hf + rw]);\n"
          "            const float d0 = dz[2 * kk + hf][2 * rw], d1 = dz[2 * kk + hf][2 * rw + 1];\n")
        if R == 0:
            w("            const float2 s = make_float2(sigm(z.x), sigm(z.y));\n")
            w("            const float o0 = d0 * s.x * (1.f + z.x * (1.f - s.x)), o1 = d1 * s.y * (1.f + z.y * (1.f - s.y));\n")
        else:
            w("            const float2 s = upk(gq[kk][2 * hf + rw]);\n")
            w("            const float o0 = d0 * s.x, o1 = d1 * s.y;\n")
            first = R in (1, 2)
            if first:
                w("            dsg[2 * kk + hf][2 * rw] = d0 * z.x; dsg[2 * kk + hf][2 * rw + 1] = d1 * z.y;\n")
            else:
                w("            dsg[2 * kk + hf][2 * rw] += d0 * z.x; dsg[2 * kk + hf][2 * rw + 1] += d1 * z.y;\n")
        w("            ow[hf][rw] = pk(o0, o1);\n          }\n        }\n"
          "        #pragma unroll\n        for (int rw = 0; rw < 2; ++rw)\n"
          f"          sts64(dh_b + (rw ? 8 * 1408 : 0) + {256 + R * 128} + 16 * kk, ow[0][rw], ow[1][rw]);\n      }}\n    }}\n")

    def gate_out(gi):
        w("    #pragma unroll\n    " + f"for (int kk = 0; kk < {KK}; ++kk)" + " {\n"
          "      uint32_t ow[2][2];\n"
          "      #pragma unroll\n      for (int hf = 0; hf < 2; ++hf) {\n"
          "        #pragma unroll\n        for (int rw = 0; rw < 2; ++rw) {\n"
          # gate words of the group's last row (5 / 8) are still in registers
          + f"          const float2 s = upk(gq{5 if gi == 0 else 8}[kk][2 * hf + rw]);\n"
          + "          const float o0 = dsg[2 * kk + hf][2 * rw] * s.x * (1.f - s.x), o1 = dsg[2 * kk + hf][2 * rw + 1] * s.y * (1.f - s.y);\n"
          + "          ow[hf][rw] = pk(o0, o1);\n        }\n      }\n"
          "      #pragma unroll\n      for (int rw = 0; rw < 2; ++rw)\n"
          f"        sts64(dh_b + (rw ? 8 * 1408 : 0) + {gi * 128} + 16 * kk, ow[0][rw], ow[1][rw]);\n    }}\n")

    def qhead():
        w(f"  #pragma unroll {QU}\n  " + f"for (int q = 0; q < {NQ}; ++q)" + " {\n")
        w("    " + f"float dsg[{2*KK}][4];" + "\n")
        w(f"    __half* const dh_b = dhp + 4 * t + {16 * KK} * q;\n    const __half* const wt_b = wt + {16 * KK} * q + 2 * t;\n")

    def qloop():
        qhead()
        order = [0, 1, 3, 5, "g0", 2, 4, 6, 7, 8, "g1"]
        for r in order:
            if isinstance(r, str):
                gate_out(int(r[1]))
            else:
                load(r)
                row_body(r)
        w("  }\n")
    qloop()
    # dDinv: P{m}{pt}[0..1] row g pairs 8pt+2t,+1 ; [2..3] row g8
    w("  const int og = __shfl_sync(F, oL, g), og8 = __shfl_sync(F, oL, g + 8);\n")
    for m in range(3):
        for pt in range(2):
            for c in range(4):
                rw = c >> 1
                pp = f"{pt * 8} + 2 * t + {c & 1}"
                eo, nn = ("og8", "tg8") if rw else ("og", "tg")
                w(f"  {{ const int k = T_GOFF[{m * 16} + {pp}];\n"
                  f"    if (k >= 0 && {eo} >= 0" + "" + f") ddinv[(long){eo} * 81 + k] = P{m}{pt}[{c}]"
                  + (f" + __ldg(ct + (long){nn} * 16 + {pp})" if m == 0 else "") + "; }\n")
    w("}\n\n")
    w('extern "C" void s1m_launch(const __half* sz, const __half* sg, const float* dinv, const int* tgt, const int* orig,\n'
      '    const __half* wt, const float* ct, __half* dh, float* ddinv, int Ep, int N, const int* cnt, void* stream) {\n'
      '  static bool init = false; const int smem = ' + str(BUFB * (CPR + 1) * BT // 32) + ';\n'
      '  if (!init) { cudaFuncSetAttribute(s1m, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }\n'
      f'  s1m<<<(Ep + {BT // 2 - 1}) / {BT // 2}, {BT}, smem, (cudaStream_t)stream>>>(sz, sg, dinv, tgt, orig, wt, ct, dh, ddinv, Ep, N, cnt);\n}}\n')
    return "".join(o)


_LIBS = {}


def _lib():
    if not _LIBS:
        lib = nvcc_build(_gen(), "uma_s1_mma", flags=("-Xptxas", "-v"), check=_check_ldgsts)
        lib.s1m_launch.restype = None
        _LIBS[0] = lib
    return _LIBS[0]


def _check_ldgsts(so):
    """reject builds where ptxas emitted LDGSTS with a uniform smem base + odd desc UR (illegal instruction at run time)"""
    import re
    try:
        sass = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True).stdout
    except OSError:          # no cuobjdump on PATH: skip the check
        return
    bad = [ln for ln in sass.splitlines() if "LDGSTS" in ln and re.search(r"\+UR\d+", ln.split(",")[0])
           and re.search(r"desc\[UR\d*[13579]\]", ln)]
    if bad:
        raise RuntimeError(f"s1_mma build {so}: ptxas emitted LDGSTS with a uniform smem base + odd desc UR "
                           f"(illegal instruction at run time): {bad[0].strip()}")


_S0P_KERN = []


def _s0p_kernel():
    if not _S0P_KERN:
        import triton
        import triton.language as tl

        @triton.jit
        def s0p(d_ptr, c0_ptr, a_ptr, ct_ptr, N, C: tl.constexpr):
            n = tl.program_id(0)
            i = tl.arange(0, 16)
            c = tl.arange(0, C)
            msk = (i < 9)[:, None]
            x = tl.load(d_ptr + n.to(tl.int64) * (9 * C) + i[:, None] * C + c[None, :], mask=msk, other=0.0)
            w = tl.load(c0_ptr + i[:, None] * C + c[None, :], mask=msk, other=0.0)
            ct = tl.sum(x * w, axis=1)
            tl.store(ct_ptr + n.to(tl.int64) * 16 + i, ct)
            n64 = n.to(tl.int64)
            row = tl.where(i < 1, n64, tl.where(i < 4, N + n64 * 3 + (i - 1), 4 * N + n64 * 5 + (i - 4)))
            tl.store(a_ptr + row[:, None] * C + c[None, :], x.to(tl.float16), mask=msk)
        _S0P_KERN.append(s0p)
    return _S0P_KERN[0]


def _s0_wt_lb_pad_p(s1w, dOut):
    """S0: Wt [N * 14208 + 1024] FP16 (l-blocked dOut . Vcat_l, zeroed 1024-element tail that the padding pairs of the
    mma B tiles point at) and ct [N, 16] fp32 (<dOut[n, i], c0_R> of the 9 m0 pairs); one Triton prep kernel writes
    the FP16 l-blocked dOut and ct, then 3 cuBLAS GEMMs"""
    N = dOut.shape[0]
    C = s1w.C
    dev = dOut.device
    d = dOut.contiguous()
    a = torch.empty(N * 9 * C, device=dev, dtype=torch.float16)
    ct = torch.empty(N, 16, device=dev, dtype=torch.float32)
    _s0p_kernel()[(N,)](d, s1w.c0rows, a, ct, N, C=C, num_warps=4)
    Wt = torch.empty(N * 14208 + 1024, device=dev, dtype=torch.float16)
    Wt[N * 14208:].zero_()
    for lb, (lo, hi) in BLK.items():
        nb = hi - lo
        st = N * LB_START[lb]
        torch.mm(a[N * lo * C: N * hi * C].view(N * nb, C), s1w.vcat[lb], out=Wt[st: st + N * nb * LB_SUMK[lb]].view(N * nb, LB_SUMK[lb]))
    return Wt, ct


def s1_mma(s1w, sz, sg, dinv, tgt, dOut, orig, E, cnt=None):
    """-> dh [Ep, 1408] FP16 row-major (columns permuted: dh_perm_index), dDinv [E, 81] fp32 (orig order, off-block
    zero)"""
    lib = _lib()
    dev = dOut.device
    Ep = tgt.numel()
    Wt, ct = _s0_wt_lb_pad_p(s1w, dOut)
    dh = torch.empty(Ep, 1408, device=dev, dtype=torch.float16)
    ddinv = torch.zeros(E, 81, device=dev, dtype=torch.float32)   # off-block entries stay zero
    ts = [sz, sg, dinv, tgt, orig, Wt, ct, dh, ddinv]
    for t in ts:
        assert t.is_contiguous()
    lib.s1m_launch(*[ctypes.c_void_p(t.data_ptr()) for t in ts], ctypes.c_int(Ep), ctypes.c_int(dOut.shape[0]),
                   ctypes.c_void_p(cnt.data_ptr() if cnt is not None else None),
                   ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    return dh, ddinv


def dh_perm_index(ncols=1408):
    """stored dh column p holds logical column dh_perm_index()[p] -> S2 must use W[idx] (rows permuted)"""
    idx = []
    for b in range(ncols // 16):
        for t in range(4):
            idx += [16 * b + 2 * t, 16 * b + 2 * t + 1, 16 * b + 8 + 2 * t, 16 * b + 9 + 2 * t]
    return idx


def dh_perm_weights(W16):
    """S2's three FP16 weights [640|512|256 rows, n] with rows permuted to the stored dh column order
    (use in place of blk.W16 in the S2 torch.mm calls; dh[:, :640] / [640:1152] / [1152:] slices unchanged)"""
    idx = torch.tensor(dh_perm_index(), device=W16[0].device)
    return [W16[0][idx[:640]].contiguous(), W16[1][idx[640:1152] - 640].contiguous(), W16[2][idx[1152:] - 1152].contiguous()]
