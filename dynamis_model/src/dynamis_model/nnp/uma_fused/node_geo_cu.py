"""S3 of the fused edge-block backward (node / geometry part) as GENERATED straight-line CUDA: every (L, j) pair
and M-row unrolled with named scalar registers (no local-memory arrays), built with nvcc (common.nvcc_build) and
called through ctypes. One warp per (node, side), lane = 4 channels; the per-edge rows (da, radial, D, old dD) are
staged by a 2-stage cp.async ring in per-warp shared memory. The tgt pass (side 1) writes dx / dD, the src pass
(side 0) adds: fixed order, deterministic. Outputs: dx [N, 9, 128] fp32, dD [num_out, 81] fp32 (original edge order,
block-diagonal entries), d radial FP16 [Ep, 1536] in PACKED row order (for the folded dh GEMM, edge_block.dh_gemm).
Small partitions (N <= SPLIT_N): every node's edge list is split over SPLIT_NS warps, dx partials summed in fixed
order by ngsum.
"""
import ctypes

import torch

from dynamis_model.nnp.uma_fused.common import nvcc_build
from dynamis_model.nnp.uma_fused.reference import M_PERM, RAD_ROW

LB = [0, 1, 1, 1, 2, 2, 2, 2, 2]
BLK = {0: (0, 1), 1: (1, 4), 2: (4, 9)}
ORDER = (0, 1, 2, 3, 5, 4, 6, 7, 8)      # M-rows; rows sharing a radial row are adjacent


def _pairs():
    ps = []
    for L in range(9):
        lo, hi = BLK[LB[L]]
        ps += [(L, j) for j in range(lo, hi)]
    return ps


PAIRS = _pairs()                          # 35 (L, j), pair index p
PIDX = {lj: p for p, lj in enumerate(PAIRS)}


SPLIT_N = 4000        # N <= SPLIT_N -> split mode (dx partials [2 * SPLIT_NS, N, 1152] + ngsum)
SPLIT_NS = 3          # parts per node edge list in split mode
WPB = 4               # warps (nodes) per block
PF = 2                # cp.async pipeline stages of per-warp SMEM
STG = 4352            # bytes per stage per warp: da 9x256 | radial 6x256 | d 2x128 | old dD 2x128


def _gen():
    o = []
    w = o.append
    w("#include <cuda_fp16.h>\n#include <cstdint>\n")
    w("__constant__ int PAIR_OFF[35] = {" + ", ".join(str(L * 9 + j) for L, j in PAIRS) + "};\n")
    w("__device__ __forceinline__ void cp8(unsigned s, const void* g) {\n"
      "  asm volatile(\"cp.async.ca.shared.global [%0], [%1], 8;\" :: \"r\"(s), \"l\"(g) : \"memory\"); }\n"
      "__device__ __forceinline__ void cp16(unsigned s, const void* g) {\n"
      "  asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\" :: \"r\"(s), \"l\"(g) : \"memory\"); }\n"
      "__device__ __forceinline__ void cp4(unsigned s, const void* g) {\n"
      "  asm volatile(\"cp.async.ca.shared.global [%0], [%1], 4;\" :: \"r\"(s), \"l\"(g) : \"memory\"); }\n"
      "__device__ __forceinline__ float4 lds_h4(const unsigned char* p) {\n"
      "  uint2 u = *reinterpret_cast<const uint2*>(p);\n"
      "  float2 a = __half22float2(*reinterpret_cast<__half2*>(&u.x)), b = __half22float2(*reinterpret_cast<__half2*>(&u.y));\n"
      "  return make_float4(a.x, a.y, b.x, b.y);\n}\n")
    w(f"template <int SIDE, bool FIRST>\n__global__ void __launch_bounds__({32 * WPB}" + ") ngk(\n"
      "  const float* __restrict__ x, const float* __restrict__ d, const __half* __restrict__ da0,\n"
      "  const __half* __restrict__ da1, const __half* __restrict__ da2, const __half* __restrict__ rad,\n"
      "  const int* __restrict__ orig, const int* __restrict__ ptr, const int* __restrict__ eidx,\n"
      f"  float* __restrict__ dx, __half* __restrict__ drad, float* __restrict__ dD, int N, int ns, float* __restrict__ dxp) {{\n")
    w(f"  const int lane = threadIdx.x & 31;\n  const long wg = (long)blockIdx.x * {WPB} + (threadIdx.x >> 5);\n"
      "  const long n = wg / ns;\n  const int hh = (int)(wg - n * ns);\n"
      "  if (n >= N) return;\n  const int c = lane * 4;\n  const unsigned F = 0xffffffffu;\n")
    w("  const int offA = PAIR_OFF[lane];\n  const int offB = PAIR_OFF[lane < 3 ? 32 + lane : 32];\n")
    for j in range(9):
        w(f"  const float4 x{j} = __ldg(reinterpret_cast<const float4*>(x + n * 1152 + {j * 128} + c));\n")
        w(f"  float4 g{j} = make_float4(0.f, 0.f, 0.f, 0.f);\n")
    # start / end through a lane-0 shuffle: provably warp-uniform -> no collective fix-up code around the shuffles
    w("  const int s0_ = __shfl_sync(F, ptr[n], 0), e0_ = __shfl_sync(F, ptr[n + 1], 0);\n"
      "  const int start = s0_ + ((e0_ - s0_) * hh) / ns, end = s0_ + ((e0_ - s0_) * (hh + 1)) / ns;\n")
    # per-warp SMEM ring of PF stages; every lane reads back only the bytes its own cp.async wrote -> no syncs
    rad_q = []
    for R in ORDER:
        if RAD_ROW[R] not in rad_q:
            rad_q.append(RAD_ROW[R])
    w("  extern __shared__ __align__(16) unsigned char smraw[];\n")
    w(f"  unsigned char* const wsm = smraw + (threadIdx.x >> 5) * {PF * STG};\n")
    w("  const unsigned wsa = (unsigned)__cvta_generic_to_shared(wsm);\n")
    w("  int kb = start;\n")
    w("  int eA = start + lane < end ? eidx[start + lane] : 0;\n  int oA = start + lane < end ? orig[eA] : 0;\n")
    w("  int eB = start + 32 + lane < end ? eidx[start + 32 + lane] : 0;\n  int oB = start + 32 + lane < end ? orig[eB] : 0;\n")
    # issue(kp, stage) as a macro-like lambda body
    def issue(kp, stage):
        w(f"   {{ const int rr = {kp} - kb;\n")
        w("     const int sA = __shfl_sync(F, eA, rr & 31), sB = __shfl_sync(F, eB, rr & 31);\n")
        w("     const long e = rr < 32 ? sA : sB;\n")
        w("     const int tA = __shfl_sync(F, oA, rr & 31), tB = __shfl_sync(F, oB, rr & 31);\n     const long eo = rr < 32 ? tA : tB;\n"
          "     (void)eo;\n")
        w(f"     if ({kp} < end) {{\n")
        w(f"      const unsigned sb = wsa + ({stage}) * {STG};\n")
        w("      const long rbase = e * 1536;\n")
        rows = []      # (smem row index, global row-start expression for this side's 128 channels)
        for t, R in enumerate(ORDER):
            if R < 3:
                base = f"da0 + e * 768 + {R * 256}"
            elif R < 7:
                base = f"da1 + e * 1024 + {(R - 3) * 256}"
            else:
                base = f"da2 + e * 512 + {(R - 7) * 256}"
            rows.append(f"{base} + SIDE * 128")
        for qi, q in enumerate(rad_q):
            rows.append(f"rad + rbase + {q * 256} + SIDE * 128")
        # piece = 16 B = 8 channels, 16 pieces per row
        w("      const int c8 = (lane & 15) * 8;\n")
        for jj in range(8):
            ra, rb = 2 * jj, 2 * jj + 1
            if rb < len(rows):
                w(f"      cp16(sb + {ra * 256} + lane * 16, (lane >> 4) ? {rows[rb]} + c8 : {rows[ra]} + c8);\n")
            else:
                w(f"      if (lane < 16) cp16(sb + {ra * 256} + lane * 16, {rows[ra]} + c8);\n")
        w("      cp4(sb + 3840 + lane * 4, d + e * 81 + offA);\n      if (lane < 3) cp4(sb + 3968 + lane * 4, d + e * 81 + offB);\n")
        w("      if (!FIRST) { cp4(sb + 4096 + lane * 4, dD + eo * 81 + offA); if (lane < 3) cp4(sb + 4224 + lane * 4, dD + eo * 81 + offB); }\n")
        w("     }\n     asm volatile(\"cp.async.commit_group;\" ::: \"memory\");\n   }\n")
    w("  {\n")
    for s_ in range(PF - 1):
        issue(f"(start + {s_})", str(s_))
    w("  }\n")
    w("  int cs = 0;\n")
    w("  for (int k = start; k < end; ++k) {\n")
    w("   if (k - kb == 32) {\n    kb += 32; eA = eB; oA = oB;\n"
      "    eB = kb + 32 + lane < end ? eidx[kb + 32 + lane] : 0;\n    oB = kb + 32 + lane < end ? orig[eB] : 0;\n   }\n")
    w("   __syncwarp();\n")
    issue(f"(k + {PF - 1})", f"(cs == 0 ? {PF - 1} : cs - 1)")
    w(f"   asm volatile(\"cp.async.wait_group {PF - 1};\" ::: \"memory\");\n")
    w("   __syncwarp();\n")
    w("   {\n    const long eo = __shfl_sync(F, oA, k - kb);\n")
    w("    const long e = __shfl_sync(F, eA, k - kb);\n")   # packed row of the consumed edge (FP16 dradial row)
    w(f"    const unsigned char* const sp = wsm + cs * {STG};\n    cs = cs == {PF - 1} ? 0 : cs + 1;\n")
    w("    const float dA = *reinterpret_cast<const float*>(sp + 3840 + lane * 4);\n"
      "    const float dB = lane < 3 ? *reinterpret_cast<const float*>(sp + 3968 + lane * 4) : 0.f;\n")
    w("    float4 dr;\n")
    for p in range(35):
        w(f"    float p{p};\n")
    for t, R in enumerate(ORDER):
        L, q = M_PERM[R], RAD_ROW[R]
        lo, hi = BLK[LB[L]]
        if R < 3:
            base = f"da0 + e * 768 + {R * 256}"
        elif R < 7:
            base = f"da1 + e * 1024 + {(R - 3) * 256}"
        else:
            base = f"da2 + e * 512 + {(R - 7) * 256}"
        w(f"    {{ // M-row {R}, L {L}, radial row {q}\n")
        w(f"      const float4 a = lds_h4(sp + {t * 256} + lane * 8);\n")
        w(f"      const float4 r = lds_h4(sp + {2304 + rad_q.index(q) * 256} + lane * 8);\n")
        for j in range(lo, hi):
            P = PIDX[(L, j)]
            src = f"__shfl_sync(F, dA, {P})" if P < 32 else f"__shfl_sync(F, dB, {P - 32})"
            w(f"      const float D{j} = {src};\n")
        for hf, (c1, c2) in (("a", ("x", "y")), ("b", ("z", "w"))):
            expr = "make_float2(0.f, 0.f)"
            for j in range(lo, hi):
                expr = f"__ffma2_rn(make_float2(D{j}, D{j}), make_float2(x{j}.{c1}, x{j}.{c2}), {expr})"
            w(f"      const float2 y{hf} = {expr};\n")
        w("      const float yx = ya.x, yy = ya.y, yz = yb.x, yw = yb.y;\n")
        w("      const float4 dy = make_float4(a.x * r.x, a.y * r.y, a.z * r.z, a.w * r.w);\n")
        first_q = t == 0 or RAD_ROW[ORDER[t - 1]] != q
        last_q = t == 8 or RAD_ROW[ORDER[t + 1]] != q
        if first_q:
            w("      dr = make_float4(a.x * yx, a.y * yy, a.z * yz, a.w * yw);\n")
        else:
            w("      dr.x += a.x * yx; dr.y += a.y * yy; dr.z += a.z * yz; dr.w += a.w * yw;\n")
        if last_q:
            w(f"      {{ __half2 h01 = __floats2half2_rn(dr.x, dr.y), h23 = __floats2half2_rn(dr.z, dr.w); uint2 u; "
              f"u.x = *reinterpret_cast<unsigned*>(&h01); u.y = *reinterpret_cast<unsigned*>(&h23); "
              f"*reinterpret_cast<uint2*>(drad + e * 1536 + {q * 256} + SIDE * 128 + c) = u; }}\n")
        for j in range(lo, hi):
            P = PIDX[(L, j)]
            w(f"      {{ const float2 t = __ffma2_rn(make_float2(dy.z, dy.w), make_float2(x{j}.z, x{j}.w), "
              f"__fmul2_rn(make_float2(dy.x, dy.y), make_float2(x{j}.x, x{j}.y))); p{P} = t.x + t.y;\n")
            w(f"        const float2 ga = __ffma2_rn(make_float2(D{j}, D{j}), make_float2(dy.x, dy.y), make_float2(g{j}.x, g{j}.y));\n")
            w(f"        const float2 gb = __ffma2_rn(make_float2(D{j}, D{j}), make_float2(dy.z, dy.w), make_float2(g{j}.z, g{j}.w));\n")
            w(f"        g{j} = make_float4(ga.x, ga.y, gb.x, gb.y); }}\n")
        w("    }\n")
    # reduce-scatter p0..p31: lane ends with sum of pair `lane` in p0
    h = 16
    while h >= 1:
        w(f"    {{ const bool up = (lane & {h}) != 0;\n")
        for i in range(h):
            w(f"      {{ const float s = up ? p{i} : p{i + h}; const float kk = up ? p{i + h} : p{i}; "
              f"p{i} = kk + __shfl_xor_sync(F, s, {h}); }}\n")
        w("    }\n")
        h //= 2
    for i in (32, 33, 34):
        for hh in (16, 8, 4, 2, 1):
            w(f"    p{i} += __shfl_xor_sync(F, p{i}, {hh});\n")
    w("    const float vB = lane == 0 ? p32 : (lane == 1 ? p33 : p34);\n")
    w("    float* o = dD + eo * 81;\n")
    w("    if (FIRST) { o[offA] = p0; if (lane < 3) o[offB] = vB; }\n")
    w("    else { o[offA] = *reinterpret_cast<const float*>(sp + 4096 + lane * 4) + p0;\n"
      "           if (lane < 3) o[offB] = *reinterpret_cast<const float*>(sp + 4224 + lane * 4) + vB; }\n")
    w("   }\n  }\n  asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");\n")
    w("  if (ns > 1) {\n    float* pp = dxp + ((long)((FIRST ? 0 : ns) + hh) * N + n) * 1152 + c;\n")
    for j in range(9):
        w(f"    *reinterpret_cast<float4*>(pp + {j * 128}) = g{j};\n")
    w("    return;\n  }\n")
    for j in range(9):
        w(f"  {{ float4* q = reinterpret_cast<float4*>(dx + n * 1152 + {j * 128} + c);\n")
        w(f"    if (FIRST) *q = g{j}; else {{ float4 v = *q; v.x += g{j}.x; v.y += g{j}.y; v.z += g{j}.z; v.w += g{j}.w; *q = v; }} }}\n")
    w("}\n\n")
    w("__global__ void ngsum(const float4* __restrict__ p, float4* __restrict__ dx, long n4, int np) {\n"
      "  const long i = (long)blockIdx.x * 256 + threadIdx.x;\n  if (i >= n4) return;\n"
      "  float4 a = p[i];\n"
      "  for (int k = 1; k < np; ++k) { const float4 b = p[k * n4 + i]; a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n"
      "  dx[i] = a;\n}\n\n")
    w('extern "C" void node_geo_launch(const float* x, const float* d, const __half* da0, const __half* da1,\n'
      '    const __half* da2, const __half* rad, const int* orig, const int* ptr0, const int* eidx0,\n'
      f'    const int* ptr1, const int* eidx1, float* dx, __half* drad, float* dD, int N, int ns, float* dxp, void* stream) {{\n'
      f'  cudaStream_t s = (cudaStream_t)stream; dim3 g((N * ns + {WPB - 1}) / {WPB}), b({32 * WPB});\n'
      f'  const int SM = {WPB * PF * STG};\n'
      '  static bool init = false;\n'
      '  if (!init && SM > 0) { init = true;\n'
      '    cudaFuncSetAttribute(ngk<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);\n'
      '    cudaFuncSetAttribute(ngk<0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM); }\n'
      '  ngk<1, true><<<g, b, SM, s>>>(x, d, da0, da1, da2, rad, orig, ptr1, eidx1, dx, drad, dD, N, ns, dxp);\n'
      '  ngk<0, false><<<g, b, SM, s>>>(x, d, da0, da1, da2, rad, orig, ptr0, eidx0, dx, drad, dD, N, ns, dxp);\n'
      '  if (ns > 1) { const long n4 = (long)N * 288; ngsum<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>((const float4*)dxp, (float4*)dx, n4, 2 * ns); }\n}\n')
    return "".join(o)


_LIBS = {}


def _lib():
    if not _LIBS:
        lib = nvcc_build(_gen(), "uma_node_geo", flags=("-Xptxas", "-v"))
        lib.node_geo_launch.restype = None
        _LIBS[0] = lib
    return _LIBS[0]


def node_geo_cu(x, d, da, radial, csr, orig, num_out):
    """x [N, 9, 128] fp32, d [Ep, 81] packed Wigner, da = (da0, da1, da2) row-major FP16 [Ep, 768 | 1024 | 512],
    radial FP16 [Ep, 1536] packed, csr = (ptr0, eidx0, ptr1, eidx1) per side -> dx [N, 9, 128], dD [num_out, 81]
    (original edge order), drad FP16 [Ep, 1536] in PACKED row order (rows of padding slots unset)"""
    lib = _lib()
    dev = x.device
    N = x.shape[0]
    dx = torch.empty(N, 9, 128, device=dev, dtype=torch.float32)
    drad = torch.empty(d.shape[0], 1536, device=dev, dtype=torch.float16)
    dD = torch.zeros(num_out, 81, device=dev, dtype=torch.float32)
    ts = [x, d, *da, radial, orig, *csr, dx, drad, dD]
    for t in ts:
        assert t.is_contiguous()
    ptrs = [ctypes.c_void_p(t.data_ptr()) for t in ts]
    ns = SPLIT_NS if N <= SPLIT_N else 1
    dxp = torch.empty(2 * ns, N, 1152, device=dev, dtype=torch.float32) if ns > 1 else None
    lib.node_geo_launch(*ptrs, ctypes.c_int(N), ctypes.c_int(ns),
                        ctypes.c_void_p(dxp.data_ptr() if dxp is not None else None),
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    return dx, dD, drad
