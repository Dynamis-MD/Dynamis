"""Exact fp32 l=2 Wigner D from quaternions (replaces the TF32 einsum of fairchem's l=2 Wigner).

D[n, i, j] = sum_{abcd} C_l2[i, j, a, b, c, d] q_a q_b q_c q_d (fairchem quaternion_to_wigner_d_l2_einsum).
Only 35 distinct degree-4 monomials: C35[k, ij] = sum of C over the index orderings of combo k (folded once in fp64).
Hand-written Triton forward (35 x 25 fp32 FMAs per edge, [BLOCK, 32] tile, never a TF32 matmul) and backward
(1-D: one edge per thread, g_k = <dD, C35[k]> from 25 registers, dq_v = sum_k g_k d(mono_k)/dq_v), no atomics
(deterministic). 16k atoms (E = 580k), GB200: fwd 0.061 ms, bwd 0.055 ms; fairchem's TF32 einsum: 0.22 + 0.19 ms,
the 35-term sum as torch code compiled by inductor: 1.64 + 0.90 ms, a 2-D backward (35 row reductions): 1.54 ms.
Exposed as the torch.library custom op ``dynamis_fused::wigner_l2`` (+ ``wigner_l2_bwd``) with autograd, so
torch.compile / CUDA graphs treat it as one opaque kernel each way.
The two kernels are generated (explicit monomial terms): benchmarks/uma_fused/gen_wigner_l2_kernels.py.
"""
import itertools
import math
from collections import Counter

import torch
import triton
import triton.language as tl

_COMBOS = list(itertools.combinations_with_replacement(range(4), 4))   # 35, sorted (a<=b<=c<=d)
_PERMS = list(itertools.permutations(range(4)))
_C35_CACHE = {}


def fold_c35(C_l2: torch.Tensor) -> torch.Tensor:
    """[35, 32] fp32 (cols 25..31 zero): C35[k] = mult_k * Csym[combo_k] (Csym = C symmetrised over the 24
    index permutations), computed in fp64."""
    key = (C_l2.data_ptr(), C_l2.device, tuple(C_l2.shape))
    c = _C35_CACHE.get(key)
    if c is None:
        C = C_l2.detach().to(dtype=torch.float64).reshape(25, 4, 4, 4, 4)
        Cs = sum(C.permute(0, p[0] + 1, p[1] + 1, p[2] + 1, p[3] + 1) for p in _PERMS) / 24.0
        out = torch.zeros(35, 32, dtype=torch.float64, device=C_l2.device)
        for k, cb in enumerate(_COMBOS):
            mult = math.factorial(4) // math.prod(math.factorial(v) for v in Counter(cb).values())
            out[k, :25] = Cs[:, cb[0], cb[1], cb[2], cb[3]] * mult
        c = out.float().contiguous()
        _C35_CACHE[key] = c
    return c

@triton.jit
def _wig_l2_fwd(q_ptr, c_ptr, d_ptr, E, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < E
    e64 = e.to(tl.int64)
    q0 = tl.load(q_ptr + e64 * 4 + 0, mask=m, other=0.0)
    q1 = tl.load(q_ptr + e64 * 4 + 1, mask=m, other=0.0)
    q2 = tl.load(q_ptr + e64 * 4 + 2, mask=m, other=0.0)
    q3 = tl.load(q_ptr + e64 * 4 + 3, mask=m, other=0.0)
    j = tl.arange(0, 32)
    acc = tl.zeros([BLOCK, 32], dtype=tl.float32)
    acc += (q0*q0*q0*q0)[:, None] * tl.load(c_ptr + 0 + j)[None, :]
    acc += (q0*q0*q0*q1)[:, None] * tl.load(c_ptr + 32 + j)[None, :]
    acc += (q0*q0*q0*q2)[:, None] * tl.load(c_ptr + 64 + j)[None, :]
    acc += (q0*q0*q0*q3)[:, None] * tl.load(c_ptr + 96 + j)[None, :]
    acc += (q0*q0*q1*q1)[:, None] * tl.load(c_ptr + 128 + j)[None, :]
    acc += (q0*q0*q1*q2)[:, None] * tl.load(c_ptr + 160 + j)[None, :]
    acc += (q0*q0*q1*q3)[:, None] * tl.load(c_ptr + 192 + j)[None, :]
    acc += (q0*q0*q2*q2)[:, None] * tl.load(c_ptr + 224 + j)[None, :]
    acc += (q0*q0*q2*q3)[:, None] * tl.load(c_ptr + 256 + j)[None, :]
    acc += (q0*q0*q3*q3)[:, None] * tl.load(c_ptr + 288 + j)[None, :]
    acc += (q0*q1*q1*q1)[:, None] * tl.load(c_ptr + 320 + j)[None, :]
    acc += (q0*q1*q1*q2)[:, None] * tl.load(c_ptr + 352 + j)[None, :]
    acc += (q0*q1*q1*q3)[:, None] * tl.load(c_ptr + 384 + j)[None, :]
    acc += (q0*q1*q2*q2)[:, None] * tl.load(c_ptr + 416 + j)[None, :]
    acc += (q0*q1*q2*q3)[:, None] * tl.load(c_ptr + 448 + j)[None, :]
    acc += (q0*q1*q3*q3)[:, None] * tl.load(c_ptr + 480 + j)[None, :]
    acc += (q0*q2*q2*q2)[:, None] * tl.load(c_ptr + 512 + j)[None, :]
    acc += (q0*q2*q2*q3)[:, None] * tl.load(c_ptr + 544 + j)[None, :]
    acc += (q0*q2*q3*q3)[:, None] * tl.load(c_ptr + 576 + j)[None, :]
    acc += (q0*q3*q3*q3)[:, None] * tl.load(c_ptr + 608 + j)[None, :]
    acc += (q1*q1*q1*q1)[:, None] * tl.load(c_ptr + 640 + j)[None, :]
    acc += (q1*q1*q1*q2)[:, None] * tl.load(c_ptr + 672 + j)[None, :]
    acc += (q1*q1*q1*q3)[:, None] * tl.load(c_ptr + 704 + j)[None, :]
    acc += (q1*q1*q2*q2)[:, None] * tl.load(c_ptr + 736 + j)[None, :]
    acc += (q1*q1*q2*q3)[:, None] * tl.load(c_ptr + 768 + j)[None, :]
    acc += (q1*q1*q3*q3)[:, None] * tl.load(c_ptr + 800 + j)[None, :]
    acc += (q1*q2*q2*q2)[:, None] * tl.load(c_ptr + 832 + j)[None, :]
    acc += (q1*q2*q2*q3)[:, None] * tl.load(c_ptr + 864 + j)[None, :]
    acc += (q1*q2*q3*q3)[:, None] * tl.load(c_ptr + 896 + j)[None, :]
    acc += (q1*q3*q3*q3)[:, None] * tl.load(c_ptr + 928 + j)[None, :]
    acc += (q2*q2*q2*q2)[:, None] * tl.load(c_ptr + 960 + j)[None, :]
    acc += (q2*q2*q2*q3)[:, None] * tl.load(c_ptr + 992 + j)[None, :]
    acc += (q2*q2*q3*q3)[:, None] * tl.load(c_ptr + 1024 + j)[None, :]
    acc += (q2*q3*q3*q3)[:, None] * tl.load(c_ptr + 1056 + j)[None, :]
    acc += (q3*q3*q3*q3)[:, None] * tl.load(c_ptr + 1088 + j)[None, :]
    tl.store(d_ptr + e64[:, None] * 25 + j[None, :], acc, mask=m[:, None] & (j[None, :] < 25))

@triton.jit
def _wig_l2_bwd1(q_ptr, c_ptr, dd_ptr, dq_ptr, E, BLOCK: tl.constexpr):
    """1-D variant: one edge per thread, dD row in 25 registers, C35 scalars (uniform loads)"""
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < E
    e64 = e.to(tl.int64)
    q0 = tl.load(q_ptr + e64 * 4 + 0, mask=m, other=0.0)
    q1 = tl.load(q_ptr + e64 * 4 + 1, mask=m, other=0.0)
    q2 = tl.load(q_ptr + e64 * 4 + 2, mask=m, other=0.0)
    q3 = tl.load(q_ptr + e64 * 4 + 3, mask=m, other=0.0)
    d0 = tl.load(dd_ptr + e64 * 25 + 0, mask=m, other=0.0)
    d1 = tl.load(dd_ptr + e64 * 25 + 1, mask=m, other=0.0)
    d2 = tl.load(dd_ptr + e64 * 25 + 2, mask=m, other=0.0)
    d3 = tl.load(dd_ptr + e64 * 25 + 3, mask=m, other=0.0)
    d4 = tl.load(dd_ptr + e64 * 25 + 4, mask=m, other=0.0)
    d5 = tl.load(dd_ptr + e64 * 25 + 5, mask=m, other=0.0)
    d6 = tl.load(dd_ptr + e64 * 25 + 6, mask=m, other=0.0)
    d7 = tl.load(dd_ptr + e64 * 25 + 7, mask=m, other=0.0)
    d8 = tl.load(dd_ptr + e64 * 25 + 8, mask=m, other=0.0)
    d9 = tl.load(dd_ptr + e64 * 25 + 9, mask=m, other=0.0)
    d10 = tl.load(dd_ptr + e64 * 25 + 10, mask=m, other=0.0)
    d11 = tl.load(dd_ptr + e64 * 25 + 11, mask=m, other=0.0)
    d12 = tl.load(dd_ptr + e64 * 25 + 12, mask=m, other=0.0)
    d13 = tl.load(dd_ptr + e64 * 25 + 13, mask=m, other=0.0)
    d14 = tl.load(dd_ptr + e64 * 25 + 14, mask=m, other=0.0)
    d15 = tl.load(dd_ptr + e64 * 25 + 15, mask=m, other=0.0)
    d16 = tl.load(dd_ptr + e64 * 25 + 16, mask=m, other=0.0)
    d17 = tl.load(dd_ptr + e64 * 25 + 17, mask=m, other=0.0)
    d18 = tl.load(dd_ptr + e64 * 25 + 18, mask=m, other=0.0)
    d19 = tl.load(dd_ptr + e64 * 25 + 19, mask=m, other=0.0)
    d20 = tl.load(dd_ptr + e64 * 25 + 20, mask=m, other=0.0)
    d21 = tl.load(dd_ptr + e64 * 25 + 21, mask=m, other=0.0)
    d22 = tl.load(dd_ptr + e64 * 25 + 22, mask=m, other=0.0)
    d23 = tl.load(dd_ptr + e64 * 25 + 23, mask=m, other=0.0)
    d24 = tl.load(dd_ptr + e64 * 25 + 24, mask=m, other=0.0)
    dq0 = tl.zeros([BLOCK], dtype=tl.float32)
    dq1 = tl.zeros([BLOCK], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK], dtype=tl.float32)
    dq3 = tl.zeros([BLOCK], dtype=tl.float32)
    g = d0 * tl.load(c_ptr + 0) + d1 * tl.load(c_ptr + 1) + d2 * tl.load(c_ptr + 2) + d3 * tl.load(c_ptr + 3) + d4 * tl.load(c_ptr + 4) + d5 * tl.load(c_ptr + 5) + d6 * tl.load(c_ptr + 6) + d7 * tl.load(c_ptr + 7) + d8 * tl.load(c_ptr + 8) + d9 * tl.load(c_ptr + 9) + d10 * tl.load(c_ptr + 10) + d11 * tl.load(c_ptr + 11) + d12 * tl.load(c_ptr + 12) + d13 * tl.load(c_ptr + 13) + d14 * tl.load(c_ptr + 14) + d15 * tl.load(c_ptr + 15) + d16 * tl.load(c_ptr + 16) + d17 * tl.load(c_ptr + 17) + d18 * tl.load(c_ptr + 18) + d19 * tl.load(c_ptr + 19) + d20 * tl.load(c_ptr + 20) + d21 * tl.load(c_ptr + 21) + d22 * tl.load(c_ptr + 22) + d23 * tl.load(c_ptr + 23) + d24 * tl.load(c_ptr + 24)
    dq0 += g * (4.0 * q0*q0*q0)
    g = d0 * tl.load(c_ptr + 32) + d1 * tl.load(c_ptr + 33) + d2 * tl.load(c_ptr + 34) + d3 * tl.load(c_ptr + 35) + d4 * tl.load(c_ptr + 36) + d5 * tl.load(c_ptr + 37) + d6 * tl.load(c_ptr + 38) + d7 * tl.load(c_ptr + 39) + d8 * tl.load(c_ptr + 40) + d9 * tl.load(c_ptr + 41) + d10 * tl.load(c_ptr + 42) + d11 * tl.load(c_ptr + 43) + d12 * tl.load(c_ptr + 44) + d13 * tl.load(c_ptr + 45) + d14 * tl.load(c_ptr + 46) + d15 * tl.load(c_ptr + 47) + d16 * tl.load(c_ptr + 48) + d17 * tl.load(c_ptr + 49) + d18 * tl.load(c_ptr + 50) + d19 * tl.load(c_ptr + 51) + d20 * tl.load(c_ptr + 52) + d21 * tl.load(c_ptr + 53) + d22 * tl.load(c_ptr + 54) + d23 * tl.load(c_ptr + 55) + d24 * tl.load(c_ptr + 56)
    dq0 += g * (3.0 * q0*q0*q1)
    dq1 += g * (q0*q0*q0)
    g = d0 * tl.load(c_ptr + 64) + d1 * tl.load(c_ptr + 65) + d2 * tl.load(c_ptr + 66) + d3 * tl.load(c_ptr + 67) + d4 * tl.load(c_ptr + 68) + d5 * tl.load(c_ptr + 69) + d6 * tl.load(c_ptr + 70) + d7 * tl.load(c_ptr + 71) + d8 * tl.load(c_ptr + 72) + d9 * tl.load(c_ptr + 73) + d10 * tl.load(c_ptr + 74) + d11 * tl.load(c_ptr + 75) + d12 * tl.load(c_ptr + 76) + d13 * tl.load(c_ptr + 77) + d14 * tl.load(c_ptr + 78) + d15 * tl.load(c_ptr + 79) + d16 * tl.load(c_ptr + 80) + d17 * tl.load(c_ptr + 81) + d18 * tl.load(c_ptr + 82) + d19 * tl.load(c_ptr + 83) + d20 * tl.load(c_ptr + 84) + d21 * tl.load(c_ptr + 85) + d22 * tl.load(c_ptr + 86) + d23 * tl.load(c_ptr + 87) + d24 * tl.load(c_ptr + 88)
    dq0 += g * (3.0 * q0*q0*q2)
    dq2 += g * (q0*q0*q0)
    g = d0 * tl.load(c_ptr + 96) + d1 * tl.load(c_ptr + 97) + d2 * tl.load(c_ptr + 98) + d3 * tl.load(c_ptr + 99) + d4 * tl.load(c_ptr + 100) + d5 * tl.load(c_ptr + 101) + d6 * tl.load(c_ptr + 102) + d7 * tl.load(c_ptr + 103) + d8 * tl.load(c_ptr + 104) + d9 * tl.load(c_ptr + 105) + d10 * tl.load(c_ptr + 106) + d11 * tl.load(c_ptr + 107) + d12 * tl.load(c_ptr + 108) + d13 * tl.load(c_ptr + 109) + d14 * tl.load(c_ptr + 110) + d15 * tl.load(c_ptr + 111) + d16 * tl.load(c_ptr + 112) + d17 * tl.load(c_ptr + 113) + d18 * tl.load(c_ptr + 114) + d19 * tl.load(c_ptr + 115) + d20 * tl.load(c_ptr + 116) + d21 * tl.load(c_ptr + 117) + d22 * tl.load(c_ptr + 118) + d23 * tl.load(c_ptr + 119) + d24 * tl.load(c_ptr + 120)
    dq0 += g * (3.0 * q0*q0*q3)
    dq3 += g * (q0*q0*q0)
    g = d0 * tl.load(c_ptr + 128) + d1 * tl.load(c_ptr + 129) + d2 * tl.load(c_ptr + 130) + d3 * tl.load(c_ptr + 131) + d4 * tl.load(c_ptr + 132) + d5 * tl.load(c_ptr + 133) + d6 * tl.load(c_ptr + 134) + d7 * tl.load(c_ptr + 135) + d8 * tl.load(c_ptr + 136) + d9 * tl.load(c_ptr + 137) + d10 * tl.load(c_ptr + 138) + d11 * tl.load(c_ptr + 139) + d12 * tl.load(c_ptr + 140) + d13 * tl.load(c_ptr + 141) + d14 * tl.load(c_ptr + 142) + d15 * tl.load(c_ptr + 143) + d16 * tl.load(c_ptr + 144) + d17 * tl.load(c_ptr + 145) + d18 * tl.load(c_ptr + 146) + d19 * tl.load(c_ptr + 147) + d20 * tl.load(c_ptr + 148) + d21 * tl.load(c_ptr + 149) + d22 * tl.load(c_ptr + 150) + d23 * tl.load(c_ptr + 151) + d24 * tl.load(c_ptr + 152)
    dq0 += g * (2.0 * q0*q1*q1)
    dq1 += g * (2.0 * q0*q0*q1)
    g = d0 * tl.load(c_ptr + 160) + d1 * tl.load(c_ptr + 161) + d2 * tl.load(c_ptr + 162) + d3 * tl.load(c_ptr + 163) + d4 * tl.load(c_ptr + 164) + d5 * tl.load(c_ptr + 165) + d6 * tl.load(c_ptr + 166) + d7 * tl.load(c_ptr + 167) + d8 * tl.load(c_ptr + 168) + d9 * tl.load(c_ptr + 169) + d10 * tl.load(c_ptr + 170) + d11 * tl.load(c_ptr + 171) + d12 * tl.load(c_ptr + 172) + d13 * tl.load(c_ptr + 173) + d14 * tl.load(c_ptr + 174) + d15 * tl.load(c_ptr + 175) + d16 * tl.load(c_ptr + 176) + d17 * tl.load(c_ptr + 177) + d18 * tl.load(c_ptr + 178) + d19 * tl.load(c_ptr + 179) + d20 * tl.load(c_ptr + 180) + d21 * tl.load(c_ptr + 181) + d22 * tl.load(c_ptr + 182) + d23 * tl.load(c_ptr + 183) + d24 * tl.load(c_ptr + 184)
    dq0 += g * (2.0 * q0*q1*q2)
    dq1 += g * (q0*q0*q2)
    dq2 += g * (q0*q0*q1)
    g = d0 * tl.load(c_ptr + 192) + d1 * tl.load(c_ptr + 193) + d2 * tl.load(c_ptr + 194) + d3 * tl.load(c_ptr + 195) + d4 * tl.load(c_ptr + 196) + d5 * tl.load(c_ptr + 197) + d6 * tl.load(c_ptr + 198) + d7 * tl.load(c_ptr + 199) + d8 * tl.load(c_ptr + 200) + d9 * tl.load(c_ptr + 201) + d10 * tl.load(c_ptr + 202) + d11 * tl.load(c_ptr + 203) + d12 * tl.load(c_ptr + 204) + d13 * tl.load(c_ptr + 205) + d14 * tl.load(c_ptr + 206) + d15 * tl.load(c_ptr + 207) + d16 * tl.load(c_ptr + 208) + d17 * tl.load(c_ptr + 209) + d18 * tl.load(c_ptr + 210) + d19 * tl.load(c_ptr + 211) + d20 * tl.load(c_ptr + 212) + d21 * tl.load(c_ptr + 213) + d22 * tl.load(c_ptr + 214) + d23 * tl.load(c_ptr + 215) + d24 * tl.load(c_ptr + 216)
    dq0 += g * (2.0 * q0*q1*q3)
    dq1 += g * (q0*q0*q3)
    dq3 += g * (q0*q0*q1)
    g = d0 * tl.load(c_ptr + 224) + d1 * tl.load(c_ptr + 225) + d2 * tl.load(c_ptr + 226) + d3 * tl.load(c_ptr + 227) + d4 * tl.load(c_ptr + 228) + d5 * tl.load(c_ptr + 229) + d6 * tl.load(c_ptr + 230) + d7 * tl.load(c_ptr + 231) + d8 * tl.load(c_ptr + 232) + d9 * tl.load(c_ptr + 233) + d10 * tl.load(c_ptr + 234) + d11 * tl.load(c_ptr + 235) + d12 * tl.load(c_ptr + 236) + d13 * tl.load(c_ptr + 237) + d14 * tl.load(c_ptr + 238) + d15 * tl.load(c_ptr + 239) + d16 * tl.load(c_ptr + 240) + d17 * tl.load(c_ptr + 241) + d18 * tl.load(c_ptr + 242) + d19 * tl.load(c_ptr + 243) + d20 * tl.load(c_ptr + 244) + d21 * tl.load(c_ptr + 245) + d22 * tl.load(c_ptr + 246) + d23 * tl.load(c_ptr + 247) + d24 * tl.load(c_ptr + 248)
    dq0 += g * (2.0 * q0*q2*q2)
    dq2 += g * (2.0 * q0*q0*q2)
    g = d0 * tl.load(c_ptr + 256) + d1 * tl.load(c_ptr + 257) + d2 * tl.load(c_ptr + 258) + d3 * tl.load(c_ptr + 259) + d4 * tl.load(c_ptr + 260) + d5 * tl.load(c_ptr + 261) + d6 * tl.load(c_ptr + 262) + d7 * tl.load(c_ptr + 263) + d8 * tl.load(c_ptr + 264) + d9 * tl.load(c_ptr + 265) + d10 * tl.load(c_ptr + 266) + d11 * tl.load(c_ptr + 267) + d12 * tl.load(c_ptr + 268) + d13 * tl.load(c_ptr + 269) + d14 * tl.load(c_ptr + 270) + d15 * tl.load(c_ptr + 271) + d16 * tl.load(c_ptr + 272) + d17 * tl.load(c_ptr + 273) + d18 * tl.load(c_ptr + 274) + d19 * tl.load(c_ptr + 275) + d20 * tl.load(c_ptr + 276) + d21 * tl.load(c_ptr + 277) + d22 * tl.load(c_ptr + 278) + d23 * tl.load(c_ptr + 279) + d24 * tl.load(c_ptr + 280)
    dq0 += g * (2.0 * q0*q2*q3)
    dq2 += g * (q0*q0*q3)
    dq3 += g * (q0*q0*q2)
    g = d0 * tl.load(c_ptr + 288) + d1 * tl.load(c_ptr + 289) + d2 * tl.load(c_ptr + 290) + d3 * tl.load(c_ptr + 291) + d4 * tl.load(c_ptr + 292) + d5 * tl.load(c_ptr + 293) + d6 * tl.load(c_ptr + 294) + d7 * tl.load(c_ptr + 295) + d8 * tl.load(c_ptr + 296) + d9 * tl.load(c_ptr + 297) + d10 * tl.load(c_ptr + 298) + d11 * tl.load(c_ptr + 299) + d12 * tl.load(c_ptr + 300) + d13 * tl.load(c_ptr + 301) + d14 * tl.load(c_ptr + 302) + d15 * tl.load(c_ptr + 303) + d16 * tl.load(c_ptr + 304) + d17 * tl.load(c_ptr + 305) + d18 * tl.load(c_ptr + 306) + d19 * tl.load(c_ptr + 307) + d20 * tl.load(c_ptr + 308) + d21 * tl.load(c_ptr + 309) + d22 * tl.load(c_ptr + 310) + d23 * tl.load(c_ptr + 311) + d24 * tl.load(c_ptr + 312)
    dq0 += g * (2.0 * q0*q3*q3)
    dq3 += g * (2.0 * q0*q0*q3)
    g = d0 * tl.load(c_ptr + 320) + d1 * tl.load(c_ptr + 321) + d2 * tl.load(c_ptr + 322) + d3 * tl.load(c_ptr + 323) + d4 * tl.load(c_ptr + 324) + d5 * tl.load(c_ptr + 325) + d6 * tl.load(c_ptr + 326) + d7 * tl.load(c_ptr + 327) + d8 * tl.load(c_ptr + 328) + d9 * tl.load(c_ptr + 329) + d10 * tl.load(c_ptr + 330) + d11 * tl.load(c_ptr + 331) + d12 * tl.load(c_ptr + 332) + d13 * tl.load(c_ptr + 333) + d14 * tl.load(c_ptr + 334) + d15 * tl.load(c_ptr + 335) + d16 * tl.load(c_ptr + 336) + d17 * tl.load(c_ptr + 337) + d18 * tl.load(c_ptr + 338) + d19 * tl.load(c_ptr + 339) + d20 * tl.load(c_ptr + 340) + d21 * tl.load(c_ptr + 341) + d22 * tl.load(c_ptr + 342) + d23 * tl.load(c_ptr + 343) + d24 * tl.load(c_ptr + 344)
    dq0 += g * (q1*q1*q1)
    dq1 += g * (3.0 * q0*q1*q1)
    g = d0 * tl.load(c_ptr + 352) + d1 * tl.load(c_ptr + 353) + d2 * tl.load(c_ptr + 354) + d3 * tl.load(c_ptr + 355) + d4 * tl.load(c_ptr + 356) + d5 * tl.load(c_ptr + 357) + d6 * tl.load(c_ptr + 358) + d7 * tl.load(c_ptr + 359) + d8 * tl.load(c_ptr + 360) + d9 * tl.load(c_ptr + 361) + d10 * tl.load(c_ptr + 362) + d11 * tl.load(c_ptr + 363) + d12 * tl.load(c_ptr + 364) + d13 * tl.load(c_ptr + 365) + d14 * tl.load(c_ptr + 366) + d15 * tl.load(c_ptr + 367) + d16 * tl.load(c_ptr + 368) + d17 * tl.load(c_ptr + 369) + d18 * tl.load(c_ptr + 370) + d19 * tl.load(c_ptr + 371) + d20 * tl.load(c_ptr + 372) + d21 * tl.load(c_ptr + 373) + d22 * tl.load(c_ptr + 374) + d23 * tl.load(c_ptr + 375) + d24 * tl.load(c_ptr + 376)
    dq0 += g * (q1*q1*q2)
    dq1 += g * (2.0 * q0*q1*q2)
    dq2 += g * (q0*q1*q1)
    g = d0 * tl.load(c_ptr + 384) + d1 * tl.load(c_ptr + 385) + d2 * tl.load(c_ptr + 386) + d3 * tl.load(c_ptr + 387) + d4 * tl.load(c_ptr + 388) + d5 * tl.load(c_ptr + 389) + d6 * tl.load(c_ptr + 390) + d7 * tl.load(c_ptr + 391) + d8 * tl.load(c_ptr + 392) + d9 * tl.load(c_ptr + 393) + d10 * tl.load(c_ptr + 394) + d11 * tl.load(c_ptr + 395) + d12 * tl.load(c_ptr + 396) + d13 * tl.load(c_ptr + 397) + d14 * tl.load(c_ptr + 398) + d15 * tl.load(c_ptr + 399) + d16 * tl.load(c_ptr + 400) + d17 * tl.load(c_ptr + 401) + d18 * tl.load(c_ptr + 402) + d19 * tl.load(c_ptr + 403) + d20 * tl.load(c_ptr + 404) + d21 * tl.load(c_ptr + 405) + d22 * tl.load(c_ptr + 406) + d23 * tl.load(c_ptr + 407) + d24 * tl.load(c_ptr + 408)
    dq0 += g * (q1*q1*q3)
    dq1 += g * (2.0 * q0*q1*q3)
    dq3 += g * (q0*q1*q1)
    g = d0 * tl.load(c_ptr + 416) + d1 * tl.load(c_ptr + 417) + d2 * tl.load(c_ptr + 418) + d3 * tl.load(c_ptr + 419) + d4 * tl.load(c_ptr + 420) + d5 * tl.load(c_ptr + 421) + d6 * tl.load(c_ptr + 422) + d7 * tl.load(c_ptr + 423) + d8 * tl.load(c_ptr + 424) + d9 * tl.load(c_ptr + 425) + d10 * tl.load(c_ptr + 426) + d11 * tl.load(c_ptr + 427) + d12 * tl.load(c_ptr + 428) + d13 * tl.load(c_ptr + 429) + d14 * tl.load(c_ptr + 430) + d15 * tl.load(c_ptr + 431) + d16 * tl.load(c_ptr + 432) + d17 * tl.load(c_ptr + 433) + d18 * tl.load(c_ptr + 434) + d19 * tl.load(c_ptr + 435) + d20 * tl.load(c_ptr + 436) + d21 * tl.load(c_ptr + 437) + d22 * tl.load(c_ptr + 438) + d23 * tl.load(c_ptr + 439) + d24 * tl.load(c_ptr + 440)
    dq0 += g * (q1*q2*q2)
    dq1 += g * (q0*q2*q2)
    dq2 += g * (2.0 * q0*q1*q2)
    g = d0 * tl.load(c_ptr + 448) + d1 * tl.load(c_ptr + 449) + d2 * tl.load(c_ptr + 450) + d3 * tl.load(c_ptr + 451) + d4 * tl.load(c_ptr + 452) + d5 * tl.load(c_ptr + 453) + d6 * tl.load(c_ptr + 454) + d7 * tl.load(c_ptr + 455) + d8 * tl.load(c_ptr + 456) + d9 * tl.load(c_ptr + 457) + d10 * tl.load(c_ptr + 458) + d11 * tl.load(c_ptr + 459) + d12 * tl.load(c_ptr + 460) + d13 * tl.load(c_ptr + 461) + d14 * tl.load(c_ptr + 462) + d15 * tl.load(c_ptr + 463) + d16 * tl.load(c_ptr + 464) + d17 * tl.load(c_ptr + 465) + d18 * tl.load(c_ptr + 466) + d19 * tl.load(c_ptr + 467) + d20 * tl.load(c_ptr + 468) + d21 * tl.load(c_ptr + 469) + d22 * tl.load(c_ptr + 470) + d23 * tl.load(c_ptr + 471) + d24 * tl.load(c_ptr + 472)
    dq0 += g * (q1*q2*q3)
    dq1 += g * (q0*q2*q3)
    dq2 += g * (q0*q1*q3)
    dq3 += g * (q0*q1*q2)
    g = d0 * tl.load(c_ptr + 480) + d1 * tl.load(c_ptr + 481) + d2 * tl.load(c_ptr + 482) + d3 * tl.load(c_ptr + 483) + d4 * tl.load(c_ptr + 484) + d5 * tl.load(c_ptr + 485) + d6 * tl.load(c_ptr + 486) + d7 * tl.load(c_ptr + 487) + d8 * tl.load(c_ptr + 488) + d9 * tl.load(c_ptr + 489) + d10 * tl.load(c_ptr + 490) + d11 * tl.load(c_ptr + 491) + d12 * tl.load(c_ptr + 492) + d13 * tl.load(c_ptr + 493) + d14 * tl.load(c_ptr + 494) + d15 * tl.load(c_ptr + 495) + d16 * tl.load(c_ptr + 496) + d17 * tl.load(c_ptr + 497) + d18 * tl.load(c_ptr + 498) + d19 * tl.load(c_ptr + 499) + d20 * tl.load(c_ptr + 500) + d21 * tl.load(c_ptr + 501) + d22 * tl.load(c_ptr + 502) + d23 * tl.load(c_ptr + 503) + d24 * tl.load(c_ptr + 504)
    dq0 += g * (q1*q3*q3)
    dq1 += g * (q0*q3*q3)
    dq3 += g * (2.0 * q0*q1*q3)
    g = d0 * tl.load(c_ptr + 512) + d1 * tl.load(c_ptr + 513) + d2 * tl.load(c_ptr + 514) + d3 * tl.load(c_ptr + 515) + d4 * tl.load(c_ptr + 516) + d5 * tl.load(c_ptr + 517) + d6 * tl.load(c_ptr + 518) + d7 * tl.load(c_ptr + 519) + d8 * tl.load(c_ptr + 520) + d9 * tl.load(c_ptr + 521) + d10 * tl.load(c_ptr + 522) + d11 * tl.load(c_ptr + 523) + d12 * tl.load(c_ptr + 524) + d13 * tl.load(c_ptr + 525) + d14 * tl.load(c_ptr + 526) + d15 * tl.load(c_ptr + 527) + d16 * tl.load(c_ptr + 528) + d17 * tl.load(c_ptr + 529) + d18 * tl.load(c_ptr + 530) + d19 * tl.load(c_ptr + 531) + d20 * tl.load(c_ptr + 532) + d21 * tl.load(c_ptr + 533) + d22 * tl.load(c_ptr + 534) + d23 * tl.load(c_ptr + 535) + d24 * tl.load(c_ptr + 536)
    dq0 += g * (q2*q2*q2)
    dq2 += g * (3.0 * q0*q2*q2)
    g = d0 * tl.load(c_ptr + 544) + d1 * tl.load(c_ptr + 545) + d2 * tl.load(c_ptr + 546) + d3 * tl.load(c_ptr + 547) + d4 * tl.load(c_ptr + 548) + d5 * tl.load(c_ptr + 549) + d6 * tl.load(c_ptr + 550) + d7 * tl.load(c_ptr + 551) + d8 * tl.load(c_ptr + 552) + d9 * tl.load(c_ptr + 553) + d10 * tl.load(c_ptr + 554) + d11 * tl.load(c_ptr + 555) + d12 * tl.load(c_ptr + 556) + d13 * tl.load(c_ptr + 557) + d14 * tl.load(c_ptr + 558) + d15 * tl.load(c_ptr + 559) + d16 * tl.load(c_ptr + 560) + d17 * tl.load(c_ptr + 561) + d18 * tl.load(c_ptr + 562) + d19 * tl.load(c_ptr + 563) + d20 * tl.load(c_ptr + 564) + d21 * tl.load(c_ptr + 565) + d22 * tl.load(c_ptr + 566) + d23 * tl.load(c_ptr + 567) + d24 * tl.load(c_ptr + 568)
    dq0 += g * (q2*q2*q3)
    dq2 += g * (2.0 * q0*q2*q3)
    dq3 += g * (q0*q2*q2)
    g = d0 * tl.load(c_ptr + 576) + d1 * tl.load(c_ptr + 577) + d2 * tl.load(c_ptr + 578) + d3 * tl.load(c_ptr + 579) + d4 * tl.load(c_ptr + 580) + d5 * tl.load(c_ptr + 581) + d6 * tl.load(c_ptr + 582) + d7 * tl.load(c_ptr + 583) + d8 * tl.load(c_ptr + 584) + d9 * tl.load(c_ptr + 585) + d10 * tl.load(c_ptr + 586) + d11 * tl.load(c_ptr + 587) + d12 * tl.load(c_ptr + 588) + d13 * tl.load(c_ptr + 589) + d14 * tl.load(c_ptr + 590) + d15 * tl.load(c_ptr + 591) + d16 * tl.load(c_ptr + 592) + d17 * tl.load(c_ptr + 593) + d18 * tl.load(c_ptr + 594) + d19 * tl.load(c_ptr + 595) + d20 * tl.load(c_ptr + 596) + d21 * tl.load(c_ptr + 597) + d22 * tl.load(c_ptr + 598) + d23 * tl.load(c_ptr + 599) + d24 * tl.load(c_ptr + 600)
    dq0 += g * (q2*q3*q3)
    dq2 += g * (q0*q3*q3)
    dq3 += g * (2.0 * q0*q2*q3)
    g = d0 * tl.load(c_ptr + 608) + d1 * tl.load(c_ptr + 609) + d2 * tl.load(c_ptr + 610) + d3 * tl.load(c_ptr + 611) + d4 * tl.load(c_ptr + 612) + d5 * tl.load(c_ptr + 613) + d6 * tl.load(c_ptr + 614) + d7 * tl.load(c_ptr + 615) + d8 * tl.load(c_ptr + 616) + d9 * tl.load(c_ptr + 617) + d10 * tl.load(c_ptr + 618) + d11 * tl.load(c_ptr + 619) + d12 * tl.load(c_ptr + 620) + d13 * tl.load(c_ptr + 621) + d14 * tl.load(c_ptr + 622) + d15 * tl.load(c_ptr + 623) + d16 * tl.load(c_ptr + 624) + d17 * tl.load(c_ptr + 625) + d18 * tl.load(c_ptr + 626) + d19 * tl.load(c_ptr + 627) + d20 * tl.load(c_ptr + 628) + d21 * tl.load(c_ptr + 629) + d22 * tl.load(c_ptr + 630) + d23 * tl.load(c_ptr + 631) + d24 * tl.load(c_ptr + 632)
    dq0 += g * (q3*q3*q3)
    dq3 += g * (3.0 * q0*q3*q3)
    g = d0 * tl.load(c_ptr + 640) + d1 * tl.load(c_ptr + 641) + d2 * tl.load(c_ptr + 642) + d3 * tl.load(c_ptr + 643) + d4 * tl.load(c_ptr + 644) + d5 * tl.load(c_ptr + 645) + d6 * tl.load(c_ptr + 646) + d7 * tl.load(c_ptr + 647) + d8 * tl.load(c_ptr + 648) + d9 * tl.load(c_ptr + 649) + d10 * tl.load(c_ptr + 650) + d11 * tl.load(c_ptr + 651) + d12 * tl.load(c_ptr + 652) + d13 * tl.load(c_ptr + 653) + d14 * tl.load(c_ptr + 654) + d15 * tl.load(c_ptr + 655) + d16 * tl.load(c_ptr + 656) + d17 * tl.load(c_ptr + 657) + d18 * tl.load(c_ptr + 658) + d19 * tl.load(c_ptr + 659) + d20 * tl.load(c_ptr + 660) + d21 * tl.load(c_ptr + 661) + d22 * tl.load(c_ptr + 662) + d23 * tl.load(c_ptr + 663) + d24 * tl.load(c_ptr + 664)
    dq1 += g * (4.0 * q1*q1*q1)
    g = d0 * tl.load(c_ptr + 672) + d1 * tl.load(c_ptr + 673) + d2 * tl.load(c_ptr + 674) + d3 * tl.load(c_ptr + 675) + d4 * tl.load(c_ptr + 676) + d5 * tl.load(c_ptr + 677) + d6 * tl.load(c_ptr + 678) + d7 * tl.load(c_ptr + 679) + d8 * tl.load(c_ptr + 680) + d9 * tl.load(c_ptr + 681) + d10 * tl.load(c_ptr + 682) + d11 * tl.load(c_ptr + 683) + d12 * tl.load(c_ptr + 684) + d13 * tl.load(c_ptr + 685) + d14 * tl.load(c_ptr + 686) + d15 * tl.load(c_ptr + 687) + d16 * tl.load(c_ptr + 688) + d17 * tl.load(c_ptr + 689) + d18 * tl.load(c_ptr + 690) + d19 * tl.load(c_ptr + 691) + d20 * tl.load(c_ptr + 692) + d21 * tl.load(c_ptr + 693) + d22 * tl.load(c_ptr + 694) + d23 * tl.load(c_ptr + 695) + d24 * tl.load(c_ptr + 696)
    dq1 += g * (3.0 * q1*q1*q2)
    dq2 += g * (q1*q1*q1)
    g = d0 * tl.load(c_ptr + 704) + d1 * tl.load(c_ptr + 705) + d2 * tl.load(c_ptr + 706) + d3 * tl.load(c_ptr + 707) + d4 * tl.load(c_ptr + 708) + d5 * tl.load(c_ptr + 709) + d6 * tl.load(c_ptr + 710) + d7 * tl.load(c_ptr + 711) + d8 * tl.load(c_ptr + 712) + d9 * tl.load(c_ptr + 713) + d10 * tl.load(c_ptr + 714) + d11 * tl.load(c_ptr + 715) + d12 * tl.load(c_ptr + 716) + d13 * tl.load(c_ptr + 717) + d14 * tl.load(c_ptr + 718) + d15 * tl.load(c_ptr + 719) + d16 * tl.load(c_ptr + 720) + d17 * tl.load(c_ptr + 721) + d18 * tl.load(c_ptr + 722) + d19 * tl.load(c_ptr + 723) + d20 * tl.load(c_ptr + 724) + d21 * tl.load(c_ptr + 725) + d22 * tl.load(c_ptr + 726) + d23 * tl.load(c_ptr + 727) + d24 * tl.load(c_ptr + 728)
    dq1 += g * (3.0 * q1*q1*q3)
    dq3 += g * (q1*q1*q1)
    g = d0 * tl.load(c_ptr + 736) + d1 * tl.load(c_ptr + 737) + d2 * tl.load(c_ptr + 738) + d3 * tl.load(c_ptr + 739) + d4 * tl.load(c_ptr + 740) + d5 * tl.load(c_ptr + 741) + d6 * tl.load(c_ptr + 742) + d7 * tl.load(c_ptr + 743) + d8 * tl.load(c_ptr + 744) + d9 * tl.load(c_ptr + 745) + d10 * tl.load(c_ptr + 746) + d11 * tl.load(c_ptr + 747) + d12 * tl.load(c_ptr + 748) + d13 * tl.load(c_ptr + 749) + d14 * tl.load(c_ptr + 750) + d15 * tl.load(c_ptr + 751) + d16 * tl.load(c_ptr + 752) + d17 * tl.load(c_ptr + 753) + d18 * tl.load(c_ptr + 754) + d19 * tl.load(c_ptr + 755) + d20 * tl.load(c_ptr + 756) + d21 * tl.load(c_ptr + 757) + d22 * tl.load(c_ptr + 758) + d23 * tl.load(c_ptr + 759) + d24 * tl.load(c_ptr + 760)
    dq1 += g * (2.0 * q1*q2*q2)
    dq2 += g * (2.0 * q1*q1*q2)
    g = d0 * tl.load(c_ptr + 768) + d1 * tl.load(c_ptr + 769) + d2 * tl.load(c_ptr + 770) + d3 * tl.load(c_ptr + 771) + d4 * tl.load(c_ptr + 772) + d5 * tl.load(c_ptr + 773) + d6 * tl.load(c_ptr + 774) + d7 * tl.load(c_ptr + 775) + d8 * tl.load(c_ptr + 776) + d9 * tl.load(c_ptr + 777) + d10 * tl.load(c_ptr + 778) + d11 * tl.load(c_ptr + 779) + d12 * tl.load(c_ptr + 780) + d13 * tl.load(c_ptr + 781) + d14 * tl.load(c_ptr + 782) + d15 * tl.load(c_ptr + 783) + d16 * tl.load(c_ptr + 784) + d17 * tl.load(c_ptr + 785) + d18 * tl.load(c_ptr + 786) + d19 * tl.load(c_ptr + 787) + d20 * tl.load(c_ptr + 788) + d21 * tl.load(c_ptr + 789) + d22 * tl.load(c_ptr + 790) + d23 * tl.load(c_ptr + 791) + d24 * tl.load(c_ptr + 792)
    dq1 += g * (2.0 * q1*q2*q3)
    dq2 += g * (q1*q1*q3)
    dq3 += g * (q1*q1*q2)
    g = d0 * tl.load(c_ptr + 800) + d1 * tl.load(c_ptr + 801) + d2 * tl.load(c_ptr + 802) + d3 * tl.load(c_ptr + 803) + d4 * tl.load(c_ptr + 804) + d5 * tl.load(c_ptr + 805) + d6 * tl.load(c_ptr + 806) + d7 * tl.load(c_ptr + 807) + d8 * tl.load(c_ptr + 808) + d9 * tl.load(c_ptr + 809) + d10 * tl.load(c_ptr + 810) + d11 * tl.load(c_ptr + 811) + d12 * tl.load(c_ptr + 812) + d13 * tl.load(c_ptr + 813) + d14 * tl.load(c_ptr + 814) + d15 * tl.load(c_ptr + 815) + d16 * tl.load(c_ptr + 816) + d17 * tl.load(c_ptr + 817) + d18 * tl.load(c_ptr + 818) + d19 * tl.load(c_ptr + 819) + d20 * tl.load(c_ptr + 820) + d21 * tl.load(c_ptr + 821) + d22 * tl.load(c_ptr + 822) + d23 * tl.load(c_ptr + 823) + d24 * tl.load(c_ptr + 824)
    dq1 += g * (2.0 * q1*q3*q3)
    dq3 += g * (2.0 * q1*q1*q3)
    g = d0 * tl.load(c_ptr + 832) + d1 * tl.load(c_ptr + 833) + d2 * tl.load(c_ptr + 834) + d3 * tl.load(c_ptr + 835) + d4 * tl.load(c_ptr + 836) + d5 * tl.load(c_ptr + 837) + d6 * tl.load(c_ptr + 838) + d7 * tl.load(c_ptr + 839) + d8 * tl.load(c_ptr + 840) + d9 * tl.load(c_ptr + 841) + d10 * tl.load(c_ptr + 842) + d11 * tl.load(c_ptr + 843) + d12 * tl.load(c_ptr + 844) + d13 * tl.load(c_ptr + 845) + d14 * tl.load(c_ptr + 846) + d15 * tl.load(c_ptr + 847) + d16 * tl.load(c_ptr + 848) + d17 * tl.load(c_ptr + 849) + d18 * tl.load(c_ptr + 850) + d19 * tl.load(c_ptr + 851) + d20 * tl.load(c_ptr + 852) + d21 * tl.load(c_ptr + 853) + d22 * tl.load(c_ptr + 854) + d23 * tl.load(c_ptr + 855) + d24 * tl.load(c_ptr + 856)
    dq1 += g * (q2*q2*q2)
    dq2 += g * (3.0 * q1*q2*q2)
    g = d0 * tl.load(c_ptr + 864) + d1 * tl.load(c_ptr + 865) + d2 * tl.load(c_ptr + 866) + d3 * tl.load(c_ptr + 867) + d4 * tl.load(c_ptr + 868) + d5 * tl.load(c_ptr + 869) + d6 * tl.load(c_ptr + 870) + d7 * tl.load(c_ptr + 871) + d8 * tl.load(c_ptr + 872) + d9 * tl.load(c_ptr + 873) + d10 * tl.load(c_ptr + 874) + d11 * tl.load(c_ptr + 875) + d12 * tl.load(c_ptr + 876) + d13 * tl.load(c_ptr + 877) + d14 * tl.load(c_ptr + 878) + d15 * tl.load(c_ptr + 879) + d16 * tl.load(c_ptr + 880) + d17 * tl.load(c_ptr + 881) + d18 * tl.load(c_ptr + 882) + d19 * tl.load(c_ptr + 883) + d20 * tl.load(c_ptr + 884) + d21 * tl.load(c_ptr + 885) + d22 * tl.load(c_ptr + 886) + d23 * tl.load(c_ptr + 887) + d24 * tl.load(c_ptr + 888)
    dq1 += g * (q2*q2*q3)
    dq2 += g * (2.0 * q1*q2*q3)
    dq3 += g * (q1*q2*q2)
    g = d0 * tl.load(c_ptr + 896) + d1 * tl.load(c_ptr + 897) + d2 * tl.load(c_ptr + 898) + d3 * tl.load(c_ptr + 899) + d4 * tl.load(c_ptr + 900) + d5 * tl.load(c_ptr + 901) + d6 * tl.load(c_ptr + 902) + d7 * tl.load(c_ptr + 903) + d8 * tl.load(c_ptr + 904) + d9 * tl.load(c_ptr + 905) + d10 * tl.load(c_ptr + 906) + d11 * tl.load(c_ptr + 907) + d12 * tl.load(c_ptr + 908) + d13 * tl.load(c_ptr + 909) + d14 * tl.load(c_ptr + 910) + d15 * tl.load(c_ptr + 911) + d16 * tl.load(c_ptr + 912) + d17 * tl.load(c_ptr + 913) + d18 * tl.load(c_ptr + 914) + d19 * tl.load(c_ptr + 915) + d20 * tl.load(c_ptr + 916) + d21 * tl.load(c_ptr + 917) + d22 * tl.load(c_ptr + 918) + d23 * tl.load(c_ptr + 919) + d24 * tl.load(c_ptr + 920)
    dq1 += g * (q2*q3*q3)
    dq2 += g * (q1*q3*q3)
    dq3 += g * (2.0 * q1*q2*q3)
    g = d0 * tl.load(c_ptr + 928) + d1 * tl.load(c_ptr + 929) + d2 * tl.load(c_ptr + 930) + d3 * tl.load(c_ptr + 931) + d4 * tl.load(c_ptr + 932) + d5 * tl.load(c_ptr + 933) + d6 * tl.load(c_ptr + 934) + d7 * tl.load(c_ptr + 935) + d8 * tl.load(c_ptr + 936) + d9 * tl.load(c_ptr + 937) + d10 * tl.load(c_ptr + 938) + d11 * tl.load(c_ptr + 939) + d12 * tl.load(c_ptr + 940) + d13 * tl.load(c_ptr + 941) + d14 * tl.load(c_ptr + 942) + d15 * tl.load(c_ptr + 943) + d16 * tl.load(c_ptr + 944) + d17 * tl.load(c_ptr + 945) + d18 * tl.load(c_ptr + 946) + d19 * tl.load(c_ptr + 947) + d20 * tl.load(c_ptr + 948) + d21 * tl.load(c_ptr + 949) + d22 * tl.load(c_ptr + 950) + d23 * tl.load(c_ptr + 951) + d24 * tl.load(c_ptr + 952)
    dq1 += g * (q3*q3*q3)
    dq3 += g * (3.0 * q1*q3*q3)
    g = d0 * tl.load(c_ptr + 960) + d1 * tl.load(c_ptr + 961) + d2 * tl.load(c_ptr + 962) + d3 * tl.load(c_ptr + 963) + d4 * tl.load(c_ptr + 964) + d5 * tl.load(c_ptr + 965) + d6 * tl.load(c_ptr + 966) + d7 * tl.load(c_ptr + 967) + d8 * tl.load(c_ptr + 968) + d9 * tl.load(c_ptr + 969) + d10 * tl.load(c_ptr + 970) + d11 * tl.load(c_ptr + 971) + d12 * tl.load(c_ptr + 972) + d13 * tl.load(c_ptr + 973) + d14 * tl.load(c_ptr + 974) + d15 * tl.load(c_ptr + 975) + d16 * tl.load(c_ptr + 976) + d17 * tl.load(c_ptr + 977) + d18 * tl.load(c_ptr + 978) + d19 * tl.load(c_ptr + 979) + d20 * tl.load(c_ptr + 980) + d21 * tl.load(c_ptr + 981) + d22 * tl.load(c_ptr + 982) + d23 * tl.load(c_ptr + 983) + d24 * tl.load(c_ptr + 984)
    dq2 += g * (4.0 * q2*q2*q2)
    g = d0 * tl.load(c_ptr + 992) + d1 * tl.load(c_ptr + 993) + d2 * tl.load(c_ptr + 994) + d3 * tl.load(c_ptr + 995) + d4 * tl.load(c_ptr + 996) + d5 * tl.load(c_ptr + 997) + d6 * tl.load(c_ptr + 998) + d7 * tl.load(c_ptr + 999) + d8 * tl.load(c_ptr + 1000) + d9 * tl.load(c_ptr + 1001) + d10 * tl.load(c_ptr + 1002) + d11 * tl.load(c_ptr + 1003) + d12 * tl.load(c_ptr + 1004) + d13 * tl.load(c_ptr + 1005) + d14 * tl.load(c_ptr + 1006) + d15 * tl.load(c_ptr + 1007) + d16 * tl.load(c_ptr + 1008) + d17 * tl.load(c_ptr + 1009) + d18 * tl.load(c_ptr + 1010) + d19 * tl.load(c_ptr + 1011) + d20 * tl.load(c_ptr + 1012) + d21 * tl.load(c_ptr + 1013) + d22 * tl.load(c_ptr + 1014) + d23 * tl.load(c_ptr + 1015) + d24 * tl.load(c_ptr + 1016)
    dq2 += g * (3.0 * q2*q2*q3)
    dq3 += g * (q2*q2*q2)
    g = d0 * tl.load(c_ptr + 1024) + d1 * tl.load(c_ptr + 1025) + d2 * tl.load(c_ptr + 1026) + d3 * tl.load(c_ptr + 1027) + d4 * tl.load(c_ptr + 1028) + d5 * tl.load(c_ptr + 1029) + d6 * tl.load(c_ptr + 1030) + d7 * tl.load(c_ptr + 1031) + d8 * tl.load(c_ptr + 1032) + d9 * tl.load(c_ptr + 1033) + d10 * tl.load(c_ptr + 1034) + d11 * tl.load(c_ptr + 1035) + d12 * tl.load(c_ptr + 1036) + d13 * tl.load(c_ptr + 1037) + d14 * tl.load(c_ptr + 1038) + d15 * tl.load(c_ptr + 1039) + d16 * tl.load(c_ptr + 1040) + d17 * tl.load(c_ptr + 1041) + d18 * tl.load(c_ptr + 1042) + d19 * tl.load(c_ptr + 1043) + d20 * tl.load(c_ptr + 1044) + d21 * tl.load(c_ptr + 1045) + d22 * tl.load(c_ptr + 1046) + d23 * tl.load(c_ptr + 1047) + d24 * tl.load(c_ptr + 1048)
    dq2 += g * (2.0 * q2*q3*q3)
    dq3 += g * (2.0 * q2*q2*q3)
    g = d0 * tl.load(c_ptr + 1056) + d1 * tl.load(c_ptr + 1057) + d2 * tl.load(c_ptr + 1058) + d3 * tl.load(c_ptr + 1059) + d4 * tl.load(c_ptr + 1060) + d5 * tl.load(c_ptr + 1061) + d6 * tl.load(c_ptr + 1062) + d7 * tl.load(c_ptr + 1063) + d8 * tl.load(c_ptr + 1064) + d9 * tl.load(c_ptr + 1065) + d10 * tl.load(c_ptr + 1066) + d11 * tl.load(c_ptr + 1067) + d12 * tl.load(c_ptr + 1068) + d13 * tl.load(c_ptr + 1069) + d14 * tl.load(c_ptr + 1070) + d15 * tl.load(c_ptr + 1071) + d16 * tl.load(c_ptr + 1072) + d17 * tl.load(c_ptr + 1073) + d18 * tl.load(c_ptr + 1074) + d19 * tl.load(c_ptr + 1075) + d20 * tl.load(c_ptr + 1076) + d21 * tl.load(c_ptr + 1077) + d22 * tl.load(c_ptr + 1078) + d23 * tl.load(c_ptr + 1079) + d24 * tl.load(c_ptr + 1080)
    dq2 += g * (q3*q3*q3)
    dq3 += g * (3.0 * q2*q3*q3)
    g = d0 * tl.load(c_ptr + 1088) + d1 * tl.load(c_ptr + 1089) + d2 * tl.load(c_ptr + 1090) + d3 * tl.load(c_ptr + 1091) + d4 * tl.load(c_ptr + 1092) + d5 * tl.load(c_ptr + 1093) + d6 * tl.load(c_ptr + 1094) + d7 * tl.load(c_ptr + 1095) + d8 * tl.load(c_ptr + 1096) + d9 * tl.load(c_ptr + 1097) + d10 * tl.load(c_ptr + 1098) + d11 * tl.load(c_ptr + 1099) + d12 * tl.load(c_ptr + 1100) + d13 * tl.load(c_ptr + 1101) + d14 * tl.load(c_ptr + 1102) + d15 * tl.load(c_ptr + 1103) + d16 * tl.load(c_ptr + 1104) + d17 * tl.load(c_ptr + 1105) + d18 * tl.load(c_ptr + 1106) + d19 * tl.load(c_ptr + 1107) + d20 * tl.load(c_ptr + 1108) + d21 * tl.load(c_ptr + 1109) + d22 * tl.load(c_ptr + 1110) + d23 * tl.load(c_ptr + 1111) + d24 * tl.load(c_ptr + 1112)
    dq3 += g * (4.0 * q3*q3*q3)
    tl.store(dq_ptr + e64 * 4 + 0, dq0, mask=m)
    tl.store(dq_ptr + e64 * 4 + 1, dq1, mask=m)
    tl.store(dq_ptr + e64 * 4 + 2, dq2, mask=m)
    tl.store(dq_ptr + e64 * 4 + 3, dq3, mask=m)


_BLOCK = 128


@torch.library.custom_op("dynamis_fused::wigner_l2", mutates_args=())
def wigner_l2(q: torch.Tensor, C_l2: torch.Tensor) -> torch.Tensor:
    """q [E, 4] fp32, C_l2 fairchem coefficient buffer -> D(l=2) [E, 5, 5] fp32 (C35 folded once, cached)"""
    c35 = fold_c35(C_l2)
    q = q.contiguous()
    E = q.shape[0]
    d = torch.empty(E, 5, 5, dtype=torch.float32, device=q.device)
    if E:
        _wig_l2_fwd[(triton.cdiv(E, _BLOCK),)](q, c35, d, E, BLOCK=_BLOCK, num_warps=4)
    return d


@wigner_l2.register_fake
def _(q, C_l2):
    return q.new_empty(q.shape[0], 5, 5)


@torch.library.custom_op("dynamis_fused::wigner_l2_bwd", mutates_args=())
def wigner_l2_bwd(q: torch.Tensor, C_l2: torch.Tensor, dd: torch.Tensor) -> torch.Tensor:
    c35 = fold_c35(C_l2)
    q = q.contiguous()
    dd = dd.contiguous()
    E = q.shape[0]
    dq = torch.empty(E, 4, dtype=torch.float32, device=q.device)
    if E:
        _wig_l2_bwd1[(triton.cdiv(E, 256),)](q, c35, dd, dq, E, BLOCK=256, num_warps=4)
    return dq


@wigner_l2_bwd.register_fake
def _(q, C_l2, dd):
    return q.new_empty(q.shape[0], 4)


def _setup_ctx(ctx, inputs, output):
    q, C_l2 = inputs
    ctx.save_for_backward(q, C_l2)


def _backward(ctx, dd):
    q, C_l2 = ctx.saved_tensors
    return wigner_l2_bwd(q, C_l2, dd), None


wigner_l2.register_autograd(_backward, setup_context=_setup_ctx)


def quaternion_to_wigner_d_l2_triton(q: torch.Tensor, C_l2: torch.Tensor) -> torch.Tensor:
    """drop-in for fairchem quaternion_to_wigner_d_l2_einsum"""
    return wigner_l2(q.float(), C_l2).to(q.dtype)
