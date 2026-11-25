"""Fused edge_degree_embedding scatter: y[n, l, c] = s * sum_{e: tgt_e = n} sum_k W[e, l, k] R[e, k, c]

fairchem (UMASFastGPUBackend.edge_degree_scatter) does bmm([E,9,3] x [E,3,C]) -> [E,9,C] fp32 (2.67 GB at
16k) + index_add. Here: one Triton program per target node walks its CSR edge segment, so [E,9,C] never exists
(edges need not be sorted: a stable argsort of tgt is cached per tgt tensor; each program walks its
node's edges through that order) and the reduction order is fixed (bitwise reproducible, no atomics). fp32 FMA ("ieee" dots).

Backward (same per-node walk, dOut[n] loaded once):
    dR[e, k, c] = s * sum_l W[e, l, k] dOut[n, l, c]
    dW[e, l, k] = s * sum_c dOut[n, l, c] R[e, k, c]
Custom ops dynamis::ede_scatter / dynamis::ede_scatter_bwd (torch.compile friendly).

    y = ede_scatter(radial [E, 3*C], wigner_inv [E, 9, 9] L-ordered (m0 columns 0, 2, 6 read), tgt [E], N, s)
"""
import torch
import triton
import triton.language as tl

BE = 16          # edges per inner step, forward (K = 4 * BE for the dots, k = 3 masked)
BE_BWD, NW_BWD = 4, 4   # backward: edges per inner step, warps per program


@triton.jit
def _ede_fwd_kernel(R, W, rowptr, ORD, Y, s, C: tl.constexpr, BE: tl.constexpr):
    n = tl.program_id(0)
    start = tl.load(rowptr + n)
    end = tl.load(rowptr + n + 1)
    ek = tl.arange(0, 4 * BE)                 # (edge, k) pairs, k in 0..3 (k = 3 masked)
    e_off = ek // 4
    k = ek % 4
    l = tl.arange(0, 16)
    c = tl.arange(0, C)
    acc = tl.zeros((16, C), dtype=tl.float32)
    for e0 in range(start, end, BE):
        e = e0 + e_off
        valid = (e < end) & (k < 3)
        e64 = tl.load(ORD + e, mask=e < end, other=0).to(tl.int64)
        # W^T chunk [16 (l), 4BE (e,k)]
        wt = tl.load(W + e64[None, :] * 81 + l[:, None] * 9 + (k * (k + 1))[None, :],
                     mask=valid[None, :] & (l[:, None] < 9), other=0.0)
        r = tl.load(R + e64[:, None] * (3 * C) + k[:, None] * C + c[None, :],
                    mask=valid[:, None], other=0.0)                      # [4BE, C]
        acc = tl.dot(wt, r, acc, input_precision="ieee")
    acc = acc * s
    tl.store(Y + n.to(tl.int64) * (9 * C) + l[:, None] * C + c[None, :], acc, mask=l[:, None] < 9)


@triton.jit
def _ede_bwd_kernel(R, W, rowptr, ORD, DY, DR, DW, s, C: tl.constexpr, BE: tl.constexpr):
    # DW must be zero-initialised (only the m0 columns of each [9, 9] are written).
    # explicit FMAs over the 9 l rows (fp32), fixed order -> deterministic
    n = tl.program_id(0)
    start = tl.load(rowptr + n)
    end = tl.load(rowptr + n + 1)
    ek = tl.arange(0, 4 * BE)
    e_off = ek // 4
    k = ek % 4
    c = tl.arange(0, C)
    dyb = DY + n.to(tl.int64) * (9 * C)
    for e0 in range(start, end, BE):
        e = e0 + e_off
        valid = (e < end) & (k < 3)
        e64 = tl.load(ORD + e, mask=e < end, other=0).to(tl.int64)
        r = tl.load(R + e64[:, None] * (3 * C) + k[:, None] * C + c[None, :],
                    mask=valid[:, None], other=0.0)                      # [4BE, C]
        dr = tl.zeros((4 * BE, C), dtype=tl.float32)
        wbase = W + e64 * 81 + k * (k + 1)
        dwbase = DW + e64 * 81 + k * (k + 1)
        for li in tl.static_range(9):
            dyl = tl.load(dyb + li * C + c) * s                          # [C]
            wl = tl.load(wbase + li * 9, mask=valid, other=0.0)          # [4BE]
            dr += wl[:, None] * dyl[None, :]
            tl.store(dwbase + li * 9, tl.sum(r * dyl[None, :], axis=1), mask=valid)
        tl.store(DR + e64[:, None] * (3 * C) + k[:, None] * C + c[None, :], dr, mask=valid[:, None])


_ROWPTR = {}


def _rowptr(tgt, N):
    """(rowptr [N+1] int32, order [E] int32): edges of node n = order[rowptr[n]:rowptr[n+1]] (stable)"""
    hit = _ROWPTR.get("last")
    if hit is not None and hit[0] is tgt and hit[1] == tgt._version and hit[2] == N:
        return hit[3]
    st, order = torch.sort(tgt, stable=True)
    rp = torch.searchsorted(st, torch.arange(N + 1, device=tgt.device, dtype=st.dtype)).to(torch.int32)
    res = (rp, order.to(torch.int32))
    _ROWPTR["last"] = (tgt, tgt._version, N, res)
    return res


@torch.library.custom_op("dynamis::ede_scatter", mutates_args=())
def ede_scatter(radial: torch.Tensor, w_m0: torch.Tensor, tgt: torch.Tensor, N: int, s: float) -> torch.Tensor:
    R = radial.float().contiguous()
    W = w_m0.float().contiguous()
    C = R.shape[1] // 3
    Y = torch.empty(N, 9, C, device=R.device, dtype=torch.float32)
    _ede_fwd_kernel[(N,)](R, W, *_rowptr(tgt, N), Y, s, C=C, BE=BE, num_warps=4)
    return Y


@ede_scatter.register_fake
def _(radial, w_m0, tgt, N, s):
    return radial.new_empty(N, 9, radial.shape[1] // 3, dtype=torch.float32)


@torch.library.custom_op("dynamis::ede_scatter_bwd", mutates_args=())
def ede_scatter_bwd(dy: torch.Tensor, radial: torch.Tensor, w_m0: torch.Tensor, tgt: torch.Tensor,
                    s: float) -> tuple[torch.Tensor, torch.Tensor]:
    R = radial.float().contiguous()
    W = w_m0.float().contiguous()
    C = R.shape[1] // 3
    N = dy.shape[0]
    DR = torch.empty_like(R)
    DW = torch.zeros_like(W)
    _ede_bwd_kernel[(N,)](R, W, *_rowptr(tgt, N), dy.float().contiguous(), DR, DW, s, C=C, BE=BE_BWD, num_warps=NW_BWD)
    # edges whose target is outside [0, N) are never visited -> keep their grads defined (zero)
    return DR.to(radial.dtype), DW.to(w_m0.dtype)


@ede_scatter_bwd.register_fake
def _(dy, radial, w_m0, tgt, s):
    return torch.empty_like(radial), torch.empty_like(w_m0)


def _setup(ctx, inputs, output):
    radial, w_m0, tgt, N, s = inputs
    ctx.save_for_backward(radial, w_m0, tgt)
    ctx.s = s


def _bwd(ctx, dy):
    radial, w_m0, tgt = ctx.saved_tensors
    dr, dw = ede_scatter_bwd(dy, radial, w_m0, tgt, ctx.s)
    return dr, dw, None, None, None


ede_scatter.register_autograd(_bwd, setup_context=_setup)

_M0_COLS = [0, 2, 6]   # fairchem _M0_COL_INDICES_L_ORDER (L-ordered wigner_inv, lmax = 2)


def edge_degree_scatter_fused(x, radial_output, wigner_inv, scatter_target, m_0_num_coefficients,
                              sphere_channels, rescale_factor):
    """drop-in for UMASFastGPUBackend.edge_degree_scatter (L-ordered wigner_inv, sorted scatter_target)"""
    assert m_0_num_coefficients == 3
    y = ede_scatter(radial_output, wigner_inv, scatter_target, x.shape[0], 1.0 / rescale_factor)
    return x + y.to(x.dtype)
