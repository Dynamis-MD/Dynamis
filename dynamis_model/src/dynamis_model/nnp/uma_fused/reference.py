"""Reference math for one UMA-s (lmax = mmax = 2) edge-wise block.

This is the exact computation the fused CuTe kernel implements, written as
plain PyTorch in the *packed m-block* form used by fairchem's umas_fast_gpu
backend (so it can be checked against fairchem bit-for-bit in fp32 and used
as an fp64 ground truth for the kernel).

Per edge e = (src, tgt), with node features x[N, 9, C] (L-major) and
block-diagonal Wigner D[e] (L-order, 9x9), radial r[e, 6*2C]:

  1. rotate:   ys = D x[src],  yt = D x[tgt]                      (L-order)
  2. L->M permute (M_PERM) and per-M-row radial scaling (RAD_ROW):
        A_m0[e] = [M0, M1, M2]   rows of (ys|yt) * r      -> [E, 3*2C]
        A_m1[e] = [M3, M4, M5, M6]                        -> [E, 4*2C]
        A_m2[e] = [M7, M8]                                -> [E, 2*2C]
  3. conv1:    h0 = A_m0 W0^T + b0  -> split (gate [E, 2*H], s0 [E, 3*H])
               h1 = A_m1 W1^T        ([E, 4*H] = r/i x l=1,2)
               h2 = A_m2 W2^T        ([E, 2*H] = r/i x l=2)
  4. gate:     row M0 -> SiLU; rows M1..M8 -> * sigmoid(gate)[l(row)-1]
  5. conv2:    g0 = z_m0 V0^T + c0 ([E, 3C]); g1 = z_m1 V1^T; g2 = z_m2 V2^T
  6. M->L unpack (inverse of M_PERM), rotate back with
     Dinv_env[e] = envelope(e) * D[e]^T, and segment-sum over edges into tgt.

H = hidden channels (128), C = sphere channels (128).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# M-row j takes L-coefficient M_PERM[j]   (fairchem triton kernels.py)
M_PERM = (0, 2, 6, 3, 7, 1, 5, 8, 4)
# L-coefficient i comes from M-row L_FROM_M[i]  (inverse permutation)
L_FROM_M = tuple(M_PERM.index(i) for i in range(9))
# radial block (width 2C) used by M-row j
RAD_ROW = (0, 1, 2, 3, 4, 3, 4, 5, 5)
# gate index (l - 1) for M-rows 1..8 (GateActivation, m_prime=True)
GATE_IDX = (0, 1, 0, 1, 0, 1, 1, 1)


@dataclass
class EdgeBlockWeights:
    """Dense weights of one merged UMA-s Edgewise block."""

    W0: torch.Tensor  # [2H + 3H, 3*2C]   conv1 fc_m0 (gate channels first)
    b0: torch.Tensor  # [2H + 3H]
    W1: torch.Tensor  # [4H, 4*2C]        conv1 m=1 block weight
    W2: torch.Tensor  # [2H, 2*2C]        conv1 m=2 block weight
    V0: torch.Tensor  # [3C, 3H]          conv2 fc_m0
    c0: torch.Tensor  # [3C]
    V1: torch.Tensor  # [4C, 4H]
    V2: torch.Tensor  # [2C, 2H]
    H: int
    C: int

    @classmethod
    def from_edgewise(cls, ew) -> "EdgeBlockWeights":
        """Extract from a fairchem Edgewise after umas_fast_gpu preparation."""
        c1, c2 = ew.so2_conv_1, ew.so2_conv_2
        for conv in (c1, c2):
            for m in conv.so2_m_conv:
                if m._w_block is None:
                    m._build_w_block()

        def lin(fc):
            w = fc.weight if hasattr(fc, "weight") else fc.fc.weight
            b = getattr(fc, "bias", None)
            return w.detach(), (b.detach() if b is not None else None)

        W0, b0 = lin(c1.fc_m0)
        V0, c0 = lin(c2.fc_m0)
        H = ew.hidden_channels
        C = ew.sphere_channels
        if b0 is None:
            b0 = torch.zeros(W0.shape[0], dtype=W0.dtype, device=W0.device)
        if c0 is None:
            c0 = torch.zeros(V0.shape[0], dtype=V0.dtype, device=V0.device)
        return cls(
            W0=W0, b0=b0,
            W1=c1.so2_m_conv[0]._w_block.detach(),
            W2=c1.so2_m_conv[1]._w_block.detach(),
            V0=V0, c0=c0,
            V1=c2.so2_m_conv[0]._w_block.detach(),
            V2=c2.so2_m_conv[1]._w_block.detach(),
            H=H, C=C,
        )

    def to(self, dtype=None, device=None) -> "EdgeBlockWeights":
        kw = {}
        for f in ("W0", "b0", "W1", "W2", "V0", "c0", "V1", "V2"):
            kw[f] = getattr(self, f).to(dtype=dtype, device=device)
        return EdgeBlockWeights(H=self.H, C=self.C, **kw)


def rotate_pack(x, src, tgt, D, radial):
    """Steps 1-2. Returns A_m0 [E,3*2C], A_m1 [E,4*2C], A_m2 [E,2*2C]."""
    E = src.numel()
    C = x.shape[2]
    ys = torch.bmm(D, x[src])  # [E, 9, C]
    yt = torch.bmm(D, x[tgt])
    y = torch.cat((ys, yt), dim=2)[:, list(M_PERM)]  # [E, 9(M), 2C]
    r = radial.view(E, 6, 2 * C)[:, list(RAD_ROW)]  # [E, 9(M), 2C]
    a = y * r
    return (a[:, 0:3].reshape(E, -1), a[:, 3:7].reshape(E, -1),
            a[:, 7:9].reshape(E, -1))


def conv_gate_conv(a0, a1, a2, w: EdgeBlockWeights):
    """Steps 3-5. Returns g0 [E,3C], g1 [E,4C], g2 [E,2C] (M-row blocks)."""
    E, H = a0.shape[0], w.H
    h0 = a0 @ w.W0.T + w.b0
    gate, s0 = h0[:, : 2 * H], h0[:, 2 * H:]
    h1 = a1 @ w.W1.T
    h2 = a2 @ w.W2.T
    # M-order hidden rows [E, 9, H]
    z = torch.cat((s0.view(E, 3, H), h1.view(E, 4, H), h2.view(E, 2, H)), dim=1)
    g = torch.sigmoid(gate).view(E, 2, H)[:, list(GATE_IDX)]  # [E, 8, H]
    z = torch.cat((torch.nn.functional.silu(z[:, :1]), z[:, 1:] * g), dim=1)
    g0 = z[:, 0:3].reshape(E, -1) @ w.V0.T + w.c0
    g1 = z[:, 3:7].reshape(E, -1) @ w.V1.T
    g2 = z[:, 7:9].reshape(E, -1) @ w.V2.T
    return g0, g1, g2


def unpack_rotate_scatter(g0, g1, g2, Dinv_env, tgt, num_nodes):
    """Step 6. Returns node messages [num_nodes, 9, C]."""
    E = g0.shape[0]
    C = g0.shape[1] // 3
    vM = torch.cat((g0.view(E, 3, C), g1.view(E, 4, C), g2.view(E, 2, C)), dim=1)
    vL = vM[:, list(L_FROM_M)]
    out_e = torch.bmm(Dinv_env, vL)
    out = torch.zeros((num_nodes, 9, C), dtype=out_e.dtype, device=out_e.device)
    return out.index_add_(0, tgt, out_e)


def edge_block_reference(x, edge_index, D, Dinv_env, radial, w: EdgeBlockWeights):
    """Full edge-wise block: node features x[N,9,C] -> messages [N,9,C]."""
    src, tgt = edge_index[0], edge_index[1]
    a0, a1, a2 = rotate_pack(x, src, tgt, D, radial)
    g0, g1, g2 = conv_gate_conv(a0, a1, a2, w)
    return unpack_rotate_scatter(g0, g1, g2, Dinv_env, tgt, x.shape[0])
