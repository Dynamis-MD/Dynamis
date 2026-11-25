"""Capacity-padded SymmMemExchanger (docs/symm_exchanger.md).

Goal: a halo rebuild (same partition, new ghost lists) must not change anything a captured CUDA graph has baked in.

Row layout of the forward output on rank `me` (fixed while the capacities hold):
    [ local rows: n_local | ghosts from peer q0: cap[q0 -> me] | ghosts from peer q1: cap[q1 -> me] | ... ]
Each directed pair (q -> me) has a fixed slot count cap[q, me] >= the current count; the first count rows of a slot
are real ghosts (same order as the unpadded exchanger), the rest are PAD ghost rows. n_full = n_local + sum_q cap.

Forward push (sender s -> receiver r):  peer_r.index_copy_(0, dst_static, x.index_select(0, src_idx))
  * dst_static = arange(slot start in r's layout, + cap[s, r]): STATIC (depends only on n_local[r] and caps).
  * src_idx [cap[s, r]] is a persistent tensor whose CONTENT is rewritten in place at every re-setup; dummy entries
    (beyond the real count) point at local row 0 -> pad ghost rows hold a copy of a real row. Nothing may read them:
    the caller must keep pad ghost rows out of the neighbour list (radius graph on real rows only, or far positions).
Backward push (ghost holder r -> owner s): r copies its slot rows g[slot(s)] (cap[s, r] rows, STATIC slice) into
  s's gbuf rows [goff[s, r], + cap[s, r]] (STATIC); s then does grad.index_add_(0, send_idx, gbuf[:sum_r cap[s, r]])
  with send_idx = the padded concatenation of the src_idx lists (dummy -> row 0). Pad ghost rows have exactly zero
  gradient (no edges, masked energy) -> the dummy entries add +0.0 to row 0 (exact, order independent).
Re-setup: all_gather of (send offsets, recv offsets, n_local) as before; caps are kept while every pair fits and
n_local is unchanged on every rank; otherwise caps = ceil(count * GHOST_HEADROOM) + GHOST_SLACK, new buffers and
`generation` += 1 (collective decision from all-gathered data -> every rank agrees). `index_generation` counts
re-setups that only rewrote index content (graph stays valid).
"""
import math

import torch
import torch.distributed as dist

from dynamis_model.comm.symm_exchanger import SymmMemExchanger, HEADROOM

import os
# per-pair slot capacity = ceil(count * GHOST_HEADROOM) + GHOST_SLACK at (re-)allocation. 1.3: measured on CH4 2k/16k
# W=2: <= 1 re-allocation per direction per 1000 steps (1.10: 2-3).
GHOST_HEADROOM = float(os.environ.get("DYNAMIS_PAD_HALO_HEADROOM", "1.3"))
GHOST_SLACK = 16

# Shared set-up plan: UMAModel.initialize_buffers used to set up 10 exchangers (position, atom
# type, 8 features) with the SAME partition_info; the all_gather of the offsets, its host copy, the padded send-index
# content and real_rows / pad_rows are identical for all of them. Between begin_setup_round() / end_setup_round()
# (called on every rank, so the collective count stays symmetric) they are computed once and reused.
_ROUND = None          # active round token (None: no sharing, every setup_exchanger computes its own)
_PLAN = {}             # {"round": token, "allm": host tensor, "derived": {cap key: (idx, real_rows, n_real, pad_rows)}}
_ROUND_COUNTER = [0]


def begin_setup_round() -> None:
    global _ROUND, _PLAN
    _ROUND_COUNTER[0] += 1
    _ROUND = _ROUND_COUNTER[0]
    _PLAN = {}


def end_setup_round() -> None:
    global _ROUND, _PLAN
    _ROUND, _PLAN = None, {}


class PaddedSymmMemExchanger(SymmMemExchanger):
    padded = True

    def setup_exchanger(self, feature_dim=None, partition_info=None, device=None, *, max_feature_dim=None):
        import torch.distributed._symmetric_memory as symm_mem

        F = self._resolve_max_feature_dim(feature_dim, max_feature_dim)
        if self._is_single_rank():
            self.is_ipc_setup = True
            return
        device = torch.device(device) if not isinstance(device, torch.device) else device
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        rank, W = dist.get_rank(), dist.get_world_size()
        n_local = len(partition_info.partition_indices)
        so = partition_info.send_offsets.to(device=device, dtype=torch.int64)
        ro = partition_info.recv_offsets.to(device=device, dtype=torch.int64)
        shared = _ROUND is not None
        if shared and _PLAN.get("round") == _ROUND and _PLAN.get("pi") is partition_info:
            allm = _PLAN["allm"]
        else:
            meta = torch.cat([so, ro, torch.tensor([n_local], device=device, dtype=torch.int64)])
            allm = [torch.zeros_like(meta) for _ in range(W)]
            dist.all_gather(allm, meta)
            allm = torch.stack(allm).cpu()
            if shared:
                _PLAN.clear()
                _PLAN.update(round=_ROUND, pi=partition_info, allm=allm, derived={})
        a_so, a_nl = allm[:, : W + 1], allm[:, -1]
        cnt = a_so[:, 1:] - a_so[:, :-1]              # cnt[s, r] = rows s sends to r (= ghosts r holds from s)
        cap = getattr(self, "cap", None)
        fits = (cap is not None and self.F == F and torch.equal(self.a_nl, a_nl) and bool((cnt <= cap).all()))
        if not fits:
            cap = torch.where(cnt > 0, torch.ceil(cnt.double() * GHOST_HEADROOM).long() + GHOST_SLACK,
                              torch.zeros_like(cnt))
            cap.fill_diagonal_(0)
            self.cap, self.a_nl, self.F = cap, a_nl.clone(), F
            # receiver layout: slot of sender q on rank r starts at n_local[r] + sum_{q' < q} cap[q', r]
            self.roff = torch.cat([torch.zeros(1, W, dtype=torch.long), torch.cumsum(cap, 0)], 0)  # [W+1, W]
            # owner gbuf layout: rows for ghost holder r start at sum_{r' < r} cap[s, r']
            self.goff = torch.cat([torch.zeros(W, 1, dtype=torch.long), torch.cumsum(cap, 1)], 1)  # [W, W+1]
            nfull = a_nl + self.roff[-1]
            need_x, need_g = int(nfull.max()), int(max(int(self.goff[:, -1].max()), 1))
            if getattr(self, "xbuf", None) is None or self.xbuf.device != device or self.xbuf.shape[1] != F \
                    or self.xbuf.shape[0] < need_x or self.gbuf.shape[0] < need_g:
                self.xbuf = self.gbuf = self.xh = self.gh = None
                cx, cg = int(need_x * HEADROOM) + 8, int(need_g * HEADROOM) + 8
                self.xbuf = symm_mem.empty(cx, F, dtype=torch.float32, device=device)
                self.xh = symm_mem.rendezvous(self.xbuf, dist.group.WORLD)
                self.gbuf = symm_mem.empty(cg, F, dtype=torch.float32, device=device)
                self.gh = symm_mem.rendezvous(self.gbuf, dist.group.WORLD)
                self.n_alloc = getattr(self, "n_alloc", 0) + 1
            cx, cg = self.xbuf.shape[0], self.gbuf.shape[0]
            self.n_local = n_local
            self.n_full = int(nfull[rank])
            self.total_send = int(self.goff[rank, -1])
            # persistent padded send index (content rewritten in place on later re-setups)
            self.send_local_indices = torch.zeros(max(self.total_send, 1), dtype=torch.int64, device=device)
            self.fwd, self.bwd = [], []
            for r in range(W):
                if r == rank:
                    continue
                c_s = int(cap[rank, r])
                if c_s:
                    g0 = int(self.goff[rank, r])
                    start = int(a_nl[r]) + int(self.roff[rank, r])
                    self.fwd.append((self.send_local_indices[g0: g0 + c_s],
                                     torch.arange(start, start + c_s, device=device),
                                     self.xh.get_buffer(r, [cx, F], torch.float32).view(-1)))
                c_r = int(cap[r, rank])
                if c_r:
                    self.bwd.append((n_local + int(self.roff[r, rank]), c_r, int(self.goff[r, rank]),
                                     self.gh.get_buffer(r, [cg, F], torch.float32).view(-1)))
            self.generation = getattr(self, "generation", 0) + 1
        key = (tuple(self.cap.flatten().tolist()), str(device))
        der = _PLAN.get("derived", {}).get(key) if shared else None
        if der is None:
            # padded send index content (dummy entries -> local row 0)
            sli = partition_info.send_local_indices.to(device=device, dtype=torch.int64)
            idx = torch.zeros(max(int(self.goff[rank, -1]), 1), dtype=torch.int64, device=device)
            for r in range(W):
                n = int(cnt[rank, r])
                if n:
                    g0 = int(self.goff[rank, r])
                    idx[g0: g0 + n] = sli[int(a_so[rank, r]): int(a_so[rank, r]) + n]
            # real-row map of this rank's padded layout: padded row of the k-th row of the UNPADDED [local | ghosts]
            # layout (callers use it to remap a neighbour list built on real rows / to scatter real rows)
            ro_h = allm[rank, W + 1: 2 * W + 2]   # this rank's recv offsets (host copy from the all_gather)
            ro_c = ro_h[1:] - ro_h[:-1]
            parts = [torch.arange(n_local)]
            for q in range(W):
                n = int(ro_c[q])
                if n:
                    s0 = n_local + int(self.roff[q, rank])
                    parts.append(torch.arange(s0, s0 + n))
            real_rows = torch.cat(parts)
            is_pad = torch.ones(self.n_full, dtype=torch.bool)
            is_pad[real_rows] = False
            der = (idx, real_rows.to(device), int(real_rows.numel()),
                   torch.nonzero(is_pad).squeeze(1).to(device))   # host-side mask: no device sync
            if shared:
                _PLAN["derived"][key] = der
        # rewrite the padded send index content in place (graph-visible tensor, fixed address)
        self.send_local_indices.copy_(der[0])
        self.real_rows, self.n_real, self.pad_rows = der[1], der[2], der[3]
        self.index_generation = getattr(self, "index_generation", 0) + 1
        self.is_ipc_setup = True
        self._ready = True

    def feature_push_backward_impl(self, grad_output, input_feature_shape, partition_info):
        g = grad_output.contiguous()
        f = int(g.shape[1])
        self.gh.barrier(channel=0)   # owners have consumed the previous backward push (pooled buffer)
        for g0, n, start, peer in self.bwd:
            peer[: self.gbuf.shape[0] * f].view(-1, f)[start: start + n].copy_(g[g0: g0 + n])
        self.gh.barrier(channel=0)
        grad_in = g[: self.n_local].clone()
        if self.total_send > 0:
            gb = self.gbuf.view(-1)[: self.gbuf.shape[0] * f].view(-1, f)
            grad_in.index_add_(0, self.send_local_indices, gb[: self.total_send])
        return grad_in
