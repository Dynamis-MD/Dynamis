"""CUDA-graph-capturable single-node halo exchanger on torch symmetric memory (docs/symm_exchanger.md).

Same contract / layout as GPUDirectExchanger: forward([n_local, F]) -> [n_local + n_ghost, F] = [local | ghosts ordered
by source rank (recv_offsets)]; backward(grad [n_full, F]) -> grad [n_local, F] = own rows + ghost grads pushed back by
the peers that hold this rank's atoms as ghosts.

Everything after setup is plain stream-ordered torch ops on fixed buffers + the symmetric-memory barrier (device-side
signal pads), no host sync and no host-side generation counter: capturable, and correct under graph replay
(benchmarks/uma_fused/dist_capture_test.py). The GPUDirectExchanger's cuStreamWaitValue32 generation is a host
counter frozen into a captured graph -> stale ghosts from the 2nd replay on.

Default exchanger of UMAModel(fused_graph=True) on a single node. Re-setup (halo rebuild /
repartition) keeps the symmetric buffers while they are large enough (allocation = 1.25x the all-rank maximum rows):
re-allocating them (symm_mem.empty + rendezvous, 12 per UMA model) cost ~1.3 s per halo rebuild, reuse ~5 ms.
Every re-setup bumps `generation` (new index tensors / row counts) -> whole-call graphs re-capture (graph key).
Limits: single node (NVLink P2P), fp32 features, no Python-side drift-flag piggyback (Simulator then uses its
"all_reduce" drift mode automatically).

Pooled (#148 Exchanger API): ONE instance serves every push width 1 <= F <= max_feature_dim (atom types, positions,
all feature layers). The symmetric buffers are [rows, max_feature_dim]; a push of width F uses the first rows * F
elements as a [rows, F] view (own and peer buffers alike). Because consecutive pushes reuse the same buffer, every
push starts with an extra barrier: a peer must have consumed (cloned) the previous push from its buffer before this
rank writes the next one into it. Without it a fast rank overwrites a slow peer's ghost rows of push k with push k+1.
"""
import torch
import torch.distributed as dist

from dynamis_model.comm.exchanger import Exchanger


HEADROOM = 1.25   # symmetric-buffer row headroom at (re-)allocation


class SymmMemExchanger(Exchanger):
    def __init__(self, max_feature_dim: int | None = None) -> None:
        super().__init__(max_feature_dim=max_feature_dim)
        self._ready = False

    @staticmethod
    def _is_single_rank() -> bool:
        # also without a process group (single-process runs skip init_process_group): identity
        return not dist.is_initialized() or dist.get_world_size() == 1

    def reset_buffers(self):
        self._ready = False
        self.is_ipc_setup = False

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
        meta = torch.cat([so, ro, torch.tensor([n_local], device=device, dtype=torch.int64)])
        allm = [torch.zeros_like(meta) for _ in range(W)]
        dist.all_gather(allm, meta)
        allm = torch.stack(allm).cpu()
        a_so, a_ro, a_nl = allm[:, : W + 1], allm[:, W + 1: 2 * W + 2], allm[:, -1]
        self.n_local = n_local
        self.n_full = n_local + int(a_ro[rank, -1])
        self.total_send = int(a_so[rank, -1])
        # forward buffer rows: [local | ghosts]; backward buffer rows: this rank's send list (owners receive grads)
        need_x = int(max(int(a_nl[r]) + int(a_ro[r, -1]) for r in range(W)))
        need_g = int(max(max(int(a_so[r, -1]) for r in range(W)), 1))
        # The symmetric buffers are kept across re-setups (halo rebuild / repartition) while they are large enough:
        # symm_mem.empty + rendezvous (IPC handle exchange + peer mappings) is the expensive part of a re-setup.
        # need_x / need_g are maxima over all ranks (all_gather above) -> every rank takes the same branch, so the
        # collective rendezvous stays matched. New allocations get HEADROOM.
        same = (getattr(self, "xbuf", None) is not None and self.F >= F and self.xbuf.device == device
                and self.xbuf.shape[0] >= need_x and self.gbuf.shape[0] >= need_g)
        if not same:
            self.xbuf = self.gbuf = self.xh = self.gh = None
            cap_x, cap_g = int(need_x * HEADROOM) + 8, int(need_g * HEADROOM) + 8
            self.xbuf = symm_mem.empty(cap_x, F, dtype=torch.float32, device=device)
            self.xh = symm_mem.rendezvous(self.xbuf, dist.group.WORLD)
            self.gbuf = symm_mem.empty(cap_g, F, dtype=torch.float32, device=device)
            self.gh = symm_mem.rendezvous(self.gbuf, dist.group.WORLD)
            self.n_alloc = getattr(self, "n_alloc", 0) + 1
        cap_x, cap_g = self.xbuf.shape[0], self.gbuf.shape[0]
        F = int(self.xbuf.shape[1])   # allocated width (>= every push width)
        sli = partition_info.send_local_indices.to(device=device, dtype=torch.int64)
        self.send_local_indices = sli
        self.fwd, self.bwd = [], []
        for r in range(W):
            if r == rank:
                continue
            n_s = int(a_so[rank, r + 1] - a_so[rank, r])   # rows I send to r
            if n_s:
                start = int(a_nl[r]) + int(a_ro[r, rank])     # where they land in r's ghost region
                self.fwd.append((sli[int(a_so[rank, r]): int(a_so[rank, r]) + n_s],
                                 torch.arange(start, start + n_s, device=device),
                                 self.xh.get_buffer(r, [cap_x, F], torch.float32).view(-1)))
            n_r = int(a_ro[rank, r + 1] - a_ro[rank, r])   # ghost rows I hold from r -> grads back to r
            if n_r:
                g0 = n_local + int(a_ro[rank, r])
                start = int(a_so[r, rank])                     # r's send-list rows for me
                self.bwd.append((g0, n_r, start, self.gh.get_buffer(r, [cap_g, F], torch.float32).view(-1)))
        self.F = F
        self.is_ipc_setup = True
        self._ready = True
        # new buffers / index tensors: a CUDA graph that captured the old ones must be re-captured (graph key)
        self.generation = getattr(self, "generation", 0) + 1

    def feature_push_impl(self, features_to_push, partition_info):
        x = features_to_push.contiguous()
        f = int(x.shape[1])
        xb = self.xbuf.view(-1)[: self.xbuf.shape[0] * f].view(-1, f)
        self.xh.barrier(channel=0)   # peers have consumed the previous push from this pooled buffer
        xb[: self.n_local].copy_(x)
        for src, dst, peer in self.fwd:
            peer[: self.xbuf.shape[0] * f].view(-1, f).index_copy_(0, dst, x.index_select(0, src))
        self.xh.barrier(channel=0)
        return xb[: self.n_full].clone()

    def feature_push_backward_impl(self, grad_output, input_feature_shape, partition_info):
        g = grad_output.contiguous()
        f = int(g.shape[1])
        self.gh.barrier(channel=0)   # owners have consumed the previous backward push
        for g0, n, start, peer in self.bwd:
            peer[: self.gbuf.shape[0] * f].view(-1, f)[start: start + n].copy_(g[g0: g0 + n])
        self.gh.barrier(channel=0)
        grad_in = g[: self.n_local].clone()
        if self.total_send > 0:
            gb = self.gbuf.view(-1)[: self.gbuf.shape[0] * f].view(-1, f)
            grad_in.index_add_(0, self.send_local_indices, gb[: self.total_send])
        return grad_in
