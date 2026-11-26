"""GPU tile packing for the fused UMA-s edge block (edge_block.py, capacity mode).

Whole target segments are packed into 128-edge tiles with <= MAX_SLOTS targets; a tile starts at the next
unassigned segment in graph order and is filled greedily with the largest segment that still fits from a look-ahead
window of `window` segments (ties -> the earliest); on the GPU and with NO host sync:

    perm, slot, tile_nodes, counts = pack_node_tiles_gpu(tgt, tile_cap, window=32)

* tgt: [E] int (edges grouped by target, e.g. radius_graph_pbc_cuda output)
* tile_cap: capacity in tiles (buffers are sized tile_cap * 128); tiles >= counts[0] are empty
  padding tiles (perm -1, slot 0, tile_nodes -1) -> with zero Dinv rows they contribute nothing.
* counts (device int32 [4]): [num_tiles, num_segments, overflow flag (tiles > tile_cap),
  bad-degree flag (a target with > 128 edges)]. The caller checks the flags lazily (one sync per
  neighbour-list rebuild at most) and re-packs at a larger capacity if needed.

Kernels (generated CUDA, common.nvcc_build + ctypes):
  seg_scatter   grid over edges: segment starts (tgt[i] != tgt[i-1]) -> compacted via a torch cumsum
  greedy_chunk  one warp per chunk of PACK_CHUNK consecutive segments runs the greedy (window candidates = lanes,
         max-length pick = __reduce_max_sync, segment lengths in shared memory)
  tiles_gather  concatenates the chunks' tiles in chunk order (result = a single greedy over the whole list, except
         that the look-ahead window does not cross chunk boundaries)
  fill          grid over tile_cap * 128 packed slots: perm / slot / tile_nodes
"""
import ctypes

import torch

from dynamis_model.nnp.uma_fused.common import nvcc_build

MAX_EDGES = 128
PACK_CHUNK = 1024   # segments per greedy chunk (one warp each; a single warp over all segments took ~2 ms at 16k atoms)

_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

// seg_start[k] for k < n (n = counts[1]); seg_start[n] = E
__global__ void seg_scatter(const int64_t* __restrict__ tgt, const int64_t* __restrict__ csum, int* __restrict__ seg_start,
                            int* __restrict__ counts, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;
  bool st = (i == 0) || (tgt[i] != tgt[i - 1]);
  if (st) seg_start[csum[i] - 1] = i;
  if (i == E - 1) { int n = (int)csum[i]; seg_start[n] = E; counts[1] = n; }
}

// packed slot p -> perm / slot; slot 0 of each tile also writes tile_nodes
__global__ void fill(const int* __restrict__ seg_start, const int* __restrict__ tile_seg, const int64_t* __restrict__ tgt,
                     const int* __restrict__ counts, int64_t* __restrict__ perm, int* __restrict__ slot,
                     int* __restrict__ tile_nodes, int tile_cap, int max_slots) {
  long p = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (long)tile_cap * 128) return;
  const int t = (int)(p >> 7);
  int off = (int)(p & 127);
  const int nt = min(counts[0], tile_cap);
  int64_t pr = -1; int sl = 0;
  if (t < nt) {
    for (int s = 0; s < max_slots; ++s) {
      const int sg = tile_seg[(long)t * max_slots + s];
      if (sg < 0) break;
      const int st = seg_start[sg], l = seg_start[sg + 1] - st;
      if (off < l) { pr = st + off; sl = s; break; }
      off -= l;
    }
  }
  perm[p] = pr; slot[p] = sl;
  if ((p & 127) < max_slots) {
    const int s = (int)(p & 127);
    int tn = -1;
    if (t < nt) { const int sg = tile_seg[(long)t * max_slots + s]; if (sg >= 0) tn = (int)tgt[seg_start[sg]]; }
    tile_nodes[(long)t * max_slots + s] = tn;
  }
}

// Chunked greedy. The segment list is cut into chunks of `chunk` consecutive segments; one warp
// per chunk runs the same greedy (look-ahead window inside the chunk) into a scratch tile list at tile offset lo
// (a chunk of m segments makes <= m tiles); tiles_gather concatenates the chunks' tiles in chunk order.
// Result = a single greedy over the whole list except at chunk boundaries (the window does not cross them).
// Tile start = the next unassigned segment; up to max_slots - 1 more segments are added, each time the largest one
// that still fits among the next `window` segments (ties -> the earliest).
__global__ void __launch_bounds__(32) greedy_chunk(const int* __restrict__ seg_start, int* __restrict__ tmp_seg,
        int* __restrict__ chunk_nt, int* __restrict__ counts, int window, int max_slots, int seg_cap, int chunk) {
  extern __shared__ uint8_t lens[];
  const int n = counts[1];
  const int c = blockIdx.x, lane = threadIdx.x;
  const int lo = c * chunk;
  if (n > seg_cap || lo >= n) { if (lane == 0) { chunk_nt[c] = 0; if (c == 0 && n > seg_cap) counts[3] = 2; } return; }
  const int m = min(n, lo + chunk) - lo;
  for (int k = lane; k < m; k += 32) {
    int l = seg_start[lo + k + 1] - seg_start[lo + k];
    if (l > 128) { counts[3] = 1; l = 128; }
    lens[k] = (uint8_t)l;
  }
  __syncwarp();
  int head = 0, nt = 0;
  unsigned long long um = 0;
  while (true) {
    while (head < m && (um & 1ull)) { um >>= 1; ++head; }
    if (head >= m) break;
    um |= 1ull;
    int room = 128 - lens[head];
    int myseg[8];
    myseg[0] = head;
    int ns = 1;
    const int k = 1 + lane;
    const int ln = (lane < window && head + k < m) ? (int)lens[head + k] : 0;
    while (ns < max_slots) {
      const bool ok = ln > 0 && ln <= room && !((um >> k) & 1ull);
      const unsigned key = ok ? ((unsigned)ln << 8) | (unsigned)(31 - lane) : 0u;
      const unsigned best = __reduce_max_sync(0xffffffffu, key);
      if (best == 0u) break;
      const int bk = 1 + (31 - (int)(best & 0xffu));
      um |= 1ull << bk;
      room -= (int)(best >> 8);
      myseg[ns++] = head + bk;
    }
    if (lane < max_slots) tmp_seg[(long)(lo + nt) * max_slots + lane] = lane < ns ? lo + myseg[lane] : -1;
    ++nt;
  }
  if (lane == 0) chunk_nt[c] = nt;
}

__global__ void tiles_gather(const int* __restrict__ tmp_seg, const int* __restrict__ chunk_nt, int* __restrict__ tile_seg,
                             int* __restrict__ counts, int tile_cap, int max_slots, int chunk, int nchunks) {
  const int c = blockIdx.x;
  int off = 0;
  for (int j = 0; j < c; ++j) off += chunk_nt[j];
  const int nt = chunk_nt[c];
  const long lo = (long)c * chunk;
  for (int i = threadIdx.x; i < nt * max_slots; i += blockDim.x) {
    const int t = off + i / max_slots;
    if (t < tile_cap) tile_seg[(long)t * max_slots + i % max_slots] = tmp_seg[lo * max_slots + i];
  }
  if (c == nchunks - 1 && threadIdx.x == 0) {
    const int tot = off + nt;
    counts[0] = counts[3] == 2 ? 0 : tot;
    counts[2] = tot > tile_cap ? 1 : 0;
  }
}

extern "C" void pack_launch_chunked(const int64_t* tgt, const int64_t* csum, int* seg_start, int* tile_seg, int* counts,
                            int64_t* perm, int* slot, int* tile_nodes, int* tmp_seg, int* chunk_nt, int E, int tile_cap,
                            int window, int max_slots, int seg_cap, int chunk, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  if (E > 0) seg_scatter<<<(E + 255) / 256, 256, 0, s>>>(tgt, csum, seg_start, counts, E);
  const int nchunks = (seg_cap + chunk - 1) / chunk;
  greedy_chunk<<<nchunks, 32, chunk, s>>>(seg_start, tmp_seg, chunk_nt, counts, window, max_slots, seg_cap, chunk);
  tiles_gather<<<nchunks, 256, 0, s>>>(tmp_seg, chunk_nt, tile_seg, counts, tile_cap, max_slots, chunk, nchunks);
  long np = (long)tile_cap * 128;
  fill<<<(unsigned)((np + 255) / 256), 256, 0, s>>>(seg_start, tile_seg, tgt, counts, perm, slot, tile_nodes,
                                                      tile_cap, max_slots);
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = nvcc_build(_SRC, "uma_gpu_pack")
        lib.pack_launch_chunked.restype = None
        _LIB = lib
    return _LIB


def pack_node_tiles_gpu(tgt, tile_cap, window=32, max_slots=7, seg_cap=None):
    """-> perm [tile_cap*128] int64 (-1 = padding), slot [tile_cap*128] int32, tile_nodes [tile_cap, max_slots]
    int32, counts [4] int32 (device: num_tiles, num_segments, overflow, bad degree). No host sync."""
    assert 0 <= window <= 32 and max_slots <= 8
    dev = tgt.device
    tgt = tgt.to(torch.int64).contiguous()
    E = tgt.numel()
    # segments <= distinct targets (<= N for a target-grouped list); more -> counts[3] = 2 (error flag)
    seg_cap = E if seg_cap is None else seg_cap
    st = torch.ones(E, dtype=torch.int64, device=dev)
    if E > 1:
        st[1:] = tgt[1:] != tgt[:-1]
    csum = torch.cumsum(st, 0)
    seg_start = torch.empty(E + 1, dtype=torch.int32, device=dev)
    tile_seg = torch.empty(tile_cap, max_slots, dtype=torch.int32, device=dev)
    counts = torch.zeros(4, dtype=torch.int32, device=dev)
    perm = torch.empty(tile_cap * MAX_EDGES, dtype=torch.int64, device=dev)
    slot = torch.empty(tile_cap * MAX_EDGES, dtype=torch.int32, device=dev)
    tnodes = torch.empty(tile_cap, max_slots, dtype=torch.int32, device=dev)
    nch = -(-max(seg_cap, 1) // PACK_CHUNK)
    tmp = torch.empty(nch * PACK_CHUNK, max_slots, dtype=torch.int32, device=dev)
    cnt = torch.empty(nch, dtype=torch.int32, device=dev)
    ts = [tgt, csum, seg_start, tile_seg, counts, perm, slot, tnodes, tmp, cnt]
    _lib().pack_launch_chunked(*[ctypes.c_void_p(t.data_ptr()) for t in ts], ctypes.c_int(E), ctypes.c_int(tile_cap),
                               ctypes.c_int(window), ctypes.c_int(max_slots), ctypes.c_int(max(seg_cap, 1)),
                               ctypes.c_int(PACK_CHUNK), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    return perm, slot, tnodes, counts


def build_csr_nosync(src, tgt, valid, num_nodes):
    """per-node CSR of the packed edges (stable sort by node: S3's fixed per-node reduction order) without host syncs:
    eidx has the capacity length (padding entries sorted to the end, beyond ptr[num_nodes])."""
    out = []
    for nodes in (src, tgt):
        key = torch.where(valid, nodes.long(), num_nodes)
        order = torch.argsort(key, stable=True)
        cnt = torch.zeros(num_nodes + 1, dtype=torch.int64, device=src.device)
        cnt.index_add_(0, key, torch.ones_like(key))
        ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=src.device)
        ptr[1:] = torch.cumsum(cnt[:num_nodes], 0)
        out += [ptr.to(torch.int32).contiguous(), order.to(torch.int32).contiguous()]
    return out


class GPUTopology:
    """Packed topology at fixed capacity (tiles, nodes), rebuilt IN PLACE on the GPU without host sync, so
    every buffer keeps its address (CUDA graphs captured on it stay valid across neighbour-list rebuilds).
    Keys: pc, padm, padz, slot, tile_nodes, perm, src, tgt, csr, orig,
    plus counts (device [num_tiles, num_segments, overflow, bad degree])."""

    def __init__(self, tile_cap, node_cap, device, window=32):
        self.tile_cap, self.node_cap, self.window = tile_cap, node_cap, window
        self.device = torch.device(device)
        self.d = None

    def update(self, edge_index):
        perm, slot, tn, counts = pack_node_tiles_gpu(edge_index[1], self.tile_cap, window=self.window,
                                                     seg_cap=self.node_cap)
        padm = perm >= 0
        pc = perm.clamp_min(0)
        src = torch.index_select(edge_index[0], 0, pc).masked_fill_(~padm, 0).to(torch.int32)
        tgt = torch.index_select(edge_index[1], 0, pc).masked_fill_(~padm, 0).to(torch.int32)
        csr = build_csr_nosync(src, tgt, padm, self.node_cap)
        new = dict(perm=perm, slot=slot, tile_nodes=tn, counts=counts, padm=padm, padz=~padm, pc=pc,
                   src=src, tgt=tgt, csr=csr, orig=perm.to(torch.int32))
        if self.d is None:
            self.d = new
        else:   # copy into the existing buffers (stable addresses)
            for k, v in new.items():
                if k == "csr":
                    for a, b in zip(self.d["csr"], v):
                        a.copy_(b)
                else:
                    self.d[k].copy_(v)
        return self.d

    def check(self):
        """one host sync: raises if the capacity was exceeded (call once per rebuild, or lazily)"""
        c = self.d["counts"].tolist()
        if c[3]:
            raise ValueError("a target degree exceeds the tile size 128" if c[3] == 1 else
                             "edges not grouped by target / more target segments than the node capacity")
        if c[2]:
            raise OverflowError(f"tile capacity {self.tile_cap} exceeded ({c[0]} tiles needed)")
        return c
