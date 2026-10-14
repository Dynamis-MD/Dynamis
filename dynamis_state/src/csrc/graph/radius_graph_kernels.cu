// radius_graph_kernels.cu — CUDA cell-list radius graph with selective PBC tiling
//
// Stage A: Selective PBC replication (only boundary atoms near cell faces)
// Stage B: Cell-list construction + two-pass neighbor search
// Stage C: Max-neighbors mask (per-atom top-K by distance)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <tuple>

static constexpr int BLOCK_SIZE = 256;

// ===== Stage A: Selective PBC Tiling =====

// Kernel A1: Count how many PBC images each atom needs
__global__ void count_pbc_images_kernel(
    const float* __restrict__ pos,      // [N, 3]
    const float* __restrict__ cell_inv,  // [3, 3] row-major
    const float* __restrict__ cutoff_frac, // [3] cutoff in fractional units
    const bool* __restrict__ pbc,        // [3]
    const int* __restrict__ rep,         // [3] replication counts per direction
    int N,
    int* __restrict__ image_counts       // [N] output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx * 3 + 0];
    float py = pos[idx * 3 + 1];
    float pz = pos[idx * 3 + 2];

    // Convert to fractional coordinates: s = pos @ cell_inv^T
    // cell_inv is [3,3] row-major: cell_inv[row][col] = cell_inv[row*3+col]
    float sx = px * cell_inv[0] + py * cell_inv[3] + pz * cell_inv[6];
    float sy = px * cell_inv[1] + py * cell_inv[4] + pz * cell_inv[7];
    float sz = px * cell_inv[2] + py * cell_inv[5] + pz * cell_inv[8];

    float s[3] = {sx, sy, sz};
    float cf[3] = {cutoff_frac[0], cutoff_frac[1], cutoff_frac[2]};
    bool pbc_d[3] = {pbc[0], pbc[1], pbc[2]};
    int rep_d[3] = {rep[0], rep[1], rep[2]};

    int count = 0;
    for (int dx = -rep_d[0]; dx <= rep_d[0]; dx++) {
        for (int dy = -rep_d[1]; dy <= rep_d[1]; dy++) {
            for (int dz = -rep_d[2]; dz <= rep_d[2]; dz++) {
                int dd[3] = {dx, dy, dz};
                bool needed = true;
                for (int d = 0; d < 3; d++) {
                    if (!pbc_d[d] && dd[d] != 0) { needed = false; break; }
                    if (dd[d] > 0) {
                        // Image shifts source UP → source must be near LOWER face
                        if (s[d] >= cf[d]) { needed = false; break; }
                    } else if (dd[d] < 0) {
                        // Image shifts source DOWN → source must be near UPPER face
                        if (s[d] <= 1.0f - cf[d]) { needed = false; break; }
                    }
                    // dd[d] == 0 → always true for this direction
                }
                if (needed) count++;
            }
        }
    }
    image_counts[idx] = count;
}

// Kernel A2: Write PBC image positions, original indices, and cell shifts
__global__ void write_pbc_images_kernel(
    const float* __restrict__ pos,           // [N, 3]
    const float* __restrict__ cell,          // [3, 3] row-major (cell vectors as rows)
    const float* __restrict__ cell_inv,      // [3, 3] row-major
    const float* __restrict__ cutoff_frac,   // [3]
    const bool* __restrict__ pbc,            // [3]
    const int* __restrict__ rep,             // [3]
    const int64_t* __restrict__ image_offsets, // [N] exclusive prefix sum
    int N,
    float* __restrict__ source_pos,          // [total_sources, 3]
    int64_t* __restrict__ source_orig_idx,   // [total_sources]
    float* __restrict__ source_cell_shift    // [total_sources, 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx * 3 + 0];
    float py = pos[idx * 3 + 1];
    float pz = pos[idx * 3 + 2];

    float sx = px * cell_inv[0] + py * cell_inv[3] + pz * cell_inv[6];
    float sy = px * cell_inv[1] + py * cell_inv[4] + pz * cell_inv[7];
    float sz = px * cell_inv[2] + py * cell_inv[5] + pz * cell_inv[8];

    float s[3] = {sx, sy, sz};
    float cf[3] = {cutoff_frac[0], cutoff_frac[1], cutoff_frac[2]};
    bool pbc_d[3] = {pbc[0], pbc[1], pbc[2]};
    int rep_d[3] = {rep[0], rep[1], rep[2]};

    int64_t offset = image_offsets[idx];
    int write_idx = 0;

    for (int dx = -rep_d[0]; dx <= rep_d[0]; dx++) {
        for (int dy = -rep_d[1]; dy <= rep_d[1]; dy++) {
            for (int dz = -rep_d[2]; dz <= rep_d[2]; dz++) {
                int dd[3] = {dx, dy, dz};
                bool needed = true;
                for (int d = 0; d < 3; d++) {
                    if (!pbc_d[d] && dd[d] != 0) { needed = false; break; }
                    if (dd[d] > 0) {
                        // Image shifts source UP → source must be near LOWER face
                        if (s[d] >= cf[d]) { needed = false; break; }
                    } else if (dd[d] < 0) {
                        // Image shifts source DOWN → source must be near UPPER face
                        if (s[d] <= 1.0f - cf[d]) { needed = false; break; }
                    }
                }
                if (needed) {
                    int64_t out = offset + write_idx;
                    // new_pos = pos + cell^T @ [dx, dy, dz]
                    // cell is row-major: cell[row][col] = cell[row*3+col]
                    // cell^T @ v = sum_row v[row] * cell[row, :]
                    float shift_x = (float)dx * cell[0] + (float)dy * cell[3] + (float)dz * cell[6];
                    float shift_y = (float)dx * cell[1] + (float)dy * cell[4] + (float)dz * cell[7];
                    float shift_z = (float)dx * cell[2] + (float)dy * cell[5] + (float)dz * cell[8];

                    source_pos[out * 3 + 0] = px + shift_x;
                    source_pos[out * 3 + 1] = py + shift_y;
                    source_pos[out * 3 + 2] = pz + shift_z;

                    source_orig_idx[out] = (int64_t)idx;

                    source_cell_shift[out * 3 + 0] = (float)dx;
                    source_cell_shift[out * 3 + 1] = (float)dy;
                    source_cell_shift[out * 3 + 2] = (float)dz;

                    write_idx++;
                }
            }
        }
    }
}

// ===== Stage B: Cell-List Neighbor Search =====

// Kernel B1: Assign each source atom to a grid cell
__global__ void assign_cells_kernel(
    const float* __restrict__ source_pos,  // [M, 3]
    float grid_res_inv,                     // 1.0 / grid_resolution
    float grid_min_x, float grid_min_y, float grid_min_z,
    int grid_nx, int grid_ny, int grid_nz,
    int M,
    int* __restrict__ cell_ids             // [M]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    float x = source_pos[idx * 3 + 0];
    float y = source_pos[idx * 3 + 1];
    float z = source_pos[idx * 3 + 2];

    int cx = (int)floorf((x - grid_min_x) * grid_res_inv);
    int cy = (int)floorf((y - grid_min_y) * grid_res_inv);
    int cz = (int)floorf((z - grid_min_z) * grid_res_inv);

    // Clamp to valid range
    cx = max(0, min(cx, grid_nx - 1));
    cy = max(0, min(cy, grid_ny - 1));
    cz = max(0, min(cz, grid_nz - 1));

    cell_ids[idx] = cx + cy * grid_nx + cz * grid_nx * grid_ny;
}

// Kernel B2: Build cell_start/cell_end ranges from sorted cell_ids
__global__ void build_cell_ranges_kernel(
    const int* __restrict__ sorted_cell_ids, // [M]
    int M,
    int num_cells,
    int* __restrict__ cell_start,            // [num_cells]
    int* __restrict__ cell_end               // [num_cells]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    int cid = sorted_cell_ids[idx];

    // Start of a new segment
    if (idx == 0 || sorted_cell_ids[idx - 1] != cid) {
        cell_start[cid] = idx;
    }
    // End of a segment
    if (idx == M - 1 || sorted_cell_ids[idx + 1] != cid) {
        cell_end[cid] = idx + 1;
    }
}

// Kernel B3: Count neighbors for each target atom
__global__ void count_neighbors_kernel(
    const float* __restrict__ target_pos,   // [N, 3]
    const float* __restrict__ source_pos,   // [M, 3] (sorted by cell)
    const int* __restrict__ target_cell_x,  // [N]
    const int* __restrict__ target_cell_y,  // [N]
    const int* __restrict__ target_cell_z,  // [N]
    const int64_t* __restrict__ source_orig_idx, // [M]
    const int* __restrict__ cell_start,     // [num_cells]
    const int* __restrict__ cell_end,       // [num_cells]
    int grid_nx, int grid_ny, int grid_nz,
    int padding_size,
    float radius_sq,
    int N,
    int* __restrict__ neighbor_counts       // [N]
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    float tx = target_pos[tidx * 3 + 0];
    float ty = target_pos[tidx * 3 + 1];
    float tz = target_pos[tidx * 3 + 2];

    int tcx = target_cell_x[tidx];
    int tcy = target_cell_y[tidx];
    int tcz = target_cell_z[tidx];

    int count = 0;

    for (int dz = -padding_size; dz <= padding_size; dz++) {
        int nz = tcz + dz;
        if (nz < 0 || nz >= grid_nz) continue;
        for (int dy = -padding_size; dy <= padding_size; dy++) {
            int ny = tcy + dy;
            if (ny < 0 || ny >= grid_ny) continue;
            for (int dx = -padding_size; dx <= padding_size; dx++) {
                int nx_val = tcx + dx;
                if (nx_val < 0 || nx_val >= grid_nx) continue;

                int cell_id = nx_val + ny * grid_nx + nz * grid_nx * grid_ny;
                int start = cell_start[cell_id];
                int end = cell_end[cell_id];

                for (int s = start; s < end; s++) {
                    float sx = source_pos[s * 3 + 0];
                    float sy = source_pos[s * 3 + 1];
                    float sz = source_pos[s * 3 + 2];

                    float dx2 = tx - sx;
                    float dy2 = ty - sy;
                    float dz2 = tz - sz;
                    float dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

                    // Check within radius and not self (same original atom at image (0,0,0))
                    if (dist_sq <= radius_sq && dist_sq > 1e-8f) {
                        count++;
                    }
                }
            }
        }
    }
    neighbor_counts[tidx] = count;
}

// ===== Stage B4: warp-per-atom neighbor-write kernels =====
//
// One templated kernel write_neighbors_kernel<USE_SHM> (defined below the constants):
// warp-per-atom scan, with nosort (no shm) vs collect (shm + coalesced) write strategies
// chosen by the B4 dispatch in radius_graph_pbc_cuda_impl_ex. Neighbor counting (Stage B3)
// stays one-thread-per-atom (count_neighbors_kernel above) — see its dispatch comment.

// nosort path: 8 warps/block (256 threads), no shared memory → full occupancy.
static constexpr int WARPS_PER_BLOCK_WARP = 8;   // BLOCK_SIZE == 256 == 8*32

// collect path: 4 warps/block (128 threads) + per-warp shm scratch (dist + src).
// shm per block = WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT * (4B + 4B)
//   = 4 * 256 * 8 = 8 KB → 12 blocks/SM → 48/64 warps = 75% occupancy (A100).
// Dispatch guard max_nbrs_pre < MAX_NEIGHBORS_BUF_COLLECT ensures the buffer never overflows.
static constexpr int MAX_NEIGHBORS_BUF_COLLECT = 256;
static constexpr int WARPS_PER_BLOCK_COLLECT   = 4;
static constexpr int BLOCK_SIZE_COLLECT        = 32 * WARPS_PER_BLOCK_COLLECT;  // 128


// Shared edge-write helper: emit one (src → tgt) edge, applying the PBC cell-shift
// correction for the target/source wrap offsets (frac_floor). Both write paths go
// through here so the edge layout lives in exactly one place.
__device__ __forceinline__ void emit_edge(
    int64_t out, int s, int64_t tgt_orig, float d2,
    float tfx, float tfy, float tfz,
    const int64_t* __restrict__ source_orig_idx,
    const float*   __restrict__ source_cell_shift,
    const float*   __restrict__ frac_floor,
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,
    float*   __restrict__ edge_dist_sq)
{
    int64_t src_orig = source_orig_idx[s];
    float sfx = frac_floor[src_orig * 3 + 0];
    float sfy = frac_floor[src_orig * 3 + 1];
    float sfz = frac_floor[src_orig * 3 + 2];
    edge_src[out]     = src_orig;
    edge_tgt[out]     = tgt_orig;
    edge_dist_sq[out] = d2;
    edge_cell_shift_out[out * 3 + 0] = source_cell_shift[s * 3 + 0] + tfx - sfx;
    edge_cell_shift_out[out * 3 + 1] = source_cell_shift[s * 3 + 1] + tfy - sfy;
    edge_cell_shift_out[out * 3 + 2] = source_cell_shift[s * 3 + 2] + tfz - sfz;
}

// Kernel B4: warp-per-atom neighbor write (max_num_neighbors == 0).
//
// One warp (32 threads) handles one target atom, striding lanes through each of the
// (2*padding+1)^3 neighbor cells (s = cell_start + it*32 + lane), so active lanes load
// contiguous source atoms. The Phase-A scan (cell loop + distance test + ballot) is
// identical for both strategies; the USE_SHM template parameter selects how the found
// neighbors are emitted — the compiler specializes one kernel per value, no runtime cost:
//
//   USE_SHM == false (nosort) : write inline during the scan via ballot+popc scatter.
//     No shared memory → full occupancy. Best for dense systems (>=5 atoms/cell).
//   USE_SHM == true  (collect): stash (dist, src) into a per-warp shm buffer, then in
//     Phase C write stride-32 (all 32 lanes active → fully coalesced). 8KB shm → 75%
//     occupancy. Best for sparse systems (<5 atoms/cell, write-bound).
//
// The collect dispatch guarantees degree < MAX_NEIGHBORS_BUF_COLLECT (buffer never overflows).
template<bool USE_SHM>
__global__ void __launch_bounds__(USE_SHM ? BLOCK_SIZE_COLLECT : BLOCK_SIZE)
write_neighbors_kernel(
    const float*   __restrict__ target_pos,
    const float*   __restrict__ source_pos,
    const int*     __restrict__ target_cell_x,
    const int*     __restrict__ target_cell_y,
    const int*     __restrict__ target_cell_z,
    const int64_t* __restrict__ source_orig_idx,
    const float*   __restrict__ source_cell_shift,
    const int*     __restrict__ cell_start,
    const int*     __restrict__ cell_end,
    int grid_nx, int grid_ny, int grid_nz,
    int padding_size,
    float radius_sq,
    int N,
    const int*     __restrict__ neighbor_counts,    // collect Phase-C count (== degree); unused when !USE_SHM
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ target_sort_perm,   // sorted_idx → original atom id
    const float*   __restrict__ frac_floor,         // [N_orig, 3] PBC wrap correction
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,      // [E, 3]
    float*   __restrict__ edge_dist_sq
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= N) return;

    // collect: per-warp shared-memory scratch. nosort launches with 0 dynamic shm and
    // never touches this (all accesses guarded by `if constexpr (USE_SHM)`).
    extern __shared__ float smem_c[];
    float* s_dist = nullptr;
    int*   s_src  = nullptr;
    if constexpr (USE_SHM) {
        int* smem_c_int = (int*)(smem_c + WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT);
        int warp_local = threadIdx.x >> 5;
        s_dist = smem_c     + warp_local * MAX_NEIGHBORS_BUF_COLLECT;
        s_src  = smem_c_int + warp_local * MAX_NEIGHBORS_BUF_COLLECT;
    }

    int64_t tgt_orig = target_sort_perm[warp_id];
    float tx = target_pos[warp_id * 3 + 0];
    float ty = target_pos[warp_id * 3 + 1];
    float tz = target_pos[warp_id * 3 + 2];
    int   tcx = target_cell_x[warp_id];
    int   tcy = target_cell_y[warp_id];
    int   tcz = target_cell_z[warp_id];
    float tfx = frac_floor[tgt_orig * 3 + 0];
    float tfy = frac_floor[tgt_orig * 3 + 1];
    float tfz = frac_floor[tgt_orig * 3 + 2];

    int64_t base     = neighbor_offsets[warp_id];
    int     produced = 0;   // running count of neighbors found by this warp so far

    // Phase A: scan the (2*padding+1)^3 neighbor cells.
    for (int dz = -padding_size; dz <= padding_size; dz++) {
        int nz = tcz + dz;
        if (nz < 0 || nz >= grid_nz) continue;
        for (int dy = -padding_size; dy <= padding_size; dy++) {
            int ny = tcy + dy;
            if (ny < 0 || ny >= grid_ny) continue;
            for (int dx = -padding_size; dx <= padding_size; dx++) {
                int nx_val = tcx + dx;
                if (nx_val < 0 || nx_val >= grid_nx) continue;

                int cell_id = nx_val + ny * grid_nx + nz * grid_nx * grid_ny;
                int cs = cell_start[cell_id];
                int ce = cell_end[cell_id];
                int iters = (ce - cs + 31) >> 5;

                for (int it = 0; it < iters; it++) {
                    int s = cs + (it << 5) + lane;
                    float d2 = 0.0f;
                    bool is_nbr = false;
                    if (s < ce) {
                        float sx = source_pos[s * 3 + 0];
                        float sy = source_pos[s * 3 + 1];
                        float sz = source_pos[s * 3 + 2];
                        d2 = (tx-sx)*(tx-sx) + (ty-sy)*(ty-sy) + (tz-sz)*(tz-sz);
                        is_nbr = (d2 <= radius_sq) && (d2 > 1e-8f);
                    }
                    uint32_t mask = __ballot_sync(0xFFFFFFFFu, is_nbr);
                    if (is_nbr) {
                        int local = __popc(mask & ((1u << lane) - 1));
                        if constexpr (USE_SHM) {
                            int store_pos = produced + local;
                            // dispatch guarantees produced + local < MAX_NEIGHBORS_BUF_COLLECT
                            if (store_pos < MAX_NEIGHBORS_BUF_COLLECT) {
                                s_dist[store_pos] = d2;
                                s_src[store_pos]  = s;
                            }
                        } else {
                            emit_edge(base + produced + local, s, tgt_orig, d2,
                                      tfx, tfy, tfz, source_orig_idx, source_cell_shift,
                                      frac_floor, edge_src, edge_tgt,
                                      edge_cell_shift_out, edge_dist_sq);
                        }
                    }
                    produced += __popc(mask);
                }
            }
        }
    }

    if constexpr (USE_SHM) {
        // Order Phase-A shm stores (each slot written by whichever lane owned it) before
        // Phase C reads them across lanes (lane L reads slot L+32k, written by another
        // lane). Without this __syncwarp(), Volta+ independent thread scheduling lets a
        // lane read a stale slot left by the previous block's warp on this SM →
        // a byte-exact duplicate edge.
        __syncwarp();

        // Phase C: stride-32 write → all 32 lanes active per iteration → fully coalesced.
        int out_count = neighbor_counts[warp_id];  // == produced (B3 and Phase A agree)
        for (int i = lane; i < out_count; i += 32) {
            emit_edge(base + i, s_src[i], tgt_orig, s_dist[i],
                      tfx, tfy, tfz, source_orig_idx, source_cell_shift,
                      frac_floor, edge_src, edge_tgt,
                      edge_cell_shift_out, edge_dist_sq);
        }
    }
}

// ===== Per-stage profiling helper (enabled via RADIUS_PBC_PROFILE=1) =====
static bool s_profile = (std::getenv("RADIUS_PBC_PROFILE") != nullptr);
struct StageClock {
    cudaEvent_t start, stop;
    const char* name;
    bool active;
    StageClock(const char* n, cudaStream_t stream) : name(n), active(s_profile) {
        if (!active) return;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    float lap(cudaStream_t stream) {
        if (!active) return 0;
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop);
        printf("  %-28s: %6.2f ms\n", name, ms);
        cudaEventRecord(start, stream);   // reset for next stage
        return ms;
    }
    ~StageClock() { if (active) { cudaEventDestroy(start); cudaEventDestroy(stop); } }
};

// ===== Dispatch Function =====

// Extended internal impl. If `target_indices` is provided (non-null), only atoms
// at those indices are used as targets (sources still come from full `pos` + PBC).
// Output edge endpoints use global atom indices regardless.
//
// Returns:
//   edge_index[2, E]            COO, global atom ids
//   cell_shifts[E, 3]           PBC image shifts per edge
//   num_neighbors_image[1]      int64, total edges
//   neighbor_counts[N_target]   int32, per-target degree (in target_indices order if provided,
//                               else atom-id order)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl_ex(
    at::Tensor pos,       // [N, 3] float32 CUDA
    at::Tensor cell,      // [3, 3] float32
    at::Tensor pbc,       // [3] bool
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly,
    c10::optional<at::Tensor> target_indices,  // [N_target] int64, nullopt = all atoms
    bool counts_only
) {
    at::cuda::CUDAGuard device_guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = pos.size(0);
    const int N_target = target_indices.has_value()
                             ? (int)target_indices->size(0)
                             : N;
    StageClock clk("total", stream.stream());

    // Handle empty input
    if (N == 0 || N_target == 0) {
        auto opts_long = at::TensorOptions().dtype(at::kLong).device(pos.device());
        auto opts_float = at::TensorOptions().dtype(at::kFloat).device(pos.device());
        auto opts_int = at::TensorOptions().dtype(at::kInt).device(pos.device());
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long),
            at::zeros({N_target}, opts_int)
        );
    }

    auto opts_float = at::TensorOptions().dtype(at::kFloat).device(pos.device());
    auto opts_int = at::TensorOptions().dtype(at::kInt).device(pos.device());
    auto opts_long = at::TensorOptions().dtype(at::kLong).device(pos.device());

    // ===== Compute cell geometry on CPU for small tensors =====
    auto cell_cpu = cell.to(at::kCPU).to(at::kFloat);
    auto pbc_cpu = pbc.to(at::kCPU).to(at::kBool);
    auto cell_acc = cell_cpu.accessor<float, 2>();
    auto pbc_acc = pbc_cpu.accessor<bool, 1>();

    // cell vectors: a0 = cell[0,:], a1 = cell[1,:], a2 = cell[2,:]
    // Cross products for inter-plane distances
    float a0[3] = {cell_acc[0][0], cell_acc[0][1], cell_acc[0][2]};
    float a1[3] = {cell_acc[1][0], cell_acc[1][1], cell_acc[1][2]};
    float a2[3] = {cell_acc[2][0], cell_acc[2][1], cell_acc[2][2]};

    // cross(a1, a2)
    float c12[3] = {
        a1[1]*a2[2] - a1[2]*a2[1],
        a1[2]*a2[0] - a1[0]*a2[2],
        a1[0]*a2[1] - a1[1]*a2[0]
    };
    float vol = a0[0]*c12[0] + a0[1]*c12[1] + a0[2]*c12[2];
    float abs_vol = std::abs(vol);

    // cross(a2, a0)
    float c20[3] = {
        a2[1]*a0[2] - a2[2]*a0[1],
        a2[2]*a0[0] - a2[0]*a0[2],
        a2[0]*a0[1] - a2[1]*a0[0]
    };
    // cross(a0, a1)
    float c01[3] = {
        a0[1]*a1[2] - a0[2]*a1[1],
        a0[2]*a1[0] - a0[0]*a1[2],
        a0[0]*a1[1] - a0[1]*a1[0]
    };

    // Inter-plane distances: d_face[d] = |vol| / |cross_d|
    float norm_c12 = std::sqrt(c12[0]*c12[0] + c12[1]*c12[1] + c12[2]*c12[2]);
    float norm_c20 = std::sqrt(c20[0]*c20[0] + c20[1]*c20[1] + c20[2]*c20[2]);
    float norm_c01 = std::sqrt(c01[0]*c01[0] + c01[1]*c01[1] + c01[2]*c01[2]);

    float d_face[3] = {
        abs_vol / norm_c12,
        abs_vol / norm_c20,
        abs_vol / norm_c01
    };

    // Replication counts and cutoff fractions
    int rep_h[3];
    float cutoff_frac_h[3];
    for (int d = 0; d < 3; d++) {
        if (pbc_acc[d]) {
            rep_h[d] = (int)std::ceil(radius / d_face[d]);
            cutoff_frac_h[d] = radius / d_face[d];
        } else {
            rep_h[d] = 0;
            cutoff_frac_h[d] = 0.0f;
        }
    }

    // Compute cell_inv = inverse(cell) on CPU
    // Using the formula: inv(A) = adj(A) / det(A)
    // adj(A) is the transpose of the cofactor matrix
    float inv_vol = 1.0f / vol;
    float cell_inv_h[9];
    // Cofactor matrix transposed / det
    cell_inv_h[0] = (a1[1]*a2[2] - a1[2]*a2[1]) * inv_vol;
    cell_inv_h[1] = (a0[2]*a2[1] - a0[1]*a2[2]) * inv_vol;
    cell_inv_h[2] = (a0[1]*a1[2] - a0[2]*a1[1]) * inv_vol;
    cell_inv_h[3] = (a1[2]*a2[0] - a1[0]*a2[2]) * inv_vol;
    cell_inv_h[4] = (a0[0]*a2[2] - a0[2]*a2[0]) * inv_vol;
    cell_inv_h[5] = (a0[2]*a1[0] - a0[0]*a1[2]) * inv_vol;
    cell_inv_h[6] = (a1[0]*a2[1] - a1[1]*a2[0]) * inv_vol;
    cell_inv_h[7] = (a0[1]*a2[0] - a0[0]*a2[1]) * inv_vol;
    cell_inv_h[8] = (a0[0]*a1[1] - a0[1]*a1[0]) * inv_vol;

    // Upload small arrays to GPU
    auto cell_inv_gpu = at::from_blob(cell_inv_h, {9}, at::kFloat).to(pos.device());
    auto cutoff_frac_gpu = at::from_blob(cutoff_frac_h, {3}, at::kFloat).to(pos.device());
    auto pbc_gpu = pbc.to(pos.device()).to(at::kBool);
    int rep_h_arr[3] = {rep_h[0], rep_h[1], rep_h[2]};
    auto rep_gpu = at::from_blob(rep_h_arr, {3}, at::kInt).to(pos.device());

    // ===== Wrap positions to [0, L) =====
    // Positions may be centered around 0 (e.g. [-L/2, L/2]) or otherwise outside [0, L).
    // Compute fractional coords, floor, and wrap positions so the selective PBC
    // tiling logic (which assumes s ∈ [0, 1)) works correctly.
    // frac = pos @ cell_inv^T  (ATen matmul on GPU)
    auto cell_inv_mat = at::from_blob(cell_inv_h, {3, 3}, at::kFloat).to(pos.device());
    auto frac_coords = at::mm(pos, cell_inv_mat);  // [N, 3]
    auto frac_floor = at::floor(frac_coords);        // [N, 3] — integer cell offsets
    // pos_wrapped = pos - frac_floor @ cell  (move all atoms into [0, L))
    auto pos_wrapped = pos - at::mm(frac_floor, cell);  // [N, 3]

    // ===== Stage A: Selective PBC Tiling =====
    int grid_size_A = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // A1: Count images per atom (using wrapped positions)
    auto image_counts = at::empty({N}, opts_int);
    count_pbc_images_kernel<<<grid_size_A, BLOCK_SIZE, 0, stream>>>(
        pos_wrapped.contiguous().data_ptr<float>(),
        cell_inv_gpu.data_ptr<float>(),
        cutoff_frac_gpu.data_ptr<float>(),
        pbc_gpu.data_ptr<bool>(),
        rep_gpu.data_ptr<int>(),
        N,
        image_counts.data_ptr<int>()
    );

    // Prefix sum → image_offsets
    auto image_counts_long = image_counts.to(at::kLong);
    auto image_offsets = at::cumsum(image_counts_long, 0) - image_counts_long;
    int64_t total_sources = (image_offsets[N-1] + image_counts_long[N-1]).item<int64_t>();

    // Handle case where total_sources == 0 (shouldn't happen since (0,0,0) is always included)
    if (total_sources == 0) {
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long),
            at::zeros({N_target}, opts_int)
        );
    }

    // A2: Write images
    auto source_pos_t = at::empty({total_sources, 3}, opts_float);
    auto source_orig_idx = at::empty({total_sources}, opts_long);
    auto source_cell_shift = at::empty({total_sources, 3}, opts_float);

    write_pbc_images_kernel<<<grid_size_A, BLOCK_SIZE, 0, stream>>>(
        pos_wrapped.contiguous().data_ptr<float>(),
        cell.data_ptr<float>(),
        cell_inv_gpu.data_ptr<float>(),
        cutoff_frac_gpu.data_ptr<float>(),
        pbc_gpu.data_ptr<bool>(),
        rep_gpu.data_ptr<int>(),
        image_offsets.data_ptr<int64_t>(),
        N,
        source_pos_t.data_ptr<float>(),
        source_orig_idx.data_ptr<int64_t>(),
        source_cell_shift.data_ptr<float>()
    );

    int M = (int)total_sources; // total source atoms
    if (s_profile) printf("  M (total source atoms)       : %d\n", M);
    clk.lap(stream.stream());  // A: wrap + PBC images

    // ===== Stage B: Cell-List Neighbor Search =====

    float grid_resolution = radius / 1.99f;
    float grid_res_inv = 1.0f / grid_resolution;

    // Compute grid bounds from source positions (on GPU via ATen)
    auto source_pos_flat = source_pos_t.view({-1, 3});
    auto grid_min_t = std::get<0>(source_pos_flat.min(0)); // [3]
    auto grid_max_t = std::get<0>(source_pos_flat.max(0)); // [3]

    // Also consider target positions (wrapped) in the grid bounds
    auto target_min_t = std::get<0>(pos_wrapped.min(0));
    auto target_max_t = std::get<0>(pos_wrapped.max(0));
    grid_min_t = at::minimum(grid_min_t, target_min_t);
    grid_max_t = at::maximum(grid_max_t, target_max_t);

    // Add small epsilon buffer to ensure all positions fit in grid
    grid_min_t = grid_min_t - 0.01f;
    grid_max_t = grid_max_t + 0.01f;

    auto grid_min_cpu = grid_min_t.to(at::kCPU);
    auto grid_max_cpu = grid_max_t.to(at::kCPU);
    float grid_min_x = grid_min_cpu[0].item<float>();
    float grid_min_y = grid_min_cpu[1].item<float>();
    float grid_min_z = grid_min_cpu[2].item<float>();
    float grid_max_x = grid_max_cpu[0].item<float>();
    float grid_max_y = grid_max_cpu[1].item<float>();
    float grid_max_z = grid_max_cpu[2].item<float>();

    int grid_nx = std::max(1, (int)std::ceil((grid_max_x - grid_min_x) * grid_res_inv));
    int grid_ny = std::max(1, (int)std::ceil((grid_max_y - grid_min_y) * grid_res_inv));
    int grid_nz = std::max(1, (int)std::ceil((grid_max_z - grid_min_z) * grid_res_inv));
    int num_cells = grid_nx * grid_ny * grid_nz;

    // B1: Assign source atoms to cells
    auto source_cell_ids = at::empty({M}, opts_int);
    int grid_size_B1 = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assign_cells_kernel<<<grid_size_B1, BLOCK_SIZE, 0, stream>>>(
        source_pos_t.data_ptr<float>(),
        grid_res_inv,
        grid_min_x, grid_min_y, grid_min_z,
        grid_nx, grid_ny, grid_nz,
        M,
        source_cell_ids.data_ptr<int>()
    );

    // Sort source atoms by cell_id
    auto sort_result = at::sort(source_cell_ids.to(at::kLong));
    auto sorted_cell_ids = std::get<0>(sort_result).to(at::kInt);
    auto sort_indices = std::get<1>(sort_result);

    // Gather source arrays by sort order
    auto sorted_source_pos = source_pos_t.index_select(0, sort_indices);
    auto sorted_source_orig_idx = source_orig_idx.index_select(0, sort_indices);
    auto sorted_source_cell_shift = source_cell_shift.index_select(0, sort_indices);

    // B2: Build cell ranges
    auto cell_start_t = at::full({num_cells}, 0, opts_int);
    auto cell_end_t = at::full({num_cells}, 0, opts_int);

    int grid_size_B2 = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    build_cell_ranges_kernel<<<grid_size_B2, BLOCK_SIZE, 0, stream>>>(
        sorted_cell_ids.data_ptr<int>(),
        M,
        num_cells,
        cell_start_t.data_ptr<int>(),
        cell_end_t.data_ptr<int>()
    );

    clk.lap(stream.stream());  // B1-B2: source cell list

    // Assign target atoms to cells (compute their grid coords).
    // If target_indices provided, restrict targets to that subset (sources stay full).
    at::Tensor target_pos_subset;
    if (target_indices.has_value()) {
        target_pos_subset = pos_wrapped.index_select(0, target_indices.value()).contiguous();
    } else {
        target_pos_subset = pos_wrapped;
    }

    // Compute target cell coords on GPU via ATen (using wrapped positions)
    auto target_grid_coords = at::floor((target_pos_subset - grid_min_t.unsqueeze(0)) * grid_res_inv).to(at::kInt);
    // Clamp
    auto grid_dims_t = at::tensor({grid_nx, grid_ny, grid_nz}, opts_int);
    target_grid_coords = at::clamp(target_grid_coords, at::tensor({0}, opts_int), grid_dims_t.unsqueeze(0) - 1);

    at::Tensor target_cell_x = target_grid_coords.select(1, 0).contiguous();
    at::Tensor target_cell_y = target_grid_coords.select(1, 1).contiguous();
    at::Tensor target_cell_z = target_grid_coords.select(1, 2).contiguous();

    // Sort target atoms by cell ID so adjacent warp lanes access spatially adjacent atoms.
    // Gives B3 and B4 the same cache-locality benefit that sorting source atoms gives B1-B2:
    // each warp's 32 threads access the same ~125 neighboring cells → cell_start/end arrays
    // stay in L1/L2 rather than being evicted between threads.
    // edge_tgt values written by B4 are in sorted-target-space; remapped to original
    // atom indices just before the cell-shift correction (after Stage C).
    auto tgt_flat_ids   = target_cell_x.to(at::kLong)
                          + target_cell_y.to(at::kLong) * grid_nx
                          + target_cell_z.to(at::kLong) * (int64_t)grid_nx * grid_ny;
    auto tgt_sort_res   = at::sort(tgt_flat_ids);
    auto sort_perm_in_subset = std::get<1>(tgt_sort_res);  // [N_target]: sorted_idx → subset_idx
    // target_sort_perm maps sorted_idx → GLOBAL atom id (used by kernels to look up frac_floor / write edge_tgt)
    at::Tensor target_sort_perm = target_indices.has_value()
        ? target_indices.value().index_select(0, sort_perm_in_subset)
        : sort_perm_in_subset;
    // Rebind pos_wrapped to the sorted target subset (subsequent code uses it as target array)
    pos_wrapped   = target_pos_subset.index_select(0, sort_perm_in_subset).contiguous();
    target_cell_x = target_cell_x.index_select(0, sort_perm_in_subset);
    target_cell_y = target_cell_y.index_select(0, sort_perm_in_subset);
    target_cell_z = target_cell_z.index_select(0, sort_perm_in_subset);

    // B3: Count neighbors (one thread per target atom — more efficient than warp-per-atom
    // for sparse systems where most cells have < 1 atom on average, as warp-per-atom
    // would require 32× more GPU waves for the same compute).
    // No truncation: every neighbor within the cutoff is counted and emitted
    // (max_num_neighbors == 0 for all current callers).
    int padding_size = (int)std::floor(radius / grid_resolution) + 1;
    float radius_sq = radius * radius;

    auto neighbor_counts = at::empty({N_target}, opts_int);
    {
        int gs = (N_target + BLOCK_SIZE - 1) / BLOCK_SIZE;
        count_neighbors_kernel<<<gs, BLOCK_SIZE, 0, stream>>>(
            pos_wrapped.contiguous().data_ptr<float>(),
            sorted_source_pos.data_ptr<float>(),
            target_cell_x.data_ptr<int>(),
            target_cell_y.data_ptr<int>(),
            target_cell_z.data_ptr<int>(),
            sorted_source_orig_idx.data_ptr<int64_t>(),  // unused in kernel but part of signature
            cell_start_t.data_ptr<int>(),
            cell_end_t.data_ptr<int>(),
            grid_nx, grid_ny, grid_nz,
            padding_size, radius_sq, N_target,
            neighbor_counts.data_ptr<int>()
        );
    }
    clk.lap(stream.stream());  // B3: count_neighbors (single-thread)

    // ===== Counts-only fast path =====
    // Skip prefix-sum / CPU sync / edge buffer / B4 write kernel entirely.
    // Used by Mode A refresh: only per-target degree is needed; edges get
    // built later by Mode B at radius r.
    if (counts_only) {
        auto inverse_perm = sort_perm_in_subset.argsort();
        auto neighbor_counts_user = neighbor_counts.index_select(0, inverse_perm);
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long),
            neighbor_counts_user
        );
    }

    // Prefix sum → neighbor_offsets
    auto neighbor_counts_long = neighbor_counts.to(at::kLong);
    auto neighbor_offsets = at::cumsum(neighbor_counts_long, 0) - neighbor_counts_long;

    // CPU sync: fetch total_edges and max_nbrs_pre (single sync, two values)
    auto totals_cpu = at::cat({
        neighbor_counts_long.sum().unsqueeze(0),
        neighbor_counts_long.max().unsqueeze(0)
    }).to(at::kCPU);
    int64_t total_edges  = totals_cpu[0].item<int64_t>();
    int64_t max_nbrs_pre = totals_cpu[1].item<int64_t>();
    if (s_profile) printf("  total_edges = %lld  max_nbrs = %lld\n",
                          (long long)total_edges, (long long)max_nbrs_pre);
    clk.lap(stream.stream());  // prefix-sum + CPU sync

    if (total_edges == 0) {
        // Un-sort neighbor_counts so the i-th value corresponds to target_indices[i]
        // (or atom i if target_indices not provided). It's all zeros here, but keep
        // the convention consistent with the non-empty path.
        auto inverse_perm = sort_perm_in_subset.argsort();
        auto neighbor_counts_user = neighbor_counts.index_select(0, inverse_perm);
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long),
            neighbor_counts_user
        );
    }

    // Allocate exact-size output arrays (no padding, no pre-fill needed)
    auto edge_src            = at::empty({total_edges}, opts_long);
    auto edge_tgt            = at::empty({total_edges}, opts_long);
    auto edge_cell_shift_out = at::empty({total_edges, 3}, opts_float);
    auto edge_dist_sq        = at::empty({total_edges}, opts_float);
    clk.lap(stream.stream());  // alloc

    // B4: Write neighbors — one templated kernel write_neighbors_kernel<USE_SHM>,
    // dispatched by system density. Both instantiations share the same Phase-A scan;
    // they differ only in how found neighbors are emitted (see the kernel comment).
    //
    //   collect (USE_SHM=true): scan→shm, then Phase C stride-32 coalesced writes.
    //     8KB shm → 75% occupancy. Wins for sparse/solvated systems (<5 atoms/cell).
    //     Example: 2rik density (2.87/cell), 9GMO solvated (2.5/cell).
    //
    //   nosort (USE_SHM=false): ballot+popc direct write, 0 shm → full occupancy.
    //     Wins for dense systems (>=5 atoms/cell, compute-bound). Example: 9GMO density (~22/cell).
    //
    // Crossover ~5 atoms/cell (calibrated on A100-SXM4-40GB). collect additionally
    // requires max_nbrs_pre < MAX_NEIGHBORS_BUF_COLLECT so its shm buffer can't overflow.
    float avg_atoms_per_cell = (num_cells > 0) ? (float)M / (float)num_cells : 0.0f;

    bool use_collect = (max_nbrs_pre < (int64_t)MAX_NEIGHBORS_BUF_COLLECT)
                    && (avg_atoms_per_cell < 5.0f);

    if (s_profile) printf("  avg_atoms_per_cell=%.2f avg_deg=%lld use_collect=%d\n",
                           avg_atoms_per_cell,
                           (long long)(N_target > 0 ? total_edges / N_target : 0),
                           (int)use_collect);

    if (use_collect) {
        int gs = (N_target + WARPS_PER_BLOCK_COLLECT - 1) / WARPS_PER_BLOCK_COLLECT;
        int smem_bytes = WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT
                         * (int)(sizeof(float) + sizeof(int));
        write_neighbors_kernel<true><<<gs, BLOCK_SIZE_COLLECT, smem_bytes, stream>>>(
            pos_wrapped.contiguous().data_ptr<float>(),
            sorted_source_pos.data_ptr<float>(),
            target_cell_x.data_ptr<int>(),
            target_cell_y.data_ptr<int>(),
            target_cell_z.data_ptr<int>(),
            sorted_source_orig_idx.data_ptr<int64_t>(),
            sorted_source_cell_shift.data_ptr<float>(),
            cell_start_t.data_ptr<int>(),
            cell_end_t.data_ptr<int>(),
            grid_nx, grid_ny, grid_nz,
            padding_size, radius_sq, N_target,
            neighbor_counts.data_ptr<int>(),
            neighbor_offsets.data_ptr<int64_t>(),
            target_sort_perm.data_ptr<int64_t>(),
            frac_floor.data_ptr<float>(),
            edge_src.data_ptr<int64_t>(),
            edge_tgt.data_ptr<int64_t>(),
            edge_cell_shift_out.data_ptr<float>(),
            edge_dist_sq.data_ptr<float>()
        );
    } else {
        int gs = (N_target + WARPS_PER_BLOCK_WARP - 1) / WARPS_PER_BLOCK_WARP;
        write_neighbors_kernel<false><<<gs, BLOCK_SIZE, 0, stream>>>(
            pos_wrapped.contiguous().data_ptr<float>(),
            sorted_source_pos.data_ptr<float>(),
            target_cell_x.data_ptr<int>(),
            target_cell_y.data_ptr<int>(),
            target_cell_z.data_ptr<int>(),
            sorted_source_orig_idx.data_ptr<int64_t>(),
            sorted_source_cell_shift.data_ptr<float>(),
            cell_start_t.data_ptr<int>(),
            cell_end_t.data_ptr<int>(),
            grid_nx, grid_ny, grid_nz,
            padding_size, radius_sq, N_target,
            neighbor_counts.data_ptr<int>(),
            neighbor_offsets.data_ptr<int64_t>(),
            target_sort_perm.data_ptr<int64_t>(),
            frac_floor.data_ptr<float>(),
            edge_src.data_ptr<int64_t>(),
            edge_tgt.data_ptr<int64_t>(),
            edge_cell_shift_out.data_ptr<float>(),
            edge_dist_sq.data_ptr<float>()
        );
    }
    clk.lap(stream.stream());  // B4-kernel (includes target remap + frac_floor correction)

    // Stage C: target remap and frac_floor correction are done inline in B4 kernels.

    auto edge_index = at::stack({edge_src, edge_tgt}, 0);
    auto num_neighbors_image = at::tensor({total_edges}, opts_long);

    // Un-sort neighbor_counts: kernel wrote in sorted-target order; user expects
    // neighbor_counts[i] = degree of target_indices[i] (or atom i in legacy mode).
    auto inverse_perm = sort_perm_in_subset.argsort();
    auto neighbor_counts_user = neighbor_counts.index_select(0, inverse_perm);

    return std::make_tuple(edge_index, edge_cell_shift_out, num_neighbors_image, neighbor_counts_user);
}

// Legacy 3-tuple wrapper: forwards to _ex without target_indices, discards neighbor_counts.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl(
    at::Tensor pos,
    at::Tensor cell,
    at::Tensor pbc,
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly
) {
    auto result = radius_graph_pbc_cuda_impl_ex(
        pos, cell, pbc, radius, max_num_neighbors, enforce_max_strictly,
        c10::nullopt, /*counts_only=*/false);
    return std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result));
}
