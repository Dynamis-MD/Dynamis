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

// Kernel B4: Write neighbor edges (original CSR version — kept for reference)
__global__ void write_neighbors_kernel_csr(
    const float* __restrict__ target_pos,   // [N, 3]
    const float* __restrict__ source_pos,   // [M, 3]
    const int* __restrict__ target_cell_x,  // [N]
    const int* __restrict__ target_cell_y,  // [N]
    const int* __restrict__ target_cell_z,  // [N]
    const int64_t* __restrict__ source_orig_idx, // [M]
    const float* __restrict__ source_cell_shift, // [M, 3]
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int grid_nx, int grid_ny, int grid_nz,
    int padding_size,
    float radius_sq,
    int N,
    const int64_t* __restrict__ neighbor_offsets, // [N] exclusive prefix sum
    int64_t* __restrict__ edge_src,          // [E] source original idx
    int64_t* __restrict__ edge_tgt,          // [E] target idx
    float* __restrict__ edge_cell_shift,     // [E, 3]
    float* __restrict__ edge_dist_sq         // [E]
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    float tx = target_pos[tidx * 3 + 0];
    float ty = target_pos[tidx * 3 + 1];
    float tz = target_pos[tidx * 3 + 2];

    int tcx = target_cell_x[tidx];
    int tcy = target_cell_y[tidx];
    int tcz = target_cell_z[tidx];

    int64_t write_offset = neighbor_offsets[tidx];
    int write_count = 0;

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

                    if (dist_sq <= radius_sq && dist_sq > 1e-8f) {
                        int64_t out = write_offset + write_count;
                        edge_src[out] = source_orig_idx[s];
                        edge_tgt[out] = (int64_t)tidx;
                        edge_cell_shift[out * 3 + 0] = source_cell_shift[s * 3 + 0];
                        edge_cell_shift[out * 3 + 1] = source_cell_shift[s * 3 + 1];
                        edge_cell_shift[out * 3 + 2] = source_cell_shift[s * 3 + 2];
                        edge_dist_sq[out] = dist_sq;
                        write_count++;
                    }
                }
            }
        }
    }
}

// Kernel B4: Write neighbor edges — warp-transposed layout for coalesced writes.
//
// Instead of CSR order (thread i writes to neighbor_offsets[i] + j, which puts
// adjacent warp-lanes ~k_avg apart), we use a warp-major round layout:
//   atom at (warp w, lane l) writes its j-th edge to warp_offsets[w] + j*32 + l
//
// Coalescing: at round j, all 32 lanes write to 32 consecutive addresses
//   warp_base+j*32+0, ..., warp_base+j*32+31  →  one 256-byte transaction.
//
// edge_src, edge_tgt, edge_dist_sq : [total_blocked] — stride-1, fully coalesced.
// edge_cell_shift_out              : [3, total_blocked] transposed layout — stride-1
//   per dimension row, fully coalesced (total_blocked passed explicitly for stride).
//
// The output is padded (max_k_w * 32 slots per warp, invalid slots have tgt=-1).
// The caller compacts, transposes cell_shift to [E,3], and sorts by tgt for Stage C.
__global__ void write_neighbors_transposed_kernel(
    const float* __restrict__ target_pos,
    const float* __restrict__ source_pos,
    const int* __restrict__ target_cell_x,
    const int* __restrict__ target_cell_y,
    const int* __restrict__ target_cell_z,
    const int64_t* __restrict__ source_orig_idx,
    const float* __restrict__ source_cell_shift,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int grid_nx, int grid_ny, int grid_nz,
    int padding_size,
    float radius_sq,
    int N,
    const int64_t* __restrict__ warp_offsets,  // [num_warps] excl. prefix sum of warp_sizes
    int64_t total_blocked,                     // size of each output array
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,            // pre-filled with -1; valid edges get tgt >= 0
    float* __restrict__ edge_cell_shift_out,   // [3, total_blocked] — transposed for coalesced writes
    float* __restrict__ edge_dist_sq
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;

    // Lane within warp (0..31) and global warp id
    int lane           = threadIdx.x & 31;
    int warp_global    = tidx >> 5;
    int64_t warp_base  = warp_offsets[warp_global];

    float tx = target_pos[tidx * 3 + 0];
    float ty = target_pos[tidx * 3 + 1];
    float tz = target_pos[tidx * 3 + 2];

    int tcx = target_cell_x[tidx];
    int tcy = target_cell_y[tidx];
    int tcz = target_cell_z[tidx];

    int write_count = 0;

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
                int start   = cell_start[cell_id];
                int end     = cell_end[cell_id];

                for (int s = start; s < end; s++) {
                    float sx = source_pos[s * 3 + 0];
                    float sy = source_pos[s * 3 + 1];
                    float sz = source_pos[s * 3 + 2];

                    float ddx = tx - sx;
                    float ddy = ty - sy;
                    float ddz = tz - sz;
                    float dist_sq = ddx*ddx + ddy*ddy + ddz*ddz;

                    if (dist_sq <= radius_sq && dist_sq > 1e-8f) {
                        // Warp-transposed index: round write_count, lane → coalesced
                        int64_t out = warp_base + (int64_t)write_count * 32 + lane;
                        edge_src[out]  = source_orig_idx[s];
                        edge_tgt[out]  = (int64_t)tidx;
                        edge_dist_sq[out] = dist_sq;
                        // cell_shift stored in [3, total_blocked] layout for coalesced writes:
                        //   all lanes write to consecutive positions within each dimension row
                        edge_cell_shift_out[0 * total_blocked + out] = source_cell_shift[s * 3 + 0];
                        edge_cell_shift_out[1 * total_blocked + out] = source_cell_shift[s * 3 + 1];
                        edge_cell_shift_out[2 * total_blocked + out] = source_cell_shift[s * 3 + 2];
                        write_count++;
                    }
                }
            }
        }
    }
}

// ===== Stage B (new): Warp-per-atom neighbor kernels =====
//
// Replace one-thread-per-atom B3/B4 with one-warp-per-atom variants.
// Each warp (32 threads) cooperatively scans the (2*padding+1)^3 neighboring cells
// for a single target atom using __ballot_sync + __popc for fast counting and
// coalesced writes.
//
// No-sort path (max_num_neighbors == 0):
//   B3 warp: count, capped at max_num_neighbors
//   B4 nosort: scan again, use ballot+popc to write to consecutive offsets (coalesced)
//   No shared memory, no padding, no compact step.
//
// Sort path (max_num_neighbors > 0):
//   B3 warp: count, capped at max_num_neighbors
//   B4 sort: scan into shm buffer, bitonic sort by dist, write top-K (coalesced)
//   Eliminates Stage C sort/truncation; B3 capping + shm sort handles it.

static constexpr int WARPS_PER_BLOCK_WARP = 8;   // count + nosort: 8*32=256 threads/block
// (BLOCK_SIZE == 256 == 8*32; reusing existing constant for those kernels)
static constexpr int WARPS_PER_BLOCK_SORT = 4;   // sort kernel: 4*32=128 threads/block
static constexpr int BLOCK_SIZE_SORT = 32 * WARPS_PER_BLOCK_SORT;  // 128
// shm per sort block: WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF * (4B float + 4B int)
//   = 4 * 1024 * 8 = 32 KB  (fits in 96 KB per SM)
static constexpr int MAX_NEIGHBORS_BUF = 1024;   // power-of-2 shm buffer per warp

// Collect kernel: Phase A + Phase C only (no sort, no truncation).
// Used when max_nbrs_pre < MAX_NEIGHBORS_BUF_COLLECT, i.e. no atom exceeds K neighbors.
// Smaller buffer → higher occupancy:
//   8 * 256 * 8 = 16 KB (nosort: 8 warps) — or 4 * 256 * 8 = 8 KB (sort: 4 warps)
// Using sort-path warp count for coalesced Phase C writes at 75% occupancy.
static constexpr int MAX_NEIGHBORS_BUF_COLLECT = 256;
static constexpr int WARPS_PER_BLOCK_COLLECT   = WARPS_PER_BLOCK_SORT;  // 4
static constexpr int BLOCK_SIZE_COLLECT        = BLOCK_SIZE_SORT;        // 128

// CTA cooperative kernel (coop path): WARPS_PER_CTA_COOP warps cooperate per target atom.
// Phase A: WARPS_PER_CTA_COOP× parallel cell scanning (27 cells / 4 warps ≈ 7 cells/warp).
// Phase C: stride-32 coalesced writes per warp (same quality as collect).
// shm: WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT * (4+4)B + 2*WARPS_PER_CTA_COOP*4B
//    = 4 * 256 * 8 + 32 = 8224B → 11 blocks/SM → 44/64 warps = 69% occupancy (A100)
static constexpr int WARPS_PER_CTA_COOP = 4;
static constexpr int BLOCK_SIZE_COOP_W  = 32 * WARPS_PER_CTA_COOP;   // 128

// Warp-cooperative ascending bitonic sort on shared memory.
// Sorts s_dist[0..n-1] ascending; s_src permuted accordingly.
// n must be a power of 2, n <= MAX_NEIGHBORS_BUF.
// Must be called by all 32 threads of a warp; uses __syncwarp() between stages.
__device__ __forceinline__ void bitonic_sort_warp_smem(
    float* __restrict__ s_dist,
    int*   __restrict__ s_src,
    int n, int lane
) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            for (int tid = lane; tid < n; tid += 32) {
                int partner = tid ^ j;
                if (partner > tid) {
                    bool ascending = ((tid & k) == 0);
                    bool do_swap = ascending ? (s_dist[tid] > s_dist[partner])
                                             : (s_dist[tid] < s_dist[partner]);
                    if (do_swap) {
                        float fd = s_dist[tid];  s_dist[tid] = s_dist[partner];  s_dist[partner] = fd;
                        int   fi = s_src[tid];   s_src[tid]  = s_src[partner];   s_src[partner]  = fi;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Kernel B3 (warp-per-atom): Count neighbors for each target atom cooperatively.
// One warp (32 lanes) handles one target atom; count is capped at max_num_neighbors if > 0.
__global__ void count_neighbors_warp_kernel(
    const float* __restrict__ target_pos,
    const float* __restrict__ source_pos,
    const int*   __restrict__ target_cell_x,
    const int*   __restrict__ target_cell_y,
    const int*   __restrict__ target_cell_z,
    const int*   __restrict__ cell_start,
    const int*   __restrict__ cell_end,
    int grid_nx, int grid_ny, int grid_nz,
    int padding_size,
    float radius_sq,
    int N,
    int max_num_neighbors,
    int* __restrict__ neighbor_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= N) return;

    float tx = target_pos[warp_id * 3 + 0];
    float ty = target_pos[warp_id * 3 + 1];
    float tz = target_pos[warp_id * 3 + 2];
    int   tcx = target_cell_x[warp_id];
    int   tcy = target_cell_y[warp_id];
    int   tcz = target_cell_z[warp_id];

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
                int cs = cell_start[cell_id];
                int ce = cell_end[cell_id];
                int iters = (ce - cs + 31) >> 5;  // ceil((ce-cs)/32)

                for (int it = 0; it < iters; it++) {
                    int s = cs + (it << 5) + lane;
                    bool is_nbr = false;
                    if (s < ce) {
                        float sx = source_pos[s * 3 + 0];
                        float sy = source_pos[s * 3 + 1];
                        float sz = source_pos[s * 3 + 2];
                        float d2 = (tx-sx)*(tx-sx) + (ty-sy)*(ty-sy) + (tz-sz)*(tz-sz);
                        is_nbr = (d2 <= radius_sq) && (d2 > 1e-8f);
                    }
                    count += __popc(__ballot_sync(0xFFFFFFFFu, is_nbr));
                }
            }
        }
    }

    if (lane == 0) {
        neighbor_counts[warp_id] = (max_num_neighbors > 0 && count > max_num_neighbors)
            ? max_num_neighbors : count;
    }
}

// Kernel B4 (no-sort path): Warp-per-atom neighbor write.
// Used when max_num_neighbors == 0.
//
// One warp (32 threads) handles all neighbor candidates for one target atom.
// For each of the (2*padding+1)^3 neighbor cells, lanes stride through the cell's
// atoms in chunks of 32: s = cell_start + it*32 + lane.  Atoms within a cell are
// stored contiguously in memory, so active lanes produce coalesced loads even when
// the cell is sparse (e.g. 2-3 atoms/cell → lanes 0-2 load consecutive addresses).
// This is preferable to collecting all candidates first and processing in full 32-lane
// batches, which would scatter reads across 32 different cells' memory regions.
__global__ void write_neighbors_nosort_kernel(
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
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ target_sort_perm,  // sorted_idx → original_idx
    const float*   __restrict__ frac_floor,         // [N_orig, 3] PBC wrap correction
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,  // [E, 3]
    float*   __restrict__ edge_dist_sq
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= N) return;

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

    int64_t base = neighbor_offsets[warp_id];
    int write_idx = 0;

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
                    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                    bool in_cell = (s < ce);
                    if (in_cell) {
                        sx = source_pos[s * 3 + 0];
                        sy = source_pos[s * 3 + 1];
                        sz = source_pos[s * 3 + 2];
                    }
                    float d2 = (tx-sx)*(tx-sx) + (ty-sy)*(ty-sy) + (tz-sz)*(tz-sz);
                    bool is_nbr = in_cell && (d2 <= radius_sq) && (d2 > 1e-8f);
                    uint32_t mask = __ballot_sync(0xFFFFFFFFu, is_nbr);
                    if (is_nbr) {
                        int local = __popc(mask & ((1u << lane) - 1));
                        int64_t out = base + write_idx + local;
                        int64_t src_orig = source_orig_idx[s];
                        float sfx = frac_floor[src_orig * 3 + 0];
                        float sfy = frac_floor[src_orig * 3 + 1];
                        float sfz = frac_floor[src_orig * 3 + 2];
                        edge_src[out] = src_orig;
                        edge_tgt[out] = tgt_orig;
                        edge_dist_sq[out] = d2;
                        edge_cell_shift_out[out * 3 + 0] = source_cell_shift[s * 3 + 0] + tfx - sfx;
                        edge_cell_shift_out[out * 3 + 1] = source_cell_shift[s * 3 + 1] + tfy - sfy;
                        edge_cell_shift_out[out * 3 + 2] = source_cell_shift[s * 3 + 2] + tfz - sfz;
                    }
                    write_idx += __popc(mask);
                }
            }
        }
    }
}

// Kernel B4 (collect path): Phase A + Phase C, no sorting.
// Used when need_sort=False AND max_nbrs_pre < MAX_NEIGHBORS_BUF_COLLECT.
// Key advantage over nosort kernel: Phase C uses stride-32 writes (all 32 lanes active
// per iteration) instead of ballot+popc scatter (only ~3 lanes active per iteration).
// Write coalescing: 100% vs ~9% → major throughput improvement.
// Shared memory: WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT * (4B + 4B)
//   = 4 * 256 * 8 = 8 KB per block → 12 blocks/SM → 48/64 warps = 75% occupancy.
__global__ void __launch_bounds__(BLOCK_SIZE_COLLECT)
write_neighbors_collect_kernel(
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
    const int* __restrict__ neighbor_counts,   // per-atom neighbor count (from B3)
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ target_sort_perm,
    const float*   __restrict__ frac_floor,
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,
    float*   __restrict__ edge_dist_sq
) {
    // Dynamic shared memory layout:
    //   float s_dist[WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT]
    //   int   s_src [WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT]
    extern __shared__ float smem_c[];
    int* smem_c_int = (int*)(smem_c + WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT);

    int warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane       = threadIdx.x & 31;
    int warp_local = threadIdx.x >> 5;

    if (warp_id >= N) return;

    float* s_dist = smem_c     + warp_local * MAX_NEIGHBORS_BUF_COLLECT;
    int*   s_src  = smem_c_int + warp_local * MAX_NEIGHBORS_BUF_COLLECT;

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

    int n_found = 0;

    // Phase A: collect all neighbors into shared memory buffer.
    // Dispatch ensures max_nbrs_pre < MAX_NEIGHBORS_BUF_COLLECT, so no overflow occurs.
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
                        int store_pos = n_found + __popc(mask & ((1u << lane) - 1));
                        // guard: dispatch guarantees n_found < MAX_NEIGHBORS_BUF_COLLECT
                        if (store_pos < MAX_NEIGHBORS_BUF_COLLECT) {
                            s_dist[store_pos] = d2;
                            s_src[store_pos]  = s;
                        }
                    }
                    n_found += __popc(mask);
                }
            }
        }
    }

    // Phase B: no sorting — dispatch ensures all neighbors fit in output.

    // Barrier: Phase A's shared-memory stores (s_dist/s_src, written by whichever
    // lane owned each store_pos) must be visible before Phase C reads them across
    // lanes (lane L reads slot L+32k, generally written by a different lane). On
    // Volta+ independent thread scheduling this ordering is NOT guaranteed without
    // an explicit warp sync; omitting it lets a lane read a stale slot left over
    // from the previous block's warp on this SM. Because targets are cell-sorted,
    // that leftover is usually a neighbor of a spatially adjacent atom and thus
    // often a real neighbor of this atom too → a byte-exact duplicate edge.
    __syncwarp();

    // Phase C: write all n_found neighbors to global memory.
    // Stride-32 pattern: all 32 lanes active per iteration → fully coalesced writes.
    int out_count = neighbor_counts[warp_id];  // == n_found (B3 and Phase A agree)
    int64_t base = neighbor_offsets[warp_id];
    for (int i = lane; i < out_count; i += 32) {
        int s = s_src[i];
        int64_t out = base + i;
        int64_t src_orig = source_orig_idx[s];
        float sfx = frac_floor[src_orig * 3 + 0];
        float sfy = frac_floor[src_orig * 3 + 1];
        float sfz = frac_floor[src_orig * 3 + 2];
        edge_src[out] = src_orig;
        edge_tgt[out] = tgt_orig;
        edge_dist_sq[out] = s_dist[i];
        edge_cell_shift_out[out * 3 + 0] = source_cell_shift[s * 3 + 0] + tfx - sfx;
        edge_cell_shift_out[out * 3 + 1] = source_cell_shift[s * 3 + 1] + tfy - sfy;
        edge_cell_shift_out[out * 3 + 2] = source_cell_shift[s * 3 + 2] + tfz - sfz;
    }
}

// Kernel B4 (cooperative path): WARPS_PER_CTA_COOP warps collaborate on one target atom.
// One block per target atom; warps scan disjoint subsets of neighbor cells in parallel.
//
// Phase A: each warp scans cells {ci : ci % WARPS_PER_CTA_COOP == warp_local}
//          and writes (dist_sq, src_idx) to its own shm slice.
//          27 cells / 4 warps ≈ 7 cells/warp vs 27 serial → ~4× Phase A speedup.
// Phase B: warp 0 computes exclusive prefix sum of per-warp counts (4-value Hillis-Steele).
// Phase C: each warp writes its shm slice at global[base + warp_offset + i],
//          stride-32 over i → all 32 lanes active → fully coalesced writes.
//
// Used when avg_candidates ≥ 150 (dense systems where Phase A parallelism pays off).
// Shared memory: WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT * (4+4)B + 32B ≈ 8KB
__global__ void __launch_bounds__(BLOCK_SIZE_COOP_W)
write_neighbors_coop_kernel(
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
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ target_sort_perm,
    const float*   __restrict__ frac_floor,
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,
    float*   __restrict__ edge_dist_sq
) {
    // Shared memory layout:
    //   float s_dist[WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT]  (per-warp dist buffer)
    //   int   s_src [WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT]  (per-warp src buffer)
    //   int   warp_count [WARPS_PER_CTA_COOP]                          (per-warp neighbor count)
    //   int   warp_offset[WARPS_PER_CTA_COOP]                          (exclusive prefix sum)
    extern __shared__ float smem_coop[];
    int* smem_coop_int = (int*)(smem_coop + WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT);
    int* warp_count_s  = smem_coop_int + WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT;
    int* warp_offset_s = warp_count_s  + WARPS_PER_CTA_COOP;

    int atom_id    = blockIdx.x;          // one block per sorted target atom
    int warp_local = threadIdx.x >> 5;    // 0..WARPS_PER_CTA_COOP-1
    int lane       = threadIdx.x & 31;

    if (atom_id >= N) return;

    float* s_dist = smem_coop     + warp_local * MAX_NEIGHBORS_BUF_COLLECT;
    int*   s_src  = smem_coop_int + warp_local * MAX_NEIGHBORS_BUF_COLLECT;

    // Load target atom data — all warps read same addresses; L1-cached after first warp.
    int64_t tgt_orig = target_sort_perm[atom_id];
    float tx  = target_pos[atom_id * 3 + 0];
    float ty  = target_pos[atom_id * 3 + 1];
    float tz  = target_pos[atom_id * 3 + 2];
    int   tcx = target_cell_x[atom_id];
    int   tcy = target_cell_y[atom_id];
    int   tcz = target_cell_z[atom_id];
    float tfx = frac_floor[tgt_orig * 3 + 0];
    float tfy = frac_floor[tgt_orig * 3 + 1];
    float tfz = frac_floor[tgt_orig * 3 + 2];

    // Phase A: each warp scans its assigned neighbor cells in parallel.
    // Flat cell index ci encodes (dx, dy, dz) with x-fastest:
    //   dx = ci % dim - padding_size
    //   dy = (ci / dim) % dim - padding_size
    //   dz = ci / dim^2 - padding_size
    int dim = 2 * padding_size + 1;
    int num_cells_3d = dim * dim * dim;  // 27 for padding_size=1

    int n_found = 0;

    for (int ci = warp_local; ci < num_cells_3d; ci += WARPS_PER_CTA_COOP) {
        int tmp    = ci;
        int dx     = tmp % dim - padding_size; tmp /= dim;
        int dy     = tmp % dim - padding_size; tmp /= dim;
        int dz     = tmp      - padding_size;

        int nx_val = tcx + dx;
        int ny_val = tcy + dy;
        int nz_val = tcz + dz;
        if (nx_val < 0 || nx_val >= grid_nx ||
            ny_val < 0 || ny_val >= grid_ny ||
            nz_val < 0 || nz_val >= grid_nz) continue;

        int cell_id = nx_val + ny_val * grid_nx + nz_val * grid_nx * grid_ny;
        int cs  = cell_start[cell_id];
        int ce  = cell_end[cell_id];
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
                int store_pos = n_found + __popc(mask & ((1u << lane) - 1));
                if (store_pos < MAX_NEIGHBORS_BUF_COLLECT) {
                    s_dist[store_pos] = d2;
                    s_src[store_pos]  = s;
                }
            }
            n_found += __popc(mask);
        }
    }

    // Write per-warp count to shared memory, then sync across CTA.
    if (lane == 0) warp_count_s[warp_local] = n_found;
    __syncthreads();

    // Phase B: warp 0 computes exclusive prefix sum of 4 warp counts.
    // All 32 lanes of warp 0 participate in __shfl_up_sync; only lanes 0..3 store results.
    if (warp_local == 0) {
        int v = (lane < WARPS_PER_CTA_COOP) ? warp_count_s[lane] : 0;
        for (int off = 1; off < WARPS_PER_CTA_COOP; off <<= 1) {
            int t = __shfl_up_sync(0xFFFFFFFFu, v, off);
            if (lane >= off && lane < WARPS_PER_CTA_COOP) v += t;
        }
        if (lane < WARPS_PER_CTA_COOP) {
            warp_offset_s[lane] = v - warp_count_s[lane];   // exclusive prefix sum
        }
    }
    __syncthreads();

    // Phase C: each warp writes its shm slice to global memory.
    // Stride-32 over i: all 32 lanes active per iteration → fully coalesced writes.
    int     my_count  = warp_count_s[warp_local];
    int     my_offset = warp_offset_s[warp_local];
    int64_t base      = neighbor_offsets[atom_id];

    for (int i = lane; i < my_count; i += 32) {
        int s        = s_src[i];
        int64_t out  = base + my_offset + i;
        int64_t src_orig = source_orig_idx[s];
        float sfx = frac_floor[src_orig * 3 + 0];
        float sfy = frac_floor[src_orig * 3 + 1];
        float sfz = frac_floor[src_orig * 3 + 2];
        edge_src[out]              = src_orig;
        edge_tgt[out]              = tgt_orig;
        edge_dist_sq[out]          = s_dist[i];
        edge_cell_shift_out[out * 3 + 0] = source_cell_shift[s * 3 + 0] + tfx - sfx;
        edge_cell_shift_out[out * 3 + 1] = source_cell_shift[s * 3 + 1] + tfy - sfy;
        edge_cell_shift_out[out * 3 + 2] = source_cell_shift[s * 3 + 2] + tfz - sfz;
    }
}

// Kernel B4 (sort path): Write neighbor edges cooperatively, warp-per-atom.
// Used when max_num_neighbors > 0. Per warp:
//   Phase A: scan all 125 cells, collect (dist_sq, src_idx) into shm buffer
//   Phase B: bitonic sort shm buffer by dist_sq (ascending) if n_found > out_count
//   Phase C: write first out_count entries to global memory (coalesced: stride-1)
// Shared memory: WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF * (sizeof(float) + sizeof(int))
//   = 4 * 1024 * 8 = 32 KB per block.
__global__ void __launch_bounds__(BLOCK_SIZE_SORT)
write_neighbors_sort_kernel(
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
    const int* __restrict__ neighbor_counts_capped,  // per-atom output count (capped at K)
    const int64_t* __restrict__ neighbor_offsets,
    const int64_t* __restrict__ target_sort_perm,  // sorted_idx → original_idx
    const float*   __restrict__ frac_floor,         // [N_orig, 3] PBC wrap correction
    int64_t* __restrict__ edge_src,
    int64_t* __restrict__ edge_tgt,
    float*   __restrict__ edge_cell_shift_out,  // [E, 3]
    float*   __restrict__ edge_dist_sq
) {
    // Dynamic shared memory: float[WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF] for dist
    //                         int  [WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF] for src
    extern __shared__ float smem[];
    int* smem_int = (int*)(smem + WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF);

    int warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane       = threadIdx.x & 31;
    int warp_local = threadIdx.x >> 5;  // 0..WARPS_PER_BLOCK_SORT-1

    if (warp_id >= N) return;

    float* s_dist = smem     + warp_local * MAX_NEIGHBORS_BUF;
    int*   s_src  = smem_int + warp_local * MAX_NEIGHBORS_BUF;

    // Original target atom index (for edge_tgt and frac_floor lookup)
    int64_t tgt_orig = target_sort_perm[warp_id];

    float tx = target_pos[warp_id * 3 + 0];
    float ty = target_pos[warp_id * 3 + 1];
    float tz = target_pos[warp_id * 3 + 2];
    int   tcx = target_cell_x[warp_id];
    int   tcy = target_cell_y[warp_id];
    int   tcz = target_cell_z[warp_id];

    // Load target frac_floor once per warp
    float tfx = frac_floor[tgt_orig * 3 + 0];
    float tfy = frac_floor[tgt_orig * 3 + 1];
    float tfz = frac_floor[tgt_orig * 3 + 2];

    int n_found = 0;

    // Phase A: collect all neighbor candidates into shared memory.
    // All 32 lanes always call __ballot_sync together; no intra-warp divergence.
    // If buffer overflows (degree > MAX_NEIGHBORS_BUF), we stop; since B3 already
    // capped at max_num_neighbors <= MAX_NEIGHBORS_BUF, this only affects pathological
    // cases with degree >> max_num_neighbors.
    bool done = false;
    for (int dz = -padding_size; dz <= padding_size && !done; dz++) {
        int nz = tcz + dz;
        if (nz < 0 || nz >= grid_nz) continue;
        for (int dy = -padding_size; dy <= padding_size && !done; dy++) {
            int ny = tcy + dy;
            if (ny < 0 || ny >= grid_ny) continue;
            for (int dx = -padding_size; dx <= padding_size && !done; dx++) {
                int nx_val = tcx + dx;
                if (nx_val < 0 || nx_val >= grid_nx) continue;

                int cell_id = nx_val + ny * grid_nx + nz * grid_nx * grid_ny;
                int cs = cell_start[cell_id];
                int ce = cell_end[cell_id];
                int iters = (ce - cs + 31) >> 5;

                for (int it = 0; it < iters && !done; it++) {
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
                    int batch = __popc(mask);
                    if (is_nbr) {
                        int local = __popc(mask & ((1u << lane) - 1));
                        int store_pos = n_found + local;
                        if (store_pos < MAX_NEIGHBORS_BUF) {
                            s_dist[store_pos] = d2;
                            s_src[store_pos]  = s;
                        }
                    }
                    n_found += batch;
                    if (n_found >= MAX_NEIGHBORS_BUF) {
                        n_found = MAX_NEIGHBORS_BUF;
                        done = true;
                    }
                }
            }
        }
    }

    int out_count = neighbor_counts_capped[warp_id];

    // Phase B: sort if we found more candidates than we need to output.
    if (n_found > out_count) {
        // Pad to next power of 2 for bitonic sort (n_found <= MAX_NEIGHBORS_BUF = 1024)
        int padded = 1;
        while (padded < n_found) padded <<= 1;

        // Fill padding slots with infinity / invalid src
        for (int i = n_found + lane; i < padded; i += 32) {
            s_dist[i] = 3.4028235e38f;  // FLT_MAX
            s_src[i]  = -1;
        }
        __syncwarp();

        bitonic_sort_warp_smem(s_dist, s_src, padded, lane);
        // s_dist[0..n_found-1] now sorted ascending (nearest first)
    }

    // Barrier: order Phase A's shared-memory stores before Phase C's cross-lane
    // reads. When sort runs, bitonic_sort_warp_smem ends on a __syncwarp(); but
    // when n_found <= out_count the sort branch is skipped, leaving Phase A writes
    // unsynchronized against Phase C reads (same warp-stale-read hazard as the
    // collect kernel). An unconditional __syncwarp() here covers both branches.
    __syncwarp();

    // Phase C: write top out_count edges to global memory.
    // Consecutive lanes write to consecutive addresses: stride-1 → coalesced.
    int64_t base = neighbor_offsets[warp_id];
    for (int i = lane; i < out_count; i += 32) {
        int s = s_src[i];
        int64_t out = base + i;
        int64_t src_orig = source_orig_idx[s];
        float sfx = frac_floor[src_orig * 3 + 0];
        float sfy = frac_floor[src_orig * 3 + 1];
        float sfz = frac_floor[src_orig * 3 + 2];
        edge_src[out] = src_orig;
        edge_tgt[out] = tgt_orig;
        edge_dist_sq[out] = s_dist[i];
        edge_cell_shift_out[out * 3 + 0] = source_cell_shift[s * 3 + 0] + tfx - sfx;
        edge_cell_shift_out[out * 3 + 1] = source_cell_shift[s * 3 + 1] + tfy - sfy;
        edge_cell_shift_out[out * 3 + 2] = source_cell_shift[s * 3 + 2] + tfz - sfz;
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
    // Counts are capped at max_num_neighbors in Stage C / B4 sort path.
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

    // B4: Write neighbors — four kernels, dispatched by system density and truncation need.
    //
    //   coop kernel: WARPS_PER_CTA_COOP warps cooperate per target atom.
    //     Phase A: ~4× parallel cell scanning; Phase C: stride-32 coalesced writes.
    //     ~8KB shm → ~69% occupancy. Best for dense systems (avg_candidates ≥ 150).
    //     Example: 9GMO density (22 atoms/cell, 594 cand/atom) → coop beats nosort.
    //
    //   collect kernel: Phase A (scan→shm) + Phase C (stride-32 coalesced writes).
    //     8KB shm → 75% occupancy. Best for sparse/solvated systems (avg_degree < 90).
    //     Example: 9GMO solvated (2.5 atoms/cell, 67 cand/atom) → collect beats nosort.
    //
    //   nosort kernel: ballot+popc direct write, 0 shm → full occupancy, ~9% coalescing.
    //     Fallback for medium density: avg_candidates < 150 but avg_degree ≥ 90.
    //
    //   sort kernel: Phase A + bitonic sort + Phase C. 32KB shm → 18% occupancy.
    //     Only used when truncation is required (max_nbrs_pre > max_num_neighbors).
    //
    // avg_candidates = (M / num_cells) × (2*padding+1)^3  — proxy for scan work per atom.
    // avg_degree     = total_edges / N                     — proxy for output write work.
    //
    float avg_atoms_per_cell = (num_cells > 0) ? (float)M / (float)num_cells : 0.0f;
    int   n_nbr_d        = 2 * padding_size + 1;
    float avg_candidates = avg_atoms_per_cell * (float)(n_nbr_d * n_nbr_d * n_nbr_d);

    bool need_sort   = (max_num_neighbors > 0) && (max_nbrs_pre > (int64_t)max_num_neighbors);
    // coop: dense Phase A (many candidates); per-warp buf must fit worst-case neighbors
    bool use_coop    = !need_sort
                    && (max_nbrs_pre < (int64_t)MAX_NEIGHBORS_BUF_COLLECT)
                    && false;  // coop is slower than nosort on tested systems (extra block
                               // overhead from N blocks vs N/8 outweighs Phase A parallelism)
                               // Kept for reference; not dispatched.
    // collect: sparse Phase A + coalesced writes.
    // Use when avg_atoms_per_cell < 5: few atoms/cell → Phase A is latency-dominated →
    // shm write overhead in collect is small, and Phase C coalescing win is meaningful.
    // Use nosort when avg_atoms_per_cell ≥ 5 (dense: 9GMO density ~22/cell) — Phase A
    // is compute-dominated and shm overhead exceeds Phase C benefit.
    // Calibrated on A100-SXM4-40GB: 2rik density (2.87/cell) collect wins; 9GMO density
    // (22/cell) nosort wins. Crossover ~5 atoms/cell.
    bool use_collect = !need_sort && !use_coop
                    && (max_nbrs_pre < (int64_t)MAX_NEIGHBORS_BUF_COLLECT)
                    && (avg_atoms_per_cell < 5.0f);

    if (s_profile) printf("  avg_cand=%.1f avg_deg=%lld use_coop=%d use_collect=%d need_sort=%d\n",
                           avg_candidates, (long long)(N_target > 0 ? total_edges / N_target : 0),
                           (int)use_coop, (int)use_collect, (int)need_sort);

    if (need_sort) {
        int gs = (N_target + WARPS_PER_BLOCK_SORT - 1) / WARPS_PER_BLOCK_SORT;
        int smem_bytes = WARPS_PER_BLOCK_SORT * MAX_NEIGHBORS_BUF
                         * (int)(sizeof(float) + sizeof(int));
        write_neighbors_sort_kernel<<<gs, BLOCK_SIZE_SORT, smem_bytes, stream>>>(
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
    } else if (use_coop) {
        // One block per target atom; all WARPS_PER_CTA_COOP warps cooperate on that atom.
        int smem_bytes = WARPS_PER_CTA_COOP * MAX_NEIGHBORS_BUF_COLLECT
                         * (int)(sizeof(float) + sizeof(int))
                         + 2 * WARPS_PER_CTA_COOP * (int)sizeof(int);  // count + offset arrays
        write_neighbors_coop_kernel<<<N_target, BLOCK_SIZE_COOP_W, smem_bytes, stream>>>(
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
            neighbor_offsets.data_ptr<int64_t>(),
            target_sort_perm.data_ptr<int64_t>(),
            frac_floor.data_ptr<float>(),
            edge_src.data_ptr<int64_t>(),
            edge_tgt.data_ptr<int64_t>(),
            edge_cell_shift_out.data_ptr<float>(),
            edge_dist_sq.data_ptr<float>()
        );
    } else if (use_collect) {
        int gs = (N_target + WARPS_PER_BLOCK_COLLECT - 1) / WARPS_PER_BLOCK_COLLECT;
        int smem_bytes = WARPS_PER_BLOCK_COLLECT * MAX_NEIGHBORS_BUF_COLLECT
                         * (int)(sizeof(float) + sizeof(int));
        write_neighbors_collect_kernel<<<gs, BLOCK_SIZE_COLLECT, smem_bytes, stream>>>(
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
        write_neighbors_nosort_kernel<<<gs, BLOCK_SIZE, 0, stream>>>(
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
