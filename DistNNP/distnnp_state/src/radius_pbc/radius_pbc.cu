// radius_pbc.cu — CUDA cell-list radius graph with selective PBC tiling
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

// Kernel B4: Write neighbor edges
__global__ void write_neighbors_kernel(
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

// ===== Dispatch Function =====

std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl(
    at::Tensor pos,       // [N, 3] float32 CUDA
    at::Tensor cell,      // [3, 3] float32
    at::Tensor pbc,       // [3] bool
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly
) {
    at::cuda::CUDAGuard device_guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = pos.size(0);

    // Handle empty input
    if (N == 0) {
        auto opts_long = at::TensorOptions().dtype(at::kLong).device(pos.device());
        auto opts_float = at::TensorOptions().dtype(at::kFloat).device(pos.device());
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long)
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
            at::zeros({1}, opts_long)
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

    // Assign target atoms to cells (compute their grid coords)
    auto target_cell_ids = at::empty({N}, opts_int);
    auto target_cell_x = at::empty({N}, opts_int);
    auto target_cell_y = at::empty({N}, opts_int);
    auto target_cell_z = at::empty({N}, opts_int);

    // Compute target cell coords on GPU via ATen (using wrapped positions)
    auto target_grid_coords = at::floor((pos_wrapped - grid_min_t.unsqueeze(0)) * grid_res_inv).to(at::kInt);
    // Clamp
    auto grid_dims_t = at::tensor({grid_nx, grid_ny, grid_nz}, opts_int);
    target_grid_coords = at::clamp(target_grid_coords, at::tensor({0}, opts_int), grid_dims_t.unsqueeze(0) - 1);

    target_cell_x = target_grid_coords.select(1, 0).contiguous();
    target_cell_y = target_grid_coords.select(1, 1).contiguous();
    target_cell_z = target_grid_coords.select(1, 2).contiguous();

    // B3: Count neighbors
    int padding_size = (int)std::floor(radius / grid_resolution) + 1;
    float radius_sq = radius * radius;

    auto neighbor_counts = at::empty({N}, opts_int);
    int grid_size_B3 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    count_neighbors_kernel<<<grid_size_B3, BLOCK_SIZE, 0, stream>>>(
        pos_wrapped.contiguous().data_ptr<float>(),
        sorted_source_pos.data_ptr<float>(),
        target_cell_x.data_ptr<int>(),
        target_cell_y.data_ptr<int>(),
        target_cell_z.data_ptr<int>(),
        sorted_source_orig_idx.data_ptr<int64_t>(),
        cell_start_t.data_ptr<int>(),
        cell_end_t.data_ptr<int>(),
        grid_nx, grid_ny, grid_nz,
        padding_size,
        radius_sq,
        N,
        neighbor_counts.data_ptr<int>()
    );

    // Prefix sum → neighbor_offsets
    auto neighbor_counts_long = neighbor_counts.to(at::kLong);
    auto neighbor_offsets = at::cumsum(neighbor_counts_long, 0) - neighbor_counts_long;
    int64_t total_edges = neighbor_counts_long.sum().item<int64_t>();

    if (total_edges == 0) {
        return std::make_tuple(
            at::empty({2, 0}, opts_long),
            at::empty({0, 3}, opts_float),
            at::zeros({1}, opts_long)
        );
    }

    // B4: Write neighbors
    auto edge_src = at::empty({total_edges}, opts_long);
    auto edge_tgt = at::empty({total_edges}, opts_long);
    auto edge_cell_shift_out = at::empty({total_edges, 3}, opts_float);
    auto edge_dist_sq = at::empty({total_edges}, opts_float);

    write_neighbors_kernel<<<grid_size_B3, BLOCK_SIZE, 0, stream>>>(
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
        padding_size,
        radius_sq,
        N,
        neighbor_offsets.data_ptr<int64_t>(),
        edge_src.data_ptr<int64_t>(),
        edge_tgt.data_ptr<int64_t>(),
        edge_cell_shift_out.data_ptr<float>(),
        edge_dist_sq.data_ptr<float>()
    );

    // ===== Stage C: Max Neighbors Mask =====
    // Using ATen ops (sort per-atom distances, top-K selection)

    auto num_neighbors_per_atom = neighbor_counts_long.clone();
    int64_t max_nbrs = num_neighbors_per_atom.max().item<int64_t>();

    at::Tensor num_neighbors_image = at::tensor({total_edges}, opts_long);

    if (max_num_neighbors > 0 && max_nbrs > max_num_neighbors) {
        // Build per-atom distance matrix for sorting
        auto num_atoms_scalar = at::tensor({N}, opts_long);

        // Clamp neighbor counts
        auto clamped_counts = at::clamp(num_neighbors_per_atom, /*min=*/0, /*max=*/max_num_neighbors);

        if (enforce_max_strictly) {
            // Strict: keep only closest max_num_neighbors per atom
            // For each atom, sort its edges by distance, keep top-K
            auto mask = at::ones({total_edges}, at::TensorOptions().dtype(at::kBool).device(pos.device()));

            // Create per-atom distance array: [num_atoms, max_nbrs] filled with inf
            auto dist_sort = at::full({N, max_nbrs}, std::numeric_limits<float>::infinity(), opts_float);

            // Fill in distances using neighbor_offsets
            // We use a scatter approach: for each edge, compute its local index within its atom
            auto atom_indices = edge_tgt; // target atom for each edge
            auto atom_local_idx = at::arange(total_edges, opts_long) - neighbor_offsets.index_select(0, atom_indices);

            // Write distances into the 2D array
            auto flat_indices = atom_indices * max_nbrs + atom_local_idx;
            dist_sort.view(-1).scatter_(0, flat_indices, edge_dist_sq.to(at::kFloat));

            // Sort distances per atom
            auto sort_result2 = at::sort(dist_sort, /*dim=*/1);
            auto sorted_dists = std::get<0>(sort_result2);
            auto sort_perm = std::get<1>(sort_result2);

            // For each atom, mark edges beyond max_num_neighbors as false
            // Map sort_perm back to global edge indices
            for (int64_t k = max_num_neighbors; k < max_nbrs; k++) {
                auto atom_mask = num_neighbors_per_atom.gt(max_num_neighbors);
                auto col = sort_perm.select(1, k);
                auto global_idx = neighbor_offsets + col;
                // Only mask atoms that actually exceed the limit
                auto valid = atom_mask.logical_and(sorted_dists.select(1, k).isfinite());
                global_idx = global_idx.masked_select(valid);
                mask.index_fill_(0, global_idx, false);
            }

            edge_src = edge_src.masked_select(mask);
            edge_tgt = edge_tgt.masked_select(mask);
            edge_cell_shift_out = edge_cell_shift_out.index_select(
                0, mask.nonzero().squeeze(1));
            edge_dist_sq = edge_dist_sq.masked_select(mask);
            total_edges = edge_src.size(0);

            num_neighbors_per_atom = clamped_counts;
        } else {
            // Non-strict: allow degeneracy tolerance (similar to v2)
            float degeneracy_tolerance = 0.01f;

            auto mask = at::ones({total_edges}, at::TensorOptions().dtype(at::kBool).device(pos.device()));

            auto dist_sort = at::full({N, max_nbrs}, std::numeric_limits<float>::infinity(), opts_float);
            auto atom_indices = edge_tgt;
            auto atom_local_idx = at::arange(total_edges, opts_long) - neighbor_offsets.index_select(0, atom_indices);
            auto flat_indices = atom_indices * max_nbrs + atom_local_idx;
            dist_sort.view(-1).scatter_(0, flat_indices, edge_dist_sq.to(at::kFloat));

            auto sort_result2 = at::sort(dist_sort, /*dim=*/1);
            auto sorted_dists = std::get<0>(sort_result2);
            auto sort_perm = std::get<1>(sort_result2);

            // Effective cutoff = distance of (max_num_neighbors+1)-th neighbor + tolerance
            auto effective_cutoff = sorted_dists.select(1, max_num_neighbors) + degeneracy_tolerance;

            // Mark edges where distance > effective_cutoff
            for (int64_t k = max_num_neighbors; k < max_nbrs; k++) {
                auto col_dist = sorted_dists.select(1, k);
                auto exceeds = col_dist.gt(effective_cutoff);
                auto col = sort_perm.select(1, k);
                auto global_idx = neighbor_offsets + col;
                auto valid = exceeds.logical_and(col_dist.isfinite());
                global_idx = global_idx.masked_select(valid);
                mask.index_fill_(0, global_idx, false);
            }

            edge_src = edge_src.masked_select(mask);
            edge_tgt = edge_tgt.masked_select(mask);
            edge_cell_shift_out = edge_cell_shift_out.index_select(
                0, mask.nonzero().squeeze(1));
            edge_dist_sq = edge_dist_sq.masked_select(mask);
            total_edges = edge_src.size(0);
        }
    }

    // ===== Correct cell shifts for position wrapping =====
    // The algorithm ran on wrapped positions. The cell shifts are relative to
    // the wrapped frame. To get shifts relative to the ORIGINAL positions:
    //   cell_shift_original = cell_shift_wrapped + frac_floor[source] - frac_floor[target]
    if (total_edges > 0) {
        auto src_frac_floor = frac_floor.index_select(0, edge_src);  // [E, 3]
        auto tgt_frac_floor = frac_floor.index_select(0, edge_tgt);  // [E, 3]
        edge_cell_shift_out = edge_cell_shift_out + tgt_frac_floor - src_frac_floor;
    }

    // Build edge_index [2, E] = [[source_idx...], [target_idx...]]
    auto edge_index = at::stack({edge_src, edge_tgt}, 0);

    // num_neighbors: total edge count
    num_neighbors_image = at::tensor({total_edges}, opts_long);

    return std::make_tuple(edge_index, edge_cell_shift_out, num_neighbors_image);
}
