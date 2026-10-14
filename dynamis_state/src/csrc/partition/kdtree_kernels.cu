// kdtree_kernels.cu — Level-by-level batched KD-tree bisection on GPU.
//
// Replaces the recursive CPU kdtree_bisect with iterative breadth-first
// bisection. All work stays on GPU — 0 .item() calls in the main loop.
//
// Sort strategy:
//   Level 0 (S=1): sort float32 directly (one global segment, no packing needed)
//   Level l (S>1): pack (seg_id << 32 | float_sort_bits(coord)) into int64,
//                  one global radix sort achieves both grouping + within-segment ordering.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <utility>

static constexpr int BLOCK = 256;

// ---------------------------------------------------------------------------
// Utility: float → sortable uint32 (preserves ordering as unsigned int)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t mask = (bits >> 31) ? 0xFFFFFFFF : 0x80000000;
    return bits ^ mask;
}

__device__ __forceinline__ float sortable_to_float(int32_t bits_i) {
    uint32_t bits;
    memcpy(&bits, &bits_i, sizeof(bits));
    uint32_t mask = (bits >> 31) ? 0x80000000 : 0xFFFFFFFF;
    bits = bits ^ mask;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// K1: Per-segment bounding box via atomicMin/Max on float-as-int
// ---------------------------------------------------------------------------
__global__ void compute_seg_bbox_kernel(
    const float* __restrict__ pos,
    const int32_t* __restrict__ seg_id,
    int32_t* __restrict__ seg_min_bits,
    int32_t* __restrict__ seg_max_bits,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t s = seg_id[i];
    for (int d = 0; d < 3; d++) {
        float v = pos[i * 3 + d];
        int32_t bits;
        uint32_t sb = float_to_sortable(v);
        memcpy(&bits, &sb, sizeof(bits));
        atomicMin(&seg_min_bits[s * 3 + d], bits);
        atomicMax(&seg_max_bits[s * 3 + d], bits);
    }
}

// ---------------------------------------------------------------------------
// K2: Per-atom: select coordinate along longest axis of its segment.
// ---------------------------------------------------------------------------
__global__ void compute_axis_and_coord_kernel(
    const float* __restrict__ pos,
    const int32_t* __restrict__ seg_id,
    const int32_t* __restrict__ seg_min_bits,
    const int32_t* __restrict__ seg_max_bits,
    float* __restrict__ sort_coord,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t s = seg_id[i];
    float best_ext = -1.0f;
    int best_d = 0;
    for (int d = 0; d < 3; d++) {
        float lo = sortable_to_float(seg_min_bits[s * 3 + d]);
        float hi = sortable_to_float(seg_max_bits[s * 3 + d]);
        float ext = hi - lo;
        if (ext > best_ext) {
            best_ext = ext;
            best_d = d;
        }
    }
    sort_coord[i] = pos[i * 3 + best_d];
}

// ---------------------------------------------------------------------------
// K3: Pack composite sort key: (seg_id << 32) | float_sort_bits(coord)
// ---------------------------------------------------------------------------
__global__ void pack_sort_key_kernel(
    const int32_t* __restrict__ seg_id,
    const float* __restrict__ sort_coord,
    int64_t* __restrict__ packed_key,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t coord_bits = float_to_sortable(sort_coord[i]);
    int64_t key = ((int64_t)(uint32_t)seg_id[i] << 32) | (int64_t)coord_bits;
    packed_key[i] = key;
}

// ---------------------------------------------------------------------------
// K4: Build inverse permutation: inv_perm[perm[i]] = i
// ---------------------------------------------------------------------------
__global__ void build_inv_perm_kernel(
    const int64_t* __restrict__ perm,
    int64_t* __restrict__ inv_perm,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    inv_perm[perm[i]] = i;
}

// ---------------------------------------------------------------------------
// K5: Unweighted split: mid = size / 2, clamped to [1, size-1]
// ---------------------------------------------------------------------------
__global__ void compute_split_unweighted_kernel(
    const int64_t* __restrict__ seg_sizes,
    const int32_t* __restrict__ seg_parts,
    int64_t* __restrict__ mid,
    int64_t S
) {
    int64_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    int32_t parts = seg_parts[s];
    int64_t sz = seg_sizes[s];
    if (parts <= 1 || sz <= 1) {
        mid[s] = sz;
    } else {
        int64_t m = sz / 2;
        m = max(m, (int64_t)1);
        m = min(m, sz - 1);
        mid[s] = m;
    }
}

// ---------------------------------------------------------------------------
// K6: Weighted split via cumsum + binary search.
//     Reads the global prefix sum of sorted weights; computes per-segment
//     total and finds the balanced split point via binary search.
// ---------------------------------------------------------------------------
__global__ void find_weighted_split_kernel(
    const float* __restrict__ cumsum_w,
    const int64_t* __restrict__ seg_starts,
    const int64_t* __restrict__ seg_sizes,
    const int32_t* __restrict__ seg_parts,
    int64_t* __restrict__ mid,
    int64_t S
) {
    int64_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    int32_t parts = seg_parts[s];
    int64_t sz = seg_sizes[s];
    if (parts <= 1 || sz <= 1) {
        mid[s] = sz;
        return;
    }

    int64_t start = seg_starts[s];
    float base = (start > 0) ? cumsum_w[start - 1] : 0.0f;
    float seg_total = cumsum_w[start + sz - 1] - base;
    float target = base + seg_total * 0.5f;

    // Binary search: first j where cumsum_w[start + j] >= target
    int64_t lo = 0, hi = sz - 1;
    while (lo < hi) {
        int64_t m = (lo + hi) / 2;
        if (cumsum_w[start + m] < target) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    lo = max(lo, (int64_t)1);
    lo = min(lo, sz - 1);
    mid[s] = lo;
}

// ---------------------------------------------------------------------------
// K7: Assign new segment IDs after split
// ---------------------------------------------------------------------------
__global__ void assign_new_seg_ids_kernel(
    const int64_t* __restrict__ inv_perm,
    const int32_t* __restrict__ seg_id,
    const int64_t* __restrict__ seg_starts,
    const int64_t* __restrict__ mid,
    const int32_t* __restrict__ seg_child_left,
    const int32_t* __restrict__ seg_child_right,
    int32_t* __restrict__ new_seg_id,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t s = seg_id[i];
    int64_t pos_in_sorted = inv_perm[i];
    int64_t pos_in_seg = pos_in_sorted - seg_starts[s];
    if (pos_in_seg < mid[s]) {
        new_seg_id[i] = seg_child_left[s];
    } else {
        new_seg_id[i] = seg_child_right[s];
    }
}

// ---------------------------------------------------------------------------
// K8: Map final segment IDs to partition labels
// ---------------------------------------------------------------------------
__global__ void map_seg_to_label_kernel(
    const int32_t* __restrict__ seg_id,
    const int64_t* __restrict__ seg_base_label,
    int64_t* __restrict__ labels,
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    labels[i] = seg_base_label[seg_id[i]];
}

// ===========================================================================
// Host: Build partition tree for non-power-of-two W
// ===========================================================================

static int ceil_log2(int64_t x) {
    if (x <= 1) return 0;
    int l = 0;
    int64_t v = 1;
    while (v < x) { v <<= 1; l++; }
    return l;
}

struct LevelInfo {
    at::Tensor seg_parts;       // [S] int32
    at::Tensor seg_child_left;  // [S] int32
    at::Tensor seg_child_right; // [S] int32
    int S;
};

// Build tree and upload all level data to GPU in one pass.
static std::pair<std::vector<LevelInfo>, at::Tensor>
build_and_upload_tree(int64_t W, int L, at::Device device) {
    auto int_opts = at::TensorOptions().dtype(at::kInt);
    auto long_opts = at::TensorOptions().dtype(at::kLong);

    std::vector<LevelInfo> levels(L);
    std::vector<int32_t> cur_p = {(int32_t)W};

    for (int l = 0; l < L; l++) {
        int S = (int)cur_p.size();
        std::vector<int32_t> cl(S), cr(S);
        std::vector<int32_t> next_p;
        int32_t next_id = 0;

        for (int s = 0; s < S; s++) {
            if (cur_p[s] <= 1) {
                cl[s] = next_id;
                cr[s] = next_id;
                next_p.push_back(cur_p[s]);
                next_id++;
            } else {
                int32_t lp = cur_p[s] / 2;
                int32_t rp = cur_p[s] - lp;
                cl[s] = next_id;
                next_p.push_back(lp);
                next_id++;
                cr[s] = next_id;
                next_p.push_back(rp);
                next_id++;
            }
        }

        // Upload to GPU (from_blob + clone ensures data is owned before vector goes out of scope)
        auto t_parts = at::from_blob(cur_p.data(), {S}, int_opts).clone().to(device);
        auto t_cl = at::from_blob(cl.data(), {S}, int_opts).clone().to(device);
        auto t_cr = at::from_blob(cr.data(), {S}, int_opts).clone().to(device);
        levels[l] = {t_parts, t_cl, t_cr, S};

        cur_p = next_p;
    }

    // Final base labels
    int64_t label = 0;
    std::vector<int64_t> base_labels(cur_p.size());
    for (size_t s = 0; s < cur_p.size(); s++) {
        base_labels[s] = label;
        label += cur_p[s];
    }
    auto gpu_base_label = at::from_blob(base_labels.data(),
        {(int64_t)base_labels.size()}, long_opts).clone().to(device);

    return {std::move(levels), gpu_base_label};
}

// ===========================================================================
// Main CUDA implementation
// ===========================================================================
at::Tensor kdtree_bisect_cuda_impl(
    at::Tensor pos,       // [N, 3] float32 CUDA contiguous
    at::Tensor weights,   // [N] float32 CUDA contiguous, or undefined
    int64_t W
) {
    c10::cuda::CUDAGuard guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = pos.device();

    int64_t N = pos.size(0);

    if (W == 1) {
        return at::zeros({N}, at::TensorOptions().dtype(at::kLong).device(device));
    }

    int L = ceil_log2(W);

    auto int_opts = at::TensorOptions().dtype(at::kInt).device(device);
    auto long_opts = at::TensorOptions().dtype(at::kLong).device(device);
    auto float_opts = at::TensorOptions().dtype(at::kFloat).device(device);

    // Upload all tree data to GPU once (O(W) total, tiny)
    auto [levels, gpu_base_label] = build_and_upload_tree(W, L, device);

    // Max segment count across all levels (for pre-allocation)
    int max_S = 1;
    for (int l = 0; l < L; l++)
        if (levels[l].S > max_S) max_S = levels[l].S;

    // Pre-allocate ALL working tensors outside the loop — avoids cudaMalloc per level.
    // N-sized buffers:
    auto seg_id = at::zeros({N}, int_opts);
    auto new_seg_id = at::empty({N}, int_opts);
    auto sort_coord = at::empty({N}, float_opts);
    auto packed_key = at::empty({N}, long_opts);
    auto inv_perm = at::empty({N}, long_opts);
    auto seg_id_long = at::empty({N}, long_opts);   // reusable cast buffer

    // S-sized buffers (sized to max_S):
    auto seg_min_bits = at::empty({max_S * 3}, int_opts);
    auto seg_max_bits = at::empty({max_S * 3}, int_opts);
    auto seg_sizes = at::empty({max_S}, long_opts);
    auto seg_starts = at::zeros({max_S + 1}, long_opts);
    auto mid = at::empty({max_S}, long_opts);

    int blocks_N = (N + BLOCK - 1) / BLOCK;

    for (int l = 0; l < L; l++) {
        auto& lv = levels[l];
        int S = lv.S;

        // K1: Bounding box — fill sentinel values for active segments only
        auto min_slice = seg_min_bits.slice(0, 0, S * 3);
        auto max_slice = seg_max_bits.slice(0, 0, S * 3);
        min_slice.fill_(INT32_MAX);
        max_slice.fill_(INT32_MIN);

        compute_seg_bbox_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            pos.data_ptr<float>(),
            seg_id.data_ptr<int32_t>(),
            seg_min_bits.data_ptr<int32_t>(),
            seg_max_bits.data_ptr<int32_t>(),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // K2: Axis + coordinate selection
        compute_axis_and_coord_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            pos.data_ptr<float>(),
            seg_id.data_ptr<int32_t>(),
            seg_min_bits.data_ptr<int32_t>(),
            seg_max_bits.data_ptr<int32_t>(),
            sort_coord.data_ptr<float>(),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Sort: level 0 has a single segment — sort float32 directly.
        // Later levels need packed int64 key for grouping + within-segment ordering.
        at::Tensor perm;
        if (S == 1) {
            perm = std::get<1>(at::sort(sort_coord));
        } else {
            pack_sort_key_kernel<<<blocks_N, BLOCK, 0, stream>>>(
                seg_id.data_ptr<int32_t>(),
                sort_coord.data_ptr<float>(),
                packed_key.data_ptr<int64_t>(),
                N
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            perm = std::get<1>(at::sort(packed_key));
        }

        // K4: Build inverse permutation
        build_inv_perm_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            perm.data_ptr<int64_t>(),
            inv_perm.data_ptr<int64_t>(),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Segment sizes from unsorted seg_id (reuse pre-allocated buffers)
        seg_id_long.copy_(seg_id);
        auto sizes_slice = seg_sizes.slice(0, 0, S);
        sizes_slice.copy_(at::bincount(seg_id_long, {}, S));

        auto starts_slice = seg_starts.slice(0, 0, S + 1);
        starts_slice.slice(0, 1, S + 1).copy_(at::cumsum(sizes_slice, 0));

        // Compute split points
        auto mid_slice = mid.slice(0, 0, S);

        if (weights.defined()) {
            // Gather sorted weights + parallel prefix sum
            auto sorted_w = weights.index_select(0, perm);
            auto cumsum_w = sorted_w.cumsum(0);

            int blocks_S = (S + BLOCK - 1) / BLOCK;
            find_weighted_split_kernel<<<blocks_S, BLOCK, 0, stream>>>(
                cumsum_w.data_ptr<float>(),
                seg_starts.data_ptr<int64_t>(),
                seg_sizes.data_ptr<int64_t>(),
                lv.seg_parts.data_ptr<int32_t>(),
                mid.data_ptr<int64_t>(),
                S
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            int blocks_S = (S + BLOCK - 1) / BLOCK;
            compute_split_unweighted_kernel<<<blocks_S, BLOCK, 0, stream>>>(
                seg_sizes.data_ptr<int64_t>(),
                lv.seg_parts.data_ptr<int32_t>(),
                mid.data_ptr<int64_t>(),
                S
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // K7: Assign new segment IDs
        assign_new_seg_ids_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            inv_perm.data_ptr<int64_t>(),
            seg_id.data_ptr<int32_t>(),
            seg_starts.data_ptr<int64_t>(),
            mid.data_ptr<int64_t>(),
            lv.seg_child_left.data_ptr<int32_t>(),
            lv.seg_child_right.data_ptr<int32_t>(),
            new_seg_id.data_ptr<int32_t>(),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        std::swap(seg_id, new_seg_id);
    }

    // Map final segment IDs to partition labels
    auto labels = at::empty({N}, long_opts);

    map_seg_to_label_kernel<<<blocks_N, BLOCK, 0, stream>>>(
        seg_id.data_ptr<int32_t>(),
        gpu_base_label.data_ptr<int64_t>(),
        labels.data_ptr<int64_t>(),
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return labels;
}
