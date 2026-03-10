// sfc_cuda.cu — CUDA kernels for SFC (Space-Filling Curve) partitioning.
//
// Kernels:
//   K1: compute_morton_keys_kernel   — normalize + 21-bit quantize + bit-spread interleave
//   K2: compute_hilbert_keys_kernel  — normalize + order=7 quantize + 7-iteration transform
//   K3: assign_partition_unweighted  — O(1) closed-form partition from sorted position
//   K4: assign_partition_weighted    — cumsum-based partition from sorted weights

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

static constexpr int BLOCK = 256;

// ---------------------------------------------------------------------------
// Morton: magic-number bit spread (O(1) per coordinate, 5 shift+mask ops)
// Spreads 21-bit value into every-3rd-bit positions of a 63-bit result.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t spread_bits_21(uint32_t v) {
    uint64_t x = v & 0x1FFFFF;  // mask to 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x <<  8)) & 0x100F00F00F00F00FULL;
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x <<  2)) & 0x1249249249249249ULL;
    return x;
}

// ---------------------------------------------------------------------------
// K1: Compute Morton (Z-curve) keys — 21-bit quantization, 63-bit interleaved key
// ---------------------------------------------------------------------------
__global__ void compute_morton_keys_kernel(
    const float* __restrict__ pos,       // [N, 3]
    const float* __restrict__ pos_min,   // [3]
    const float* __restrict__ pos_range, // [3]
    int64_t* __restrict__ keys,          // [N]
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Normalize to [0, 1]
    float nx = (pos[i * 3 + 0] - pos_min[0]) / pos_range[0];
    float ny = (pos[i * 3 + 1] - pos_min[1]) / pos_range[1];
    float nz = (pos[i * 3 + 2] - pos_min[2]) / pos_range[2];

    // Quantize to 21-bit grid
    const uint32_t grid_max = (1u << 21) - 1;
    uint32_t ix = min((uint32_t)(nx * grid_max), grid_max);
    uint32_t iy = min((uint32_t)(ny * grid_max), grid_max);
    uint32_t iz = min((uint32_t)(nz * grid_max), grid_max);

    // Interleave: key = spread(x) | (spread(y) << 1) | (spread(z) << 2)
    uint64_t key = spread_bits_21(ix) | (spread_bits_21(iy) << 1) | (spread_bits_21(iz) << 2);
    keys[i] = static_cast<int64_t>(key);
}

// ---------------------------------------------------------------------------
// K2: Compute Hilbert keys — order=7, 7-iteration bit transformation
// Matches baseline hilbert_curve_index_tensor logic.
// ---------------------------------------------------------------------------
__global__ void compute_hilbert_keys_kernel(
    const float* __restrict__ pos,       // [N, 3]
    const float* __restrict__ pos_min,   // [3]
    const float* __restrict__ pos_range, // [3]
    int64_t* __restrict__ keys,          // [N]
    int64_t N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int order = 7;
    const int grid_max = (1 << order) - 1;  // 127

    // Normalize to [0, 1]
    float nx = (pos[i * 3 + 0] - pos_min[0]) / pos_range[0];
    float ny = (pos[i * 3 + 1] - pos_min[1]) / pos_range[1];
    float nz = (pos[i * 3 + 2] - pos_min[2]) / pos_range[2];

    // Quantize to order-bit grid
    int x = min(max((int)(nx * grid_max), 0), grid_max);
    int y = min(max((int)(ny * grid_max), 0), grid_max);
    int z = min(max((int)(nz * grid_max), 0), grid_max);

    // 7-iteration Hilbert transform (matching baseline)
    int64_t result = 0;
    for (int b = 0; b < order; b++) {
        int xb = (x >> b) & 1;
        int yb = (y >> b) & 1;
        int zb = (z >> b) & 1;

        int new_xb, new_yb, new_zb;
        if (b == 0) {
            new_xb = xb;
            new_yb = yb;
            new_zb = zb;
        } else {
            // Apply Hilbert curve transformation (matching baseline)
            new_xb = xb ? zb : xb;
            int tmp_yb = xb ? yb : (yb ? zb : yb);
            new_zb = xb ? yb : (yb ? xb : (zb ? yb : zb));
            new_yb = tmp_yb;
        }
        result |= ((int64_t)new_xb << (3 * b))
                | ((int64_t)new_yb << (3 * b + 1))
                | ((int64_t)new_zb << (3 * b + 2));
    }
    keys[i] = result;
}

// ---------------------------------------------------------------------------
// K3: Assign partitions (unweighted) — O(1) closed-form per sorted position
// ---------------------------------------------------------------------------
__global__ void assign_partition_unweighted_kernel(
    const int64_t* __restrict__ perm,   // [N] sort permutation
    int64_t* __restrict__ labels,       // [N] output partition labels
    int64_t N,
    int64_t W
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Closed-form: first `rem` partitions get (base+1) atoms, rest get base
    int64_t base = N / W;
    int64_t rem  = N % W;
    int64_t threshold = rem * (base + 1);

    int64_t p;
    if (i < threshold) {
        p = i / (base + 1);
    } else {
        p = rem + (i - threshold) / base;
    }
    p = min(p, W - 1);
    labels[perm[i]] = p;
}

// ---------------------------------------------------------------------------
// K4: Assign partitions (weighted) — cumsum-based, 0 CPU-GPU sync
// ---------------------------------------------------------------------------
__global__ void assign_partition_weighted_kernel(
    const int64_t* __restrict__ perm,      // [N] sort permutation
    const float* __restrict__ cumsum_w,    // [N] prefix sum of sorted weights
    int64_t* __restrict__ labels,          // [N] output partition labels
    int64_t N,
    int64_t W
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Read total cost from last element (on device, no sync)
    float total = cumsum_w[N - 1];
    float cs = cumsum_w[i];

    int64_t p = (int64_t)(cs * (float)W / total);
    // Clamp to [0, W-1]
    p = max(p, (int64_t)0);
    p = min(p, W - 1);
    labels[perm[i]] = p;
}

// ===========================================================================
// Host: SFC partition CUDA implementation
// ===========================================================================
at::Tensor sfc_partition_cuda_impl(
    at::Tensor pos,       // [N, 3] float32 CUDA contiguous
    at::Tensor weights,   // [N] float32 CUDA contiguous, or undefined
    int64_t W,
    int64_t curve_type    // 0=Morton, 1=Hilbert
) {
    c10::cuda::CUDAGuard guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = pos.device();

    int64_t N = pos.size(0);
    auto long_opts = at::TensorOptions().dtype(at::kLong).device(device);

    if (W == 1) {
        return at::zeros({N}, long_opts);
    }

    // 1. Global bounds (ATen min/max reductions)
    auto pos_min = std::get<0>(pos.min(0));       // [3]
    auto pos_range = std::get<0>(pos.max(0)) - pos_min;  // [3]
    pos_range = at::where(pos_range > 0, pos_range, at::ones_like(pos_range));

    // 2. Compute SFC keys
    auto sfc_keys = at::empty({N}, long_opts);
    int blocks_N = (N + BLOCK - 1) / BLOCK;

    if (curve_type == 0) {
        compute_morton_keys_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            pos.data_ptr<float>(),
            pos_min.data_ptr<float>(),
            pos_range.data_ptr<float>(),
            sfc_keys.data_ptr<int64_t>(),
            N
        );
    } else {
        compute_hilbert_keys_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            pos.data_ptr<float>(),
            pos_min.data_ptr<float>(),
            pos_range.data_ptr<float>(),
            sfc_keys.data_ptr<int64_t>(),
            N
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 3. Sort by SFC key → get permutation
    auto perm = std::get<1>(at::sort(sfc_keys));

    // 4. Assign partitions
    auto labels = at::empty({N}, long_opts);

    if (weights.defined()) {
        // Weighted: cumsum of sorted weights
        auto sorted_w = weights.index_select(0, perm);
        auto cumsum_w = sorted_w.cumsum(0);

        assign_partition_weighted_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            perm.data_ptr<int64_t>(),
            cumsum_w.data_ptr<float>(),
            labels.data_ptr<int64_t>(),
            N, W
        );
    } else {
        assign_partition_unweighted_kernel<<<blocks_N, BLOCK, 0, stream>>>(
            perm.data_ptr<int64_t>(),
            labels.data_ptr<int64_t>(),
            N, W
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return labels;
}
