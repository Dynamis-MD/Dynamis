#include "distnnp_model/macros.hpp"
#include <torch/extension.h>

__global__ void push_buffer_gather_kernel(const float* input, float* buffer,
                    const int64_t* input_locs, int num_indices, int feature_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_indices && col < feature_dim) {
        int64_t src_row = input_locs[row];
        buffer[row * feature_dim + col] = input[src_row * feature_dim + col];
    }
}

// Optimized version for when feature_dim is divisible by 4
__global__ void push_buffer_gather_kernel_vec4(const float* input, float* buffer,
                    const int64_t* input_locs, int num_indices, int feature_dim_div4) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_indices && col < feature_dim_div4) {
        int64_t src_row = input_locs[row];

        // Advance to the start of the row in float units (feature_dim_div4 * 4 = feature_dim),
        // then reinterpret as float4* so each index covers 4 consecutive floats
        const float4* in_row = reinterpret_cast<const float4*>(input + src_row * feature_dim_div4 * 4);
        float4* out_row = reinterpret_cast<float4*>(buffer + row * feature_dim_div4 * 4);

        // Single float4 load/store moves 16 bytes instead of 4
        out_row[col] = in_row[col];
    }
}

namespace distnnp_model
{

at::Tensor push_buffer_create(at::Tensor input, at::Tensor buffer, at::Tensor input_locs) {
    // Base input checks
    CHECK_INPUT(input);
    CHECK_INPUT(buffer);
    CHECK_INPUT(input_locs);
    TORCH_CHECK(input.dim() == 2, "input must have 2 dimensions");
    TORCH_CHECK(buffer.dim() == 2, "buffer must have 2 dimensions");
    TORCH_CHECK(input_locs.dim() == 1, "input_locs must have 1 dimension");
    TORCH_CHECK(buffer.size(0) == input_locs.size(0), "buffer rows must match number of indices");
    TORCH_CHECK(buffer.size(1) == input.size(1), "buffer and input must have same feature dimension");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input datatype must be float32");
    TORCH_CHECK(buffer.scalar_type() == at::kFloat, "buffer datatype must be float32");
    TORCH_CHECK(input_locs.scalar_type() == at::kLong, "input_locs datatype must be int64");

    // Obtain dims
    int num_indices = input_locs.size(0);
    int feature_dim = input.size(1);

    // Grid dims capped at 65535 (CUDA hardware limit)
    TORCH_CHECK(num_indices <= 65535 * 32, "num_indices exceeds max kernel coverage");
    TORCH_CHECK(feature_dim <= 65535 * 32, "feature_dim exceeds max kernel coverage");

    // Check for early return
    if (num_indices == 0) {
        return buffer;
    }

    // Claim CUDA device
    at::cuda::CUDAGuard device_guard(input.device());

    // obtain CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream();

    // Extract raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* buffer_ptr = buffer.data_ptr<float>();
    const int64_t* input_locs_ptr = input_locs.data_ptr<int64_t>();

    // Check if we can take the vec4 path
    bool vec4_path = (feature_dim % 4 == 0);

    // 32x32 threads per block, enough blocks to cover all rows and columns
    dim3 block_dims, grid_dims;
    block_dims.x = 32;
    block_dims.y = 32;
    block_dims.z = 1;

    grid_dims.x = vec4_path ? // For the vec4 path
        (feature_dim / 4 + block_dims.x - 1) / block_dims.x : (feature_dim + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (num_indices + block_dims.y - 1) / block_dims.y;
    grid_dims.z = 1;

    // Launch kernel
    if (vec4_path) {
        push_buffer_gather_kernel_vec4<<<grid_dims, block_dims, 0, stream>>>(
            input_ptr, buffer_ptr, input_locs_ptr, num_indices, feature_dim / 4);
    } else {
        push_buffer_gather_kernel<<<grid_dims, block_dims, 0, stream>>>(
            input_ptr, buffer_ptr, input_locs_ptr, num_indices, feature_dim);
    }

    // Check for launch errors
    CUDACHECK(cudaGetLastError());

    // Return final buffer
    return buffer;
}

} // namespace distnnp_model