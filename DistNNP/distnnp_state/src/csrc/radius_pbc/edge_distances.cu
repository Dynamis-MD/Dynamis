#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

static constexpr int BLOCK_SIZE = 256;

__global__ void compute_edge_distances_kernel(
    const float* __restrict__ pos,               // [N, 3]
    const int64_t* __restrict__ edge_index,      // [2, E]
    float* __restrict__ edge_distance_vec,       // [E, 3]
    float* __restrict__ edge_distance,           // [E]
    int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    int64_t src = edge_index[idx];
    int64_t tgt = edge_index[E + idx]; // row 2 (flattened)

    // Vector from source to target
    float dx = pos[tgt * 3 + 0] - pos[src * 3 + 0];
    float dy = pos[tgt * 3 + 1] - pos[src * 3 + 1];
    float dz = pos[tgt * 3 + 2] - pos[src * 3 + 2];

    edge_distance_vec[idx * 3 + 0] = dx;
    edge_distance_vec[idx * 3 + 1] = dy;
    edge_distance_vec[idx * 3 + 2] = dz;

    edge_distance[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

std::tuple<at::Tensor, at::Tensor> edge_distances_cuda_impl(
    at::Tensor pos,               // [N, 3] float32
    at::Tensor edge_index        // [2, E] int64
) {
    // Ensure inputs are on the correct device
    c10::cuda::CUDAGuard device_guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    
    int E = edge_index.size(1);

    // Prepare output tensors
    auto opts_float = at::TensorOptions().dtype(at::kFloat).device(pos.device());
    auto edge_distance_vec = at::empty({E, 3}, opts_float);
    auto edge_distance = at::empty({E}, opts_float);

    if (E == 0) {
        return std::make_tuple(edge_distance_vec, edge_distance);
    }

    int grid_size = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;

compute_edge_distances_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        pos.data_ptr<float>(),
        edge_index.data_ptr<int64_t>(),
        edge_distance_vec.data_ptr<float>(),
        edge_distance.data_ptr<float>(),
        E
    );

    return std::make_tuple(edge_distance_vec, edge_distance);
}
