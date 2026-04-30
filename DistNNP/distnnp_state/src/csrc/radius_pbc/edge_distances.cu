#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

__global__ void edge_distances_cuda_impl(
    const float* __restrict__ pos,               // [N, 3]
    const int64_t* __restrict__ edge_index,      // [2, E]
    const float* __restrict__ edge_cell_shift,   // [E, 3]
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

    // Apply PBC shifts if provided
    if (edge_cell_shift != nullptr) {
        dx += edge_cell_shift[idx * 3 + 0];
        dy += edge_cell_shift[idx * 3 + 1];
        dz += edge_cell_shift[idx * 3 + 2];
    }

    edge_distance_vec[idx * 3 + 0] = dx;
    edge_distance_vec[idx * 3 + 1] = dy;
    edge_distance_vec[idx * 3 + 2] = dz;

    edge_distance[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}
