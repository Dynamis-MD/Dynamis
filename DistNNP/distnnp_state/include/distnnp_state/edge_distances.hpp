#pragma once

#include <ATen/ATen.h>
#include <tuple>
#include <stdint.h>

void compute_edge_distances_kernel(
    const float* __restrict__ pos,               // [N, 3]
    const int64_t* __restrict__ edge_index,      // [2, E]
    float* __restrict__ edge_distance_vec,       // [E, 3]
    float* __restrict__ edge_distance,           // [E]
    int E
);

std::tuple<at::Tensor, at::Tensor> edge_distances_cuda_impl(
    at::Tensor pos,               // [N, 3] float32
    at::Tensor edge_index        // [2, E] int64
);

std::tuple<at::Tensor, at::Tensor> edge_distances(
    at::Tensor pos,               // [N, 3] float32
    at::Tensor edge_index        // [2, E] int64
);
