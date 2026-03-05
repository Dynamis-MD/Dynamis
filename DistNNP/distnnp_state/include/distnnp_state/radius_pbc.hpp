#pragma once

#include <ATen/ATen.h>
#include <tuple>

// CUDA implementation (defined in radius_pbc.cu)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl(
    at::Tensor pos,
    at::Tensor cell,
    at::Tensor pbc,
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly
);

// C++ entry point with input validation
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc(
    at::Tensor pos,       // [N, 3] float32 CUDA
    at::Tensor cell,      // [3, 3] float32
    at::Tensor pbc,       // [3] bool
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly
);
