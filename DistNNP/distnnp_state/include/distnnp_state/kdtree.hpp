#pragma once

#include <ATen/ATen.h>

// C++ entry point with input validation (calls internal CUDA implementation)
at::Tensor kdtree_bisect_cuda(at::Tensor pos,     // [N, 3] float32 CUDA
                              at::Tensor weights, // [N] or undefined
                              int64_t W           // number of partitions
);
