#pragma once

#include <ATen/ATen.h>

// C++ entry point with input validation (calls internal CUDA implementation)
// curve_type: 0 = Morton, 1 = Hilbert
at::Tensor sfc_partition_cuda(
    at::Tensor pos,       // [N, 3] float32 CUDA
    at::Tensor weights,   // [N]
    int64_t W,            // number of partitions
    int64_t curve_type    // 0=Morton, 1=Hilbert
);
