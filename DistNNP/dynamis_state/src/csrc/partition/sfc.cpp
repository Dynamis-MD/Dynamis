// sfc.cpp — C++ entry point with input validation for CUDA SFC partition

#include "dynamis_state/sfc.hpp"

// Forward declaration of internal CUDA implementation (defined in sfc_kernels.cu)
at::Tensor sfc_partition_cuda_impl(at::Tensor pos, at::Tensor weights, int64_t W,
                                   int64_t curve_type);

at::Tensor sfc_partition_cuda(at::Tensor pos, at::Tensor weights, int64_t W, int64_t curve_type)
{
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos must have shape [N, 3], got [",
                pos.size(0), ", ", pos.size(1), "]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat, "pos must be float32, got ", pos.scalar_type());
    pos = pos.contiguous();

    TORCH_CHECK(W >= 1, "W must be >= 1, got ", W);
    TORCH_CHECK(curve_type == 0 || curve_type == 1,
                "curve_type must be 0 (Morton) or 1 (Hilbert), got ", curve_type);

    if (weights.defined()) {
        TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
        TORCH_CHECK(weights.dim() == 1 && weights.size(0) == pos.size(0),
                    "weights must have shape [N], got [", weights.size(0), "]");
        // Keep weights as exact integers: the weighted partition uses an int64
        // prefix sum so cut boundaries are deterministic and identical on every
        // rank. A float32 cumsum rounds above 2^24 and made boundary atoms flip
        // partitions per-rank → duplicate/missing edges.
        weights = weights.to(at::kLong).contiguous();
    }

    return sfc_partition_cuda_impl(pos, weights, W, curve_type);
}
