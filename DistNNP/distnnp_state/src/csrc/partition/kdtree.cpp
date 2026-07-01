// kdtree.cpp — C++ entry point with input validation for CUDA KD-tree

#include "distnnp_state/kdtree.hpp"

// Forward declaration of internal CUDA implementation (defined in kdtree_kernels.cu)
at::Tensor kdtree_bisect_cuda_impl(at::Tensor pos, at::Tensor weights, int64_t W);

at::Tensor kdtree_bisect_cuda(
    at::Tensor pos,
    at::Tensor weights,
    int64_t W
) {
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3,
                "pos must have shape [N, 3], got [", pos.size(0), ", ", pos.size(1), "]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat,
                "pos must be float32, got ", pos.scalar_type());
    pos = pos.contiguous();

    TORCH_CHECK(W >= 1, "W must be >= 1, got ", W);

    if (weights.defined()) {
        TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
        TORCH_CHECK(weights.dim() == 1 && weights.size(0) == pos.size(0),
                    "weights must have shape [N], got [", weights.size(0), "]");
        weights = weights.to(at::kFloat).contiguous();
    }

    return kdtree_bisect_cuda_impl(pos, weights, W);
}
