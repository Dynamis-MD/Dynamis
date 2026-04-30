#include "distnnp_state/edge_distances.hpp"

static constexpr int BLOCK_SIZE = 256;

std::tuple<at::Tensor, at::Tensor> edge_distances(
    at::Tensor pos,               // [N, 3] float32
    at::Tensor edge_index        // [2, E] int64
) {
    // Validate pos: [N, 3] float32 CUDA contiguous
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3,
                "pos must have shape [N, 3], got [", pos.size(0), ", ", pos.size(1), "]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat,
                "pos must be float32, got ", pos.scalar_type());
    pos = pos.contiguous();

    // Validate edge_index: [2, E] int64 CUDA contiguous
    TORCH_CHECK(edge_index.is_cuda(), "edge_index must be a CUDA tensor");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must have shape [2, E], got [", edge_index.size(0), ", ", edge_index.size(1), "]");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong,
                "edge_index must be int64, got ", edge_index.scalar_type());
    edge_index = edge_index.contiguous();

    return edge_distances_cuda_impl(pos, edge_index);
}
