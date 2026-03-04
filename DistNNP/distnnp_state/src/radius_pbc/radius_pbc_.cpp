// radius_pbc_.cpp — C++ entry point with input validation

#include "radius_pbc_.h"
#include <stdexcept>

std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc(
    at::Tensor pos,
    at::Tensor cell,
    at::Tensor pbc,
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly
) {
    // Validate pos: [N, 3] float32 CUDA contiguous
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3,
                "pos must have shape [N, 3], got [", pos.size(0), ", ", pos.size(1), "]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat,
                "pos must be float32, got ", pos.scalar_type());
    pos = pos.contiguous();

    // Validate cell: [3, 3] float32
    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "cell must have shape [3, 3], got [", cell.size(0), ", ", cell.size(1), "]");
    cell = cell.to(pos.device()).to(at::kFloat).contiguous();

    // Validate pbc: [3] bool
    TORCH_CHECK(pbc.dim() == 1 && pbc.size(0) == 3,
                "pbc must have shape [3], got [", pbc.size(0), "]");
    pbc = pbc.to(at::kBool).contiguous();

    // Validate radius
    TORCH_CHECK(radius > 0.0f, "radius must be positive, got ", radius);

    // Validate max_num_neighbors
    TORCH_CHECK(max_num_neighbors >= 0,
                "max_num_neighbors must be non-negative, got ", max_num_neighbors);

    return radius_graph_pbc_cuda_impl(pos, cell, pbc, radius, max_num_neighbors, enforce_max_strictly);
}
