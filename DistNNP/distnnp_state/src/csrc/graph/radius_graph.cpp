// radius_graph.cpp — C++ entry point with input validation

#include "distnnp_state/radius_graph.hpp"
#include "distnnp_state/timing.hpp"
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

    distnnp_state::Timer t;

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

    // pos.is_cuda() is guaranteed by the TORCH_CHECK above, so device-side event
    // timing on that device's current stream is safe here.
    t.start("build_graph", distnnp_state::TimerClass::Compute,
            distnnp_state::Where::Device);

    auto result = radius_graph_pbc_cuda_impl(pos, cell, pbc, radius, max_num_neighbors, enforce_max_strictly);

    t.stop();

    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_ex(
    at::Tensor pos,
    at::Tensor cell,
    at::Tensor pbc,
    float radius,
    int64_t max_num_neighbors,
    bool enforce_max_strictly,
    c10::optional<at::Tensor> target_indices
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

    // Validate radius / max_num_neighbors
    TORCH_CHECK(radius > 0.0f, "radius must be positive, got ", radius);
    TORCH_CHECK(max_num_neighbors >= 0,
                "max_num_neighbors must be non-negative, got ", max_num_neighbors);

    // Validate target_indices: [N_target] int64 on same device as pos
    if (target_indices.has_value()) {
        auto ti = target_indices.value();
        TORCH_CHECK(ti.device() == pos.device(),
                    "target_indices must be on the same device as pos");
        TORCH_CHECK(ti.dim() == 1, "target_indices must be 1-D, got shape ", ti.sizes());
        TORCH_CHECK(ti.scalar_type() == at::kLong,
                    "target_indices must be int64, got ", ti.scalar_type());
        target_indices = ti.contiguous();
    }

    return radius_graph_pbc_cuda_impl_ex(
        pos, cell, pbc, radius, max_num_neighbors, enforce_max_strictly,
        target_indices, /*counts_only=*/false);
}

at::Tensor
radius_graph_pbc_counts_ex(
    at::Tensor pos,
    at::Tensor cell,
    at::Tensor pbc,
    float radius,
    c10::optional<at::Tensor> target_indices
) {
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3,
                "pos must have shape [N, 3], got [", pos.size(0), ", ", pos.size(1), "]");
    TORCH_CHECK(pos.scalar_type() == at::kFloat,
                "pos must be float32, got ", pos.scalar_type());
    pos = pos.contiguous();

    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "cell must have shape [3, 3], got [", cell.size(0), ", ", cell.size(1), "]");
    cell = cell.to(pos.device()).to(at::kFloat).contiguous();

    TORCH_CHECK(pbc.dim() == 1 && pbc.size(0) == 3,
                "pbc must have shape [3], got [", pbc.size(0), "]");
    pbc = pbc.to(at::kBool).contiguous();

    TORCH_CHECK(radius > 0.0f, "radius must be positive, got ", radius);

    if (target_indices.has_value()) {
        auto ti = target_indices.value();
        TORCH_CHECK(ti.device() == pos.device(),
                    "target_indices must be on the same device as pos");
        TORCH_CHECK(ti.dim() == 1, "target_indices must be 1-D, got shape ", ti.sizes());
        TORCH_CHECK(ti.scalar_type() == at::kLong,
                    "target_indices must be int64, got ", ti.scalar_type());
        target_indices = ti.contiguous();
    }

    auto result = radius_graph_pbc_cuda_impl_ex(
        pos, cell, pbc, radius, /*max_num_neighbors=*/0, /*enforce_max_strictly=*/false,
        target_indices, /*counts_only=*/true);
    return std::get<3>(result);
}
