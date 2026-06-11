// dist_graph_ops_.cpp — C++ entry points with input validation.

#include "distnnp_state/dist_graph_ops.hpp"

std::tuple<at::Tensor, at::Tensor>
filter_edges_to_radius(
    at::Tensor edges,
    at::Tensor cell_shifts,
    at::Tensor pos,
    at::Tensor cell,
    double radius)
{
    TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
    TORCH_CHECK(edges.dim() == 2 && edges.size(0) == 2,
                "edges must have shape [2, E], got ", edges.sizes());
    TORCH_CHECK(edges.scalar_type() == at::kLong,
                "edges must be int64, got ", edges.scalar_type());

    const int64_t E = edges.size(1);

    TORCH_CHECK(cell_shifts.dim() == 2 && cell_shifts.size(1) == 3,
                "cell_shifts must have shape [E, 3], got ", cell_shifts.sizes());
    TORCH_CHECK(cell_shifts.size(0) == E,
                "cell_shifts rows (", cell_shifts.size(0),
                ") must equal number of edges E (", E, ")");

    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3,
                "pos must have shape [N, 3], got ", pos.sizes());
    TORCH_CHECK(pos.scalar_type() == at::kFloat,
                "pos must be float32, got ", pos.scalar_type());

    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "cell must have shape [3, 3], got ", cell.sizes());
    TORCH_CHECK(radius > 0.0, "radius must be positive, got ", radius);

    edges = edges.contiguous();
    cell_shifts = cell_shifts.contiguous();
    pos = pos.contiguous();
    cell = cell.to(pos.device()).to(at::kFloat).contiguous();

    return filter_edges_to_radius_cuda_impl(edges, cell_shifts, pos, cell, radius);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
extract_halo_info(
    at::Tensor edges,
    at::Tensor pid_outer,
    at::Tensor my_outer_atoms,
    int64_t my_rank,
    int64_t world_size,
    int64_t N)
{
    TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
    TORCH_CHECK(edges.dim() == 2 && edges.size(0) == 2,
                "edges must have shape [2, E], got ", edges.sizes());
    TORCH_CHECK(edges.scalar_type() == at::kLong, "edges must be int64");
    TORCH_CHECK(pid_outer.is_cuda() && pid_outer.dim() == 1,
                "pid_outer must be a 1-D CUDA tensor");
    TORCH_CHECK(pid_outer.size(0) == N,
                "pid_outer must have length N (", N, "), got ", pid_outer.size(0));
    TORCH_CHECK(my_outer_atoms.is_cuda() && my_outer_atoms.dim() == 1
                    && my_outer_atoms.scalar_type() == at::kLong,
                "my_outer_atoms must be 1-D int64 CUDA");
    TORCH_CHECK(world_size >= 1, "world_size must be >= 1, got ", world_size);
    TORCH_CHECK(my_rank >= 0 && my_rank < world_size,
                "my_rank must be in [0, world_size)");

    edges = edges.contiguous();
    pid_outer = pid_outer.to(at::kLong).contiguous();
    my_outer_atoms = my_outer_atoms.contiguous();

    return extract_halo_info_cuda_impl(
        edges, pid_outer, my_outer_atoms, my_rank, world_size, N);
}
