#include "distnnp_state/edge_distances.hpp"

std::tuple<at::Tensor, at::Tensor> edge_distances(
    at::Tensor pos,               // [N, 3] float32
    at::Tensor edge_index,        // [2, E] int64
    std::optional<at::Tensor> edge_cell_shift = std::nullopt // [E, 3] float32
) {
    // Ensure inputs are on the correct device
    at::cuda::CUDAGuard device_guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    
    int E = edge_index.size(1);

    // Prepare output tensors
    auto opts_float = at::TensorOptions().dtype(at::kFloat).device(pos.device());
    auto edge_distance_vec = at::empty({E, 3}, opts_float);
    auto edge_distance = at::empty({E}, opts_float);

    if (E == 0) {
        return std::make_tuple(edge_distance_vec, edge_distance);
    }

    // Extract cell shift pointer if provided
    const float* shift_ptr = nullptr;
    if (edge_cell_shift.has_value()) {
        shift_ptr = edge_cell_shift.value().contiguous().data_ptr<float>();
    }

    int grid_size = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;

    edge_distances_cuda_impl<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        pos.contiguous().data_ptr<float>(),
        edge_index.contiguous().data_ptr<int64_t>(),
        shift_ptr,
        edge_distance_vec.data_ptr<float>(),
        edge_distance.data_ptr<float>(),
        E
    );

    return std::make_tuple(edge_distance_vec, edge_distance);
}
