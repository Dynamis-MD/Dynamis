// exchanger.cu — FeatureExchanger implementation.
// Delegates to IPCExchanger for NVLink-based one-sided push.

#include "distnnp_model/exchanger.hpp"
#include <c10/cuda/CUDAStream.h>

namespace distnnp_model
{

FeatureExchanger::~FeatureExchanger()
{
    if (ipc_.is_setup())
        teardown();
}

void FeatureExchanger::setup_ipc(
    torch::Tensor x_buffer,
    torch::Tensor grad_recv_buffer,
    const std::vector<std::vector<uint8_t>> &all_x_handles,
    const std::vector<std::vector<uint8_t>> &all_grad_handles,
    torch::Tensor send_offsets,
    torch::Tensor send_local_indices,
    torch::Tensor recv_offsets,
    torch::Tensor all_recv_offsets,
    torch::Tensor all_num_local,
    int64_t num_local,
    int64_t feature_dim,
    int64_t rank,
    int64_t world_size)
{
    x_buffer_ = x_buffer;
    grad_recv_buffer_ = grad_recv_buffer;

    ipc_.setup(x_buffer, grad_recv_buffer,
               all_x_handles, all_grad_handles,
               send_offsets, send_local_indices,
               recv_offsets, all_recv_offsets, all_num_local,
               num_local, feature_dim, rank, world_size);
}

void FeatureExchanger::teardown()
{
    ipc_.teardown();
    x_buffer_.reset();
    grad_recv_buffer_.reset();
}

torch::Tensor FeatureExchanger::feature_push_forward(
    const torch::Tensor &local_features,
    const distnnp_state::PartitionMessagePassingInfo &partition_info)
{
    TORCH_CHECK(ipc_.is_setup(),
                "FeatureExchanger: IPC not set up. Call setup_ipc() first.");

    int64_t N_local = ipc_.num_local();
    int64_t F = ipc_.feature_dim();

    TORCH_CHECK(local_features.size(0) == N_local && local_features.size(1) == F,
                "local_features shape mismatch: expected [", N_local, ", ", F,
                "], got [", local_features.size(0), ", ", local_features.size(1), "]");

    // Use PyTorch's current CUDA stream so all ops (copy, push, event, clone) are ordered.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        local_features.device().index()).stream();

    
    // Copy local features into x_buffer's local region [0:N_local, :]
    x_buffer_.narrow(0, 0, N_local).copy_(local_features);

    // Push send data to peers' ghost regions via IPC
    ipc_.push_forward(x_buffer_, stream, ce_threshold_bytes_);

    // Record our completion and wait for all peers
    ipc_.record_push_complete(stream);
    ipc_.wait_for_pushes(stream);

    // x_buffer now has [local_features | ghost_from_all_peers].

    auto return_tensor = x_buffer_.clone(); // Clone to ensure the returned tensor is not modified in-place by future pushes.
    return return_tensor;
}

torch::Tensor FeatureExchanger::feature_push_backward(
    const torch::Tensor &grad_output,
    const torch::IntArrayRef &grad_input_shape,
    const distnnp_state::PartitionMessagePassingInfo &partition_info)
{
    TORCH_CHECK(ipc_.is_setup(),
                "FeatureExchanger: IPC not set up. Call setup_ipc() first.");

    int64_t N_local = ipc_.num_local();
    int64_t F = ipc_.feature_dim();

    // Use PyTorch's current CUDA stream for consistent ordering.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        grad_output.device().index()).stream();

    // Push ghost gradients to peers' grad_recv_buffers.
    // Note: grad_recv_buffer_ is fully overwritten by peers' pushes (CSR covers all total_send rows),
    // so no zeroing is needed. Zeroing here would race with peers' IPC writes.
    // grad_output has shape [N_local + N_ghost, F]
    ipc_.push_backward(grad_output, stream);

    // Record our completion and wait for all peers
    ipc_.record_push_complete(stream);
    ipc_.wait_for_pushes(stream);

    // grad_recv_buffer_ now has incoming gradients from peers.
    // Each row j corresponds to local atom send_local_indices[j].
    // Start with the direct gradient for local atoms.
    torch::Tensor grad_input = grad_output.narrow(0, 0, N_local).clone();

    // Scatter-add received gradients into local positions
    if (ipc_.total_send() > 0)
    {
        grad_input.index_add_(0, ipc_.send_local_indices(), grad_recv_buffer_);
    }

    return grad_input;
}

} // namespace distnnp_model
