// exchanger.cu — FeatureExchanger implementation.
// Manages double-buffered max_feature_dim halo buffers and delegates communication
// to a pluggable CommBackend (default: IPCExchanger for NVLink / Fabric one-sided push).

#include "dynamis_model/exchanger.hpp"
#include <c10/cuda/CUDAStream.h>

namespace dynamis_model
{

FeatureExchanger::FeatureExchanger(const std::string &backend_type)
{
    if (backend_type.empty() || backend_type == "ipc") {
        backend_ = std::make_unique<IPCExchanger>();
    } else {
        TORCH_CHECK(false, "Unsupported FeatureExchanger C++ backend: '", backend_type,
                    "'. Supported backends: 'ipc'.");
    }
}

FeatureExchanger::~FeatureExchanger()
{
    if (is_setup())
        teardown();
}

std::vector<uint8_t> FeatureExchanger::prepare_setup(
    const dynamis_state::PartitionMessagePassingInfo &partition_info,
    int64_t max_feature_dim,
    int64_t rank,
    int64_t world_size,
    int64_t device_index)
{
    TORCH_CHECK(backend_ != nullptr, "FeatureExchanger: backend is null");
    TORCH_CHECK(max_feature_dim > 0, "FeatureExchanger: max_feature_dim must be positive");

    auto cpu_so = partition_info.send_offsets.to(torch::kCPU, torch::kInt64).contiguous();
    auto cpu_ro = partition_info.recv_offsets.to(torch::kCPU, torch::kInt64).contiguous();

    num_local_ = partition_info.partition_indices.numel();
    total_send_ = cpu_so.data_ptr<int64_t>()[world_size];
    total_recv_ = cpu_ro.data_ptr<int64_t>()[world_size];
    max_feature_dim_ = max_feature_dim;
    num_slots_ = 2;

    send_local_indices_ =
            partition_info.send_local_indices
                    .to(torch::Device(torch::kCUDA, device_index), torch::kInt64)
                    .contiguous();

    return backend_->prepare_local_metadata(x_buffer_, grad_recv_buffer_, partition_info,
                                            max_feature_dim, rank, world_size, device_index);
}

void FeatureExchanger::complete_setup(const std::vector<std::vector<uint8_t>> &all_rank_metadata)
{
    TORCH_CHECK(backend_ != nullptr, "FeatureExchanger: backend is null");
    backend_->connect(all_rank_metadata);
}

void FeatureExchanger::teardown()
{
    if (backend_)
        backend_->teardown();
    x_buffer_.reset();
    grad_recv_buffer_.reset();
    send_local_indices_.reset();
}

torch::Tensor FeatureExchanger::feature_push_forward(const torch::Tensor &local_features)
{
    TORCH_CHECK(is_setup(),
                "FeatureExchanger: not set up. Call prepare_setup()/complete_setup() first.");
    TORCH_CHECK(local_features.dim() == 2,
                "local_features must be 2-D [num_local, feature_dim], got dim=",
                local_features.dim());

    int64_t N_local = num_local_;
    int64_t F = local_features.size(1);

    TORCH_CHECK(local_features.size(0) == N_local,
                "local_features num_local mismatch: expected ", N_local,
                ", got ", local_features.size(0));
    TORCH_CHECK(F > 0 && F <= max_feature_dim_,
                "local_features feature_dim (", F, ") exceeds configured max_feature_dim (",
                max_feature_dim_, ")");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        local_features.device().index()).stream();

    int64_t slot = (num_slots_ > 1) ? backend_->next_slot() : 0;
    int64_t x_rows_per_slot = std::max<int64_t>(1, N_local + total_recv_);
    auto x_slot = x_buffer_.narrow(0, slot * x_rows_per_slot, x_rows_per_slot);

    int64_t total_atoms = N_local + total_recv_;
    auto x_view = x_slot.reshape({ -1 }).slice(0, 0, total_atoms * F).view({ total_atoms, F });

    if (N_local > 0) {
        x_view.narrow(0, 0, N_local).copy_(local_features);
    }

    backend_->push_forward_slot(x_view, F, slot, stream, ce_threshold_bytes_);

    return x_view.clone();
}

torch::Tensor FeatureExchanger::feature_push_backward(const torch::Tensor &grad_output)
{
    TORCH_CHECK(is_setup(),
                "FeatureExchanger: not set up. Call prepare_setup()/complete_setup() first.");
    TORCH_CHECK(grad_output.dim() == 2,
                "grad_output must be 2-D [num_local + total_recv, feature_dim], got dim=",
                grad_output.dim());

    int64_t N_local = num_local_;
    int64_t F = grad_output.size(1);

    TORCH_CHECK(grad_output.size(0) == N_local + total_recv_,
                "grad_output row mismatch: expected ", N_local + total_recv_,
                ", got ", grad_output.size(0));
    TORCH_CHECK(F > 0 && F <= max_feature_dim_,
                "grad_output feature_dim (", F, ") exceeds configured max_feature_dim (",
                max_feature_dim_, ")");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        grad_output.device().index()).stream();

    auto grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    int64_t slot = (num_slots_ > 1) ? backend_->next_slot() : 0;

    backend_->push_backward_slot(grad_contig, F, slot, stream);

    torch::Tensor grad_input = grad_output.narrow(0, 0, N_local).clone();

    if (total_send_ > 0) {
        int64_t grad_rows_per_slot = std::max<int64_t>(1, total_send_);
        auto grad_slot = grad_recv_buffer_.narrow(0, slot * grad_rows_per_slot, grad_rows_per_slot);
        auto grad_view = grad_slot.reshape({ -1 }).slice(0, 0, total_send_ * F).view({ total_send_, F });
        grad_input.index_add_(0, send_local_indices_, grad_view);
    }

    return grad_input;
}

// --- Legacy IPC setup helpers ---

std::vector<uint8_t> FeatureExchanger::export_ipc_handle(torch::Tensor buffer)
{
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr, "export_ipc_handle is only supported on IPCExchanger backend");
    return ipc->export_ipc_handle(buffer);
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
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr, "setup_ipc is only supported on IPCExchanger backend");

    x_buffer_ = x_buffer;
    grad_recv_buffer_ = grad_recv_buffer;
    num_local_ = num_local;
    max_feature_dim_ = feature_dim;
    num_slots_ = 1;
    send_local_indices_ = send_local_indices.to(x_buffer.device(), torch::kInt64).contiguous();

    ipc->setup(x_buffer, grad_recv_buffer,
               all_x_handles, all_grad_handles,
               send_offsets, send_local_indices,
               recv_offsets, all_recv_offsets, all_num_local,
               num_local, feature_dim, rank, world_size);

    total_send_ = ipc->total_send();
    total_recv_ = ipc->total_recv();
}

std::vector<uint8_t> FeatureExchanger::create_and_export_sync_handle()
{
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr,
                "create_and_export_sync_handle is only supported on IPCExchanger backend");
    return ipc->create_and_export_sync_handle();
}

void FeatureExchanger::setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles)
{
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr, "setup_sync is only supported on IPCExchanger backend");
    ipc->setup_sync(all_sync_handles);
}

void FeatureExchanger::record_push_complete()
{
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr, "record_push_complete is only supported on IPCExchanger backend");
    ipc->record_push_complete(0);
}

void FeatureExchanger::wait_for_pushes()
{
    auto *ipc = dynamic_cast<IPCExchanger *>(backend_.get());
    TORCH_CHECK(ipc != nullptr, "wait_for_pushes is only supported on IPCExchanger backend");
    ipc->wait_for_pushes(0);
}

} // namespace dynamis_model
