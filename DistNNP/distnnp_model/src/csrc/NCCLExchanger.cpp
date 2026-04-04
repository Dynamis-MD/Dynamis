#include "distnnp_model/exchanger.hpp"
#include "distnnp_model/push_buffer.hpp"
#include "distnnp_model/macros.hpp"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace distnnp_model
{

  at::Tensor NCCLFeatureExchangerRMA::feature_push_forward(const at::Tensor &local_features,
                                                           const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {

    NCCLCHECK(ncclGroupStart());
    for (auto r = 0; r < this->world_size_; ++r)
    {
      if (r == this->rank_)
        continue;
      if (send_size[r] == 0)
        continue;
      const auto send_size = send_size[r] * message_size;

      /*
      ncclResult_t ncclPutSignal(const void *localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream
       */
      ncclPutSignal(send_buf, send_size, float, r, recv_buf + offset, )
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // Convert CSR offsets to per-rank split sizes for ncclSend/ncclRecv
  void FeatureExchanger::update_splits_cache_(
      const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {
    //
    auto send_off = partition_info.send_offsets.cpu().contiguous();
    // recv_off[i] number of unique vertices received from process i to current process
    auto recv_off = partition_info.recv_offsets.cpu().contiguous();

    // int64_t W = send_off.size(0) - 1;
    const auto world_size = this->world_size_;

    const int64_t *sp = send_off.data_ptr<int64_t>();
    const int64_t *rp = recv_off.data_ptr<int64_t>();

    // send_off[i] number of unique vertices sent from current process to process i
    cached_send_splits_.resize(world_size);
    cached_recv_splits_.resize(world_size);

    for (int64_t r = 0; r < world_size; ++r)
    {
      cached_send_splits_[r] = sp[r + 1] - sp[r];
      cached_recv_splits_[r] = rp[r + 1] - rp[r];
    }
  }

  FeatureExchanger::FeatureExchanger(ncclComm_t comm, int rank, int world_size)
      : comm_(comm), rank_(rank), world_size_(world_size) {}

  at::Tensor FeatureExchanger::feature_push_forward(const at::Tensor &local_features,
                                                    const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {
    // Claim CUDA device
    at::cuda::CUDAGuard device_guard(local_features.device());

    // Update splits cache
    update_splits_cache_(partition_info);

    // Gather buffer size
    int64_t total_send = partition_info.send_local_indices.size(0);
    int64_t F = local_features.size(1);

    // Create buffer
    at::Tensor send_buf = at::empty({total_send, F}, local_features.options());

    // Fill buffer — gather rows at send_local_indices into send_buf
    push_buffer_create(local_features, send_buf, partition_info.send_local_indices);

    // Allocate receive buffer
    int64_t total_recv = partition_info.recv_ghost_indices.size(0);
    at::Tensor recv_buf = at::empty({total_recv, F}, local_features.options());

    // Compute per-rank element counts (atoms × features)
    int64_t W = cached_send_splits_.size();
    std::vector<int64_t> send_counts(W), recv_counts(W);
    for (int64_t r = 0; r < W; ++r)
    {
      send_counts[r] = cached_send_splits_[r] * F;
      recv_counts[r] = cached_recv_splits_[r] * F;
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Exchange via NCCL point-to-point sends/recvs
    float *send_ptr = send_buf.data_ptr<float>();
    float *recv_ptr = recv_buf.data_ptr<float>();

    NCCLCHECK(ncclGroupStart());
    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    for (int64_t r = 0; r < W; ++r)
    {
      if (send_counts[r] > 0)
      {
        NCCLCHECK(ncclSend(send_ptr + send_offset, send_counts[r], ncclFloat, r, comm_, stream));
      }
      if (recv_counts[r] > 0)
      {
        NCCLCHECK(ncclRecv(recv_ptr + recv_offset, recv_counts[r], ncclFloat, r, comm_, stream));
      }
      send_offset += send_counts[r];
      recv_offset += recv_counts[r];
    }
    NCCLCHECK(ncclGroupEnd());

    // Block until communication finishes
    CUDACHECK(cudaStreamSynchronize(stream));

    // Return the pushed features
    return recv_buf;
  }

  at::Tensor FeatureExchanger::feature_push_backward(const at::Tensor &grad_output,
                                                     const at::IntArrayRef &grad_input_shape,
                                                     const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {
    // grad_output [num_boundary_vertices, F] grouped together by rank

    // grad_output[recv_offset[i+1]:recv_offset[i],:] are all gradient we must send to rank i

    // Claim CUDA device
    at::cuda::CUDAGuard device_guard(grad_output.device());

    // Update splits cache
    update_splits_cache_(partition_info);

    // Buffer sizes
    int64_t total_send = partition_info.send_local_indices.size(0);
    int64_t F = grad_output.size(1);

    // Allocate buffer for gradients coming back to this rank
    at::Tensor grad_owner = at::empty({total_send, F}, grad_output.options());

    // Compute per-rank element counts (atoms × features)
    int64_t W = cached_send_splits_.size();
    std::vector<int64_t> send_counts(W), recv_counts(W);
    for (int64_t r = 0; r < W; ++r)
    {
      send_counts[r] = cached_send_splits_[r] * F;
      recv_counts[r] = cached_recv_splits_[r] * F;
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Reverse exchange — send ghost grads back to owners
    float *grad_ghost_ptr = grad_output.data_ptr<float>();
    float *grad_owner_ptr = grad_owner.data_ptr<float>();

    NCCLCHECK(ncclGroupStart());
    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    for (int64_t r = 0; r < W; ++r)
    {
      // Swap vs forward: send with recv_counts, receive with send_counts
      if (recv_counts[r] > 0)
      {
        NCCLCHECK(ncclSend(grad_ghost_ptr + send_offset, recv_counts[r], ncclFloat, r, comm_, stream));
      }
      if (send_counts[r] > 0)
      {
        NCCLCHECK(ncclRecv(grad_owner_ptr + recv_offset, send_counts[r], ncclFloat, r, comm_, stream));
      }
      send_offset += recv_counts[r];
      recv_offset += send_counts[r];
    }
    NCCLCHECK(ncclGroupEnd());

    // Block until communication finishes
    CUDACHECK(cudaStreamSynchronize(stream));

    // Scatter-add gradients back to local atom positions
    at::Tensor grad_input = at::zeros(grad_input_shape, grad_output.options());
    grad_input.index_add_(0, partition_info.send_local_indices, grad_owner);

    return grad_input;
  }
}
