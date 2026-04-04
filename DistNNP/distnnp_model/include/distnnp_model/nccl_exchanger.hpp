#pragma once
#include <torch/extension.h>
#include <nccl.h>
#include "distnnp_state/distnnp_state.hpp"

namespace distnnp_model
{

  class NCCLFeatureExchangerA2A:FeatureExchanger 
  {
    private:
      ncclComm_t comm_;
      int rank_;
      int world_size_;

      // Number of atoms that go to and from each rank
      std::vector<int64_t> cached_send_splits_;
      std::vector<int64_t> cached_recv_splits_;

      // Convert CSR offsets to per-rank split sizes for ncclSend/ncclRecv
      void update_splits_cache_(const distnnp_state::PartitionMessagePassingInfo &partition_info);

  public:
    NCCLFeatureExchangerRMA(ncclComm_t comm, int rank, int world_size);

    at::Tensor feature_push_forward(const at::Tensor &local_features,
                                    const distnnp_state::PartitionMessagePassingInfo &partition_info);

    at::Tensor feature_push_backward(const at::Tensor &grad_output,
                                     const at::IntArrayRef &grad_input_shape,
                                     const distnnp_state::PartitionMessagePassingInfo &partition_info);
  };

  
  class NCCLFeatureExchangerRMA:FeatureExchanger 
  {
    private:
      ncclComm_t comm_;
      int rank_;
      int world_size_;

      // Number of atoms that go to and from each rank
      std::vector<int64_t> cached_send_splits_;
      std::vector<int64_t> cached_recv_splits_;

      // Convert CSR offsets to per-rank split sizes for ncclSend/ncclRecv
      void update_splits_cache_(const distnnp_state::PartitionMessagePassingInfo &partition_info);

  public:
    NCCLFeatureExchangerRMA(ncclComm_t comm, int rank, int world_size);

    at::Tensor feature_push_forward(const at::Tensor &local_features,
                                    const distnnp_state::PartitionMessagePassingInfo &partition_info);

    at::Tensor feature_push_backward(const at::Tensor &grad_output,
                                     const at::IntArrayRef &grad_input_shape,
                                     const distnnp_state::PartitionMessagePassingInfo &partition_info);
  };

  

}
