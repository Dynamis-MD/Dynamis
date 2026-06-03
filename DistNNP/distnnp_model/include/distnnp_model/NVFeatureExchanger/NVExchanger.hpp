#pragma once
#include "distnnp_model/exchanger.hpp"

namespace distnnp_model {

// Implementation of the NVidia specific exchanger
// using GPUDirect P2P and RDMA technologies

class NVFeatureExchanger : public FeatureExchanger {
public:
  void
  setup(const distnnp_state::DistEnvironment &dist_env,
        const distnnp_state::PartitionMessagePassingInfo &partition_comm_info,
        const size_t max_feature_dim) override;
  void
  reset(const distnnp_state::PartitionMessagePassingInfo &partition_comm_info,
        const size_t max_feature_dim) override;
  void feature_push_forward(torch::Tensor return_features,
                            const torch::Tensor local_features,
                            const distnnp_state::PartitionMessagePassingInfo
                                &partition_info) override;
  void feature_push_grad(torch::Tensor grad_local_features,
                         const torch::Tensor grad_output_features,
                         const distnnp_state::PartitionMessagePassingInfo
                             &partition_info) override;

private:
  void setup_routing_table();
};
} // namespace distnnp_model
