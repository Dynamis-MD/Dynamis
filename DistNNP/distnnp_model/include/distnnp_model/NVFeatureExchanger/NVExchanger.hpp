#pragma once
#include "distnnp_model/exchanger.hpp"

namespace distnnp_model {

// Implementation of the NVidia specific exchanger
// using GPUDirect P2P and RDMA technologies

class NVFeatureExchanger : public FeatureExchanger {
public:
  void setup() override;
  void reset(const distnnp_state::PartitionMessagePassingInfo &partition_info);
  torch::Tensor feature_push(const torch::Tensor local_features,
                             const distnnp_state::PartitionMessagePassingInfo
                                 &partition_info) override;
};
} // namespace distnnp_model
