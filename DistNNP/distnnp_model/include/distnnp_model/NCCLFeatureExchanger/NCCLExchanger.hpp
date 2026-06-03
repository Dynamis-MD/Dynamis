#pragma once
#include "distnnp_model/exchanger.hpp"
#include <nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace distnnp_model {

// ---- global registration slot ----
static c10::intrusive_ptr<c10d::ProcessGroupNCCL> g_pg;
static int g_device_idx = -1;

void register_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg,
                            int device_idx) {
  g_pg = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupNCCL>(pg);
  TORCH_CHECK(g_pg != nullptr, "Expected ProcessGroupNCCL");
  g_device_idx = device_idx;
}

class NCCLFeatureExchanger : public FeatureExchanger {
public:
  void setup(const distnnp_state::DistEnvironment &dist_env,
             const distnnp_state::PartitionMessagePassingInfo &partition_comm_info,
             const size_t max_feature_dim) override;
  void reset(const distnnp_state::PartitionMessagePassingInfo &partition_comm_info,
             const size_t max_feature_dim) override;
  void feature_push_forward(
      torch::Tensor return_features,
      const torch::Tensor local_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info) override;
  void feature_push_grad(
      torch::Tensor grad_local_features,
      const torch::Tensor grad_output_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info) override;
};
} // namespace distnnp_model