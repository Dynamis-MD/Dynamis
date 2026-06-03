#pragma once
#include "distnnp_state/distnnp_state.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <vector>

namespace distnnp_model {

// High-level feature exchanger interface.
// Wraps IPCExchanger for NVLink-based one-sided push
// and wraps RDMAExchanger for LibFabric(interconnect)-based
// one-sided push and sync primitives

class FeatureExchanger {
public:
  FeatureExchanger() = default;
  ~FeatureExchanger();

public:
  virtual void setup(const DistEnvironment &dist_env,
                     const PartitionMessagePassingInfo &partition_comm_info,
                     const size_t max_feature_dim);
  virtual void reset(const PartitionMessagePassingInfo &partition_comm_info,
                     const size_t max_feature_dim);
  virtual void feature_push_forward(
      torch::Tensor return_features, const torch::Tensor local_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info);
  virtual void feature_push_grad(
      torch::Tensor grad_local_features,
      const torch::Tensor grad_output_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info);

protected:
  size_t max_feature_dim_ = 0;
};

// Trampoline Class
class PyFeatureExchanger : public FeatureExchanger,
                           public py::trampoline_self_life_support {
public:
  using FeatureExchanger::FeatureExchanger;

  void setup() override {
    PYBIND11_OVERRIDE_PURE(void, FeatureExchanger, setup);
  }

  void reset() override {
    PYBIND11_OVERRIDE_PURE(void, FeatureExchanger, reset);
  }

  void feature_push_forward(torch::Tensor return_features,
                            const torch::Tensor local_features,
                            const distnnp_state::PartitionMessagePassingInfo
                                &partition_info) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, FeatureExchanger, feature_push,
                           return_features, local_features, partition_info);
  }

  void feature_push_grad(torch::Tensor grad_local_features,
                         const torch::Tensor grad_output_features,
                         const distnnp_state::PartitionMessagePassingInfo
                             &partition_info) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, FeatureExchanger, feature_push,
                           grad_local_features, grad_output_features,
                           partition_info);
  }
};

} // namespace distnnp_model
