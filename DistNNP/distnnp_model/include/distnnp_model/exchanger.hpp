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
  // --- Public interface ---
  virtual void setup();
  virtual void reset();
  virtual torch::Tensor feature_push(
      const torch::Tensor local_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info);
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

  torch::Tensor feature_push(const torch::Tensor local_features,
                             const distnnp_state::PartitionMessagePassingInfo
                                 &partition_info) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, FeatureExchanger, feature_push,
                           local_features, partition_info);
  }
};

} // namespace distnnp_model
