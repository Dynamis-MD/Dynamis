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
  void setup() override;
  void reset() override;
};
} // namespace distnnp_model