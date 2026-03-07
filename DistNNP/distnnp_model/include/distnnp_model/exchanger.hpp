#pragma once
#include <torch/extension.h>
#include "distnnp_state/distnnp_state.hpp"

namespace distnnp_model
{

  class FeatureExchanger
  {
    // TODO: Add any necessary member variables for managing communication state, buffers, etc.
  public:
    torch::Tensor feature_push_forward(const torch::Tensor &local_features,
                                       const distnnp_state::PartitionMessagePassingInfo &partition_info);

    torch::Tensor feature_push_backward(const torch::Tensor &grad_output,
                                        const torch::IntArrayRef &grad_input_shape,
                                        const distnnp_state::PartitionMessagePassingInfo &partition_info);
  };

}