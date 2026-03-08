#include "distnnp_model/exchanger.hpp"

namespace distnnp_model
{

  torch::Tensor FeatureExchanger::feature_push_forward(const torch::Tensor &local_features,
                                                       const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {

    // TODO: Implement the actual feature push logic here.
    torch::Tensor pushed_features = torch::zeros_like(local_features); // Placeholder for the actual pushed features
    return pushed_features;
  }

  torch::Tensor FeatureExchanger::feature_push_backward(const torch::Tensor &grad_output,
                                                        const torch::IntArrayRef &grad_input_shape,
                                                        const distnnp_state::PartitionMessagePassingInfo &partition_info)
  {
    // TODO: Implement the actual feature push backward logic here.
    torch::Tensor grad_input = torch::zeros(grad_input_shape, torch::TensorOptions().device(grad_output.device()).dtype(grad_output.dtype())); // Placeholder for the actual gradients for input features
    return grad_input;
  }
}
