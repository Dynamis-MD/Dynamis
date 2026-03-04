#pragma once
#include <torch/torch.h>

namespace distnnp_integrator
{
  class AbstractDistNNPIntegrator
  {
  public:
    virtual ~AbstractDistNNPIntegrator() = default;
    // Pure virtual method to perform a single integration step
    virtual void step(torch::Tensor &positions,
                      torch::Tensor &velocities,
                      const torch::Tensor &forces,
                      const torch::Tensor &massees,
                      const float dt) = 0;
  };

  class EulerIntegrator : public AbstractDistNNPIntegrator
  {
  public:
    void step(torch::Tensor &positions,
                      torch::Tensor &velocities,
                      const torch::Tensor &forces,
                      const torch::Tensor &massees,
                      const float dt) override;
  };
} // namespace distnnp_integrator
