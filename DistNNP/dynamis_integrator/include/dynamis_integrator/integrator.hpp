#pragma once
#include <torch/torch.h>

namespace dynamis_integrator
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

  class VerletIntegrator : public AbstractDistNNPIntegrator
  {
  public:
    // Retain step for interface compliance, but we'll primarily use step1 and step2
    void step(torch::Tensor &positions,
              torch::Tensor &velocities,
              const torch::Tensor &forces,
              const torch::Tensor &masses,
              const float dt) override {}

    void halfstep1(torch::Tensor &positions,
                   torch::Tensor &velocities,
                   const torch::Tensor &forces,
                   const torch::Tensor &masses,
                   const float dt);

    void halfstep2(torch::Tensor &velocities,
                   const torch::Tensor &forces,
                   const torch::Tensor &masses,
                   const float dt);
  };

  class LangevinIntegrator : public AbstractDistNNPIntegrator
  {
  public:
    void step(torch::Tensor &positions,
              torch::Tensor &velocities,
              const torch::Tensor &forces,
              const torch::Tensor &masses,
              const float dt) override {}

    void halfstep1(torch::Tensor &positions,
                   torch::Tensor &velocities,
                   const torch::Tensor &forces,
                   const torch::Tensor &masses,
                   const torch::Tensor &noise,
                   const float gamma,
                   const float kT,
                   const float dt);
    void halfstep2(torch::Tensor &velocities,
                   const torch::Tensor &forces,
                   const torch::Tensor &masses,
                   const torch::Tensor &noise,
                   const float gamma,
                   const float kT,
                   const float dt);
  };
} // namespace dynamis_integrator
