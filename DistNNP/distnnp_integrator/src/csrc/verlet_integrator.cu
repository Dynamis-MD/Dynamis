#include "distnnp_integrator/integrator.hpp"
#include "distnnp_integrator/macros.hpp"
#include "distnnp_state/timing.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace
{
  __global__ void verlet_halfstep1_kernel(
          float *positions,
          float *velocities,
          const float *forces,
          const float *masses,
          const float dt,
          const int num_floats)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_floats) {
      velocities[idx] += 0.5f * (forces[idx] / masses[idx / 3]) * dt;
      positions[idx] += velocities[idx] * dt;
    }
  }

  __global__ void verlet_halfstep2_kernel(
          float *velocities,
          const float *forces,
          const float *masses,
          const float dt,
          const int num_floats)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_floats) {
      velocities[idx] += 0.5f * (forces[idx] / masses[idx / 3]) * dt;
    }
  }
}

namespace distnnp_integrator
{
  void VerletIntegrator::halfstep1(
          torch::Tensor &positions,
          torch::Tensor &velocities,
          const torch::Tensor &forces,
          const torch::Tensor &masses,
          const float dt)
  {
    assert(positions.is_contiguous() && positions.is_cuda());
    assert(velocities.is_contiguous() && velocities.is_cuda());
    assert(forces.is_contiguous() && forces.is_cuda());
    assert(masses.is_contiguous() && masses.is_cuda());

    assert(positions.device() == velocities.device());
    assert(positions.device() == forces.device());
    assert(positions.device() == masses.device());

    int num_floats = positions.numel();  // N*3 — every coordinate component

    int threads_per_block = 256;
    int blocks = (num_floats + threads_per_block - 1) / threads_per_block;

    auto device_index = positions.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

#ifdef DISTNNP_TIMING
    distnnp_state::Timer t;
    t.start("integrator_halfstep1", distnnp_state::TimerClass::Compute,
            distnnp_state::Where::Device, stream.stream());
#endif

    verlet_halfstep1_kernel<<<blocks, threads_per_block, 0, stream>>>(
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        forces.data_ptr<float>(),
        masses.data_ptr<float>(),
        dt,
        num_floats);

#ifdef DISTNNP_TIMING
    t.stop();
#endif

    CUDACHECK(cudaGetLastError());
  }

  void VerletIntegrator::halfstep2(
          torch::Tensor &velocities,
          const torch::Tensor &forces,
          const torch::Tensor &masses,
          const float dt)
  {
    assert(velocities.is_contiguous() && velocities.is_cuda());
    assert(forces.is_contiguous() && forces.is_cuda());
    assert(masses.is_contiguous() && masses.is_cuda());

    assert(velocities.device() == forces.device());
    assert(velocities.device() == masses.device());

    int num_floats = velocities.numel();  // N*3 — every coordinate component

    int threads_per_block = 256;
    int blocks = (num_floats + threads_per_block - 1) / threads_per_block;

    auto device_index = velocities.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

#ifdef DISTNNP_TIMING
    distnnp_state::Timer t;
    t.start("integrator_halfstep2", distnnp_state::TimerClass::Compute,
            distnnp_state::Where::Device, stream.stream());
#endif

    verlet_halfstep2_kernel<<<blocks, threads_per_block, 0, stream>>>(
        velocities.data_ptr<float>(),
        forces.data_ptr<float>(),
        masses.data_ptr<float>(),
        dt,
        num_floats);

#ifdef DISTNNP_TIMING
    t.stop();
#endif

    CUDACHECK(cudaGetLastError());
  }
}
