#include "distnnp_integrator/integrator.hpp"
#include "distnnp_integrator/macros.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace
{
  __global__ void langevin_halfstep1_kernel(
          float *positions,
          float *velocities,
          const float *forces,
          const float *masses,
          const float *noise,
          const float gamma,
          const float kT,
          const float dt,
          const int num_elements)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
      const int atom_idx = idx / 3;
      float m = masses[atom_idx];
      float c1 = 1.0f - 0.5f * gamma * dt;
      float c2 = sqrtf(gamma * kT * dt / m);

      velocities[idx] = velocities[idx] * c1 + 0.5f * (forces[idx] / m) * dt + c2 * noise[idx];
      positions[idx] += velocities[idx] * dt;
    }
  }

  __global__ void langevin_halfstep2_kernel(
          float *velocities,
          const float *forces,
          const float *masses,
          const float *noise,
          const float gamma,
          const float kT,
          const float dt,
          const int num_elements)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
      const int atom_idx = idx / 3;
      float m = masses[atom_idx];
      float c3 = 1.0f / (1.0f + 0.5f * gamma * dt);
      float c2 = sqrtf(gamma * kT * dt / m);

      velocities[idx] = (velocities[idx] + 0.5f * (forces[idx] / m) * dt + c2 * noise[idx]) * c3;
    }
  }
}

namespace distnnp_integrator
{
  void LangevinIntegrator::halfstep1(
          torch::Tensor &positions,
          torch::Tensor &velocities,
          const torch::Tensor &forces,
          const torch::Tensor &masses,
          const torch::Tensor &noise,
          const float gamma,
          const float kT,
          const float dt)
  {
    assert(positions.is_contiguous() && positions.is_cuda());
    assert(velocities.is_contiguous() && velocities.is_cuda());
    assert(forces.is_contiguous() && forces.is_cuda());
    assert(masses.is_contiguous() && masses.is_cuda());
    assert(noise.is_contiguous() && noise.is_cuda());

    assert(positions.device() == velocities.device());
    assert(positions.device() == forces.device());
    assert(positions.device() == masses.device());
    assert(positions.device() == noise.device());

    int num_atoms = positions.size(0);
    int num_elements = num_atoms * 3;

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    auto device_index = positions.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    langevin_halfstep1_kernel<<<blocks, threads, 0, stream>>>(
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        forces.data_ptr<float>(),
        masses.data_ptr<float>(),
        noise.data_ptr<float>(),
        gamma,
        kT,
        dt,
        num_elements);

    CUDACHECK(cudaGetLastError());
  }

  void LangevinIntegrator::halfstep2(
          torch::Tensor &velocities,
          const torch::Tensor &forces,
          const torch::Tensor &masses,
          const torch::Tensor &noise,
          const float gamma,
          const float kT,
          const float dt)
  {
    assert(velocities.is_contiguous() && velocities.is_cuda());
    assert(forces.is_contiguous() && forces.is_cuda());
    assert(masses.is_contiguous() && masses.is_cuda());
    assert(noise.is_contiguous() && noise.is_cuda());

    assert(velocities.device() == forces.device());
    assert(velocities.device() == masses.device());
    assert(velocities.device() == noise.device());

    int num_atoms = velocities.size(0);
    int num_elements = num_atoms * 3;

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    auto device_index = velocities.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    langevin_halfstep2_kernel<<<blocks, threads, 0, stream>>>(
        velocities.data_ptr<float>(),
        forces.data_ptr<float>(),
        masses.data_ptr<float>(),
        noise.data_ptr<float>(),
        gamma,
        kT,
        dt,
        num_elements);

    CUDACHECK(cudaGetLastError());
  }
}
