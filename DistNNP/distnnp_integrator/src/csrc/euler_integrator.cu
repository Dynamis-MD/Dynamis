#include "distnnp_integrator/integrator.hpp"
#include "distnnp_integrator/macros.hpp"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>


namespace
{
  __global__ void euler_step_kernel(
      float *positions,
      float *velocities,
      const float *forces,
      const float *masses,
      const float dt,
      const int num_atoms)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_atoms * 3)
    {
      const int atom_idx = idx / 3;
      velocities[idx] += (forces[idx] / masses[idx / 3]) * dt;
      // Update position: x(t + dt) = x(t) + v(t + dt) * dt
      positions[idx] += velocities[idx] * dt;
    }
  }
}
} // anonymous namespace

namespace distnnp_integrator {
void EulerIntegrator::step(torch::Tensor &positions, torch::Tensor &velocities,
                           const torch::Tensor &forces,
                           const torch::Tensor &masses, const float dt) {
  // Ensure tensors are contiguous and on GPU
  assert(positions.is_contiguous() && positions.is_cuda());
  assert(velocities.is_contiguous() && velocities.is_cuda());
  assert(forces.is_contiguous() && forces.is_cuda());
  assert(masses.is_contiguous() && masses.is_cuda());
  // Ensure tensors are on the same device
  assert(positions.device() == velocities.device());
  assert(positions.device() == forces.device());
  assert(positions.device() == masses.device());

  // Get the number of atoms
  int num_atoms = positions.size(0);

  // Launch the CUDA kernel to perform the Euler integration step
  int threads_per_block = 256;
  int blocks = (num_atoms * 3 + threads_per_block - 1) / threads_per_block;

  auto device_index = positions.device().index();
  auto stream = at::cuda::getCurrentCUDAStream(device_index);

  euler_step_kernel<<<blocks, threads_per_block, 0, stream>>>(
      positions.data_ptr<float>(), velocities.data_ptr<float>(),
      forces.data_ptr<float>(), masses.data_ptr<float>(), dt, num_atoms);

  // Check for CUDA errors
  CUDACHECK(cudaGetLastError());
}

} // namespace distnnp_integrator
