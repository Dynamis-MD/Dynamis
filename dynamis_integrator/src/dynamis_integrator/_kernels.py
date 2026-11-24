"""Triton integrator updates, replacing the dynamis_integrator_impl CUDA extension.

Each function mirrors the bound C++ method of the same name and updates its
tensor arguments in place.
"""

import torch
import triton
import triton.language as tl
from dynamis_state import Timer, TimerClass, Where

# Device-side kernel timer. Keys match the ones the .cu kernels record, so
# CUDA and Triton runs line up in post_processing/timing_summary.py.
_timer = Timer()

@triton.jit
def euler_step_kernel(
    positions_ptr,
    velocities_ptr,
    forces_ptr,
    masses_ptr,
    dt,
    num_atoms,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel processes the flattened coordinate arrays (num_atoms * 3)
    total_elements = num_atoms * 3

    # Calculate global element offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boolean mask to protect against out-of-bounds array access
    mask = offsets < total_elements

    # Since each atom has 3 spatial coordinates (X, Y, Z),
    # we map the coordinate offset back to its corresponding atom index
    atom_indices = offsets // 3

    # Load data from global GPU memory into registers
    velocity = tl.load(velocities_ptr + offsets, mask=mask)
    force = tl.load(forces_ptr + offsets, mask=mask)

    # Masses are indexed per-atom, not per-coordinate
    mass = tl.load(masses_ptr + atom_indices, mask=mask)

    # 1. Update velocity: v(t + dt) = v(t) + (f(t) / m) * dt
    vel_new = velocity + (force / mass) * dt

    # 2. Update position: x(t + dt) = x(t) + v(t + dt) * dt
    position = tl.load(positions_ptr + offsets, mask=mask)
    pos_new = position + vel_new * dt

    # Store updated values back to global GPU memory
    tl.store(velocities_ptr + offsets, vel_new, mask=mask)
    tl.store(positions_ptr + offsets, pos_new, mask=mask)

class EulerIntegrator:
    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        dt: float,
    ) -> None:
        # Ensure tensors are contiguous and on GPU
        assert positions.is_contiguous() == True and positions.is_cuda
        assert velocities.is_contiguous() == True and velocities.is_cuda
        assert forces.is_contiguous() == True and forces.is_cuda
        assert masses.is_contiguous() == True and masses.is_cuda

        # Get the number of atoms
        num_atoms = positions.size(0)
        total_elements = num_atoms * 3

        # Launch the CUDA kernel to perform the Euler integration step
        BLOCK_SIZE = 256
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        # Triton automatically extracts the stream and .data_ptr() from PyTorch tensors
        _timer.start("integrator_kernel", TimerClass.Compute, Where.Device)
        euler_step_kernel[grid](
            positions,
            velocities,
            forces,
            masses,
            dt,
            num_atoms,
            BLOCK_SIZE=256  # Passed as a meta-parameter for tl.constexpr
        )
        _timer.stop()

@triton.jit
def verlet_halfstep1_kernel(
    positions_ptr,
    velocities_ptr,
    forces_ptr,
    masses_ptr,
    dt,
    num_floats,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global element offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boolean mask to protect against out-of-bounds array access
    mask = offsets < num_floats

    # Since each atom has 3 spatial coordinates (X, Y, Z),
    # we map the coordinate offset back to its corresponding atom index
    atom_indices = offsets // 3

    # Load data from global GPU memory into registers
    velocity = tl.load(velocities_ptr + offsets, mask=mask)
    force = tl.load(forces_ptr + offsets, mask=mask)

    # Masses are indexed per-atom, not per-coordinate
    mass = tl.load(masses_ptr + atom_indices, mask=mask)

    # 1. Update velocity: v(t + dt) = v(t) + 0.5 * (f(t) / m) * dt
    vel_new = velocity + 0.5 * (force / mass) * dt

    # 2. Update position: x(t + dt) = x(t) + v(t + dt) * dt
    position = tl.load(positions_ptr + offsets, mask=mask)
    pos_new = position + vel_new * dt

    # Store updated values back to global GPU memory
    tl.store(velocities_ptr + offsets, vel_new, mask=mask)
    tl.store(positions_ptr + offsets, pos_new, mask=mask)

@triton.jit
def verlet_halfstep2_kernel(
    velocities_ptr,
    forces_ptr,
    masses_ptr,
    dt,
    num_floats,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global element offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boolean mask to protect against out-of-bounds array access
    mask = offsets < num_floats

    # Since each atom has 3 spatial coordinates (X, Y, Z),
    # we map the coordinate offset back to its corresponding atom index
    atom_indices = offsets // 3

    # Load data from global GPU memory into registers
    velocity = tl.load(velocities_ptr + offsets, mask=mask)
    force = tl.load(forces_ptr + offsets, mask=mask)

    # Masses are indexed per-atom, not per-coordinate
    mass = tl.load(masses_ptr + atom_indices, mask=mask)

    # Update velocity: v(t + dt) = v(t) + 0.5 * (f(t) / m) * dt
    vel_new = velocity + 0.5 * (force / mass) * dt

    # Store updated values back to global GPU memory
    tl.store(velocities_ptr + offsets, vel_new, mask=mask)


class VerletIntegrator:
    def halfstep1(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        dt: float,
    ) -> None:
        # Ensure tensors are contiguous and on GPU
        assert positions.is_contiguous() == True and positions.is_cuda
        assert velocities.is_contiguous() == True and velocities.is_cuda
        assert forces.is_contiguous() == True and forces.is_cuda
        assert masses.is_contiguous() == True and masses.is_cuda
        # Ensure tensors are on the same device
        assert positions.device == velocities.device
        assert positions.device == forces.device
        assert positions.device == masses.device

        # Get the number of floats (atoms * 3)
        num_floats = positions.numel()


        # Launch the CUDA kernel to perform the Euler integration step
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_floats, BLOCK_SIZE),)

        # Triton automatically extracts the stream and .data_ptr() from PyTorch tensors
        _timer.start("integrator_halfstep1", TimerClass.Compute, Where.Device)
        verlet_halfstep1_kernel[grid](
            positions,
            velocities,
            forces,
            masses,
            dt,
            num_floats,
            BLOCK_SIZE=256  # Passed as a meta-parameter for tl.constexpr
        )
        _timer.stop()

    def halfstep2(
        self,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        dt: float,
    ) -> None:
        # Ensure tensors are contiguous and on GPU
        assert velocities.is_contiguous() == True and velocities.is_cuda
        assert forces.is_contiguous() == True and forces.is_cuda
        assert masses.is_contiguous() == True and masses.is_cuda

        # Get the number of floats (atoms * 3)
        num_floats = velocities.numel()


        # Launch the CUDA kernel to perform the Euler integration step
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_floats, BLOCK_SIZE),)

        # Triton automatically extracts the stream and .data_ptr() from PyTorch tensors
        _timer.start("integrator_halfstep2", TimerClass.Compute, Where.Device)
        verlet_halfstep2_kernel[grid](
            velocities,
            forces,
            masses,
            dt,
            num_floats,
            BLOCK_SIZE=256  # Passed as a meta-parameter for tl.constexpr
        )
        _timer.stop()

@triton.jit
def langevin_halfstep1_kernel(
    positions_ptr,
    velocities_ptr,
    forces_ptr,
    masses_ptr,
    noise_ptr,
    gamma,
    kT,
    dt,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global element offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boolean mask to protect against out-of-bounds array access
    mask = offsets < num_elements

    # Since each atom has 3 spatial coordinates (X, Y, Z),
    # we map the coordinate offset back to its corresponding atom index
    atom_indices = offsets // 3

    # Load data from global GPU memory into registers
    velocity = tl.load(velocities_ptr + offsets, mask=mask)
    force = tl.load(forces_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask)

    # Masses are indexed per-atom, not per-coordinate
    mass = tl.load(masses_ptr + atom_indices, mask=mask)

    # Retrieve langevin constants
    c1 = 1.0 - 0.5 * gamma * dt
    c2 = tl.sqrt(gamma * kT * dt / mass)

    # 1. Update velocity: v(t + dt) = v(t) * c1 + 0.5 * (f(t) / m) * dt + c2 * noise(t)
    vel_new = velocity * c1 + 0.5 * (force / mass) * dt + c2 * noise

    # 2. Update position: x(t + dt) = x(t) + v(t + dt) * dt
    position = tl.load(positions_ptr + offsets, mask=mask)
    pos_new = position + vel_new * dt

    # Store updated values back to global GPU memory
    tl.store(velocities_ptr + offsets, vel_new, mask=mask)
    tl.store(positions_ptr + offsets, pos_new, mask=mask)

@triton.jit
def langevin_halfstep2_kernel(
    velocities_ptr,
    forces_ptr,
    masses_ptr,
    noise_ptr,
    gamma,
    kT,
    dt,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global element offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boolean mask to protect against out-of-bounds array access
    mask = offsets < num_elements

    # Since each atom has 3 spatial coordinates (X, Y, Z),
    # we map the coordinate offset back to its corresponding atom index
    atom_indices = offsets // 3

    # Load data from global GPU memory into registers
    velocity = tl.load(velocities_ptr + offsets, mask=mask)
    force = tl.load(forces_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask)

    # Masses are indexed per-atom, not per-coordinate
    mass = tl.load(masses_ptr + atom_indices, mask=mask)

    # Retrieve langevin constants
    c3 = 1.0 / (1.0 + 0.5 * gamma * dt)
    c2 = tl.sqrt(gamma * kT * dt / mass)

    # 1. Update velocity: v(t + dt) = (v(t) + 0.5 * (f(t) / m) * dt + c2 * noise(t)) * c3
    vel_new = (velocity + 0.5 * (force / mass) * dt + c2 * noise) * c3

    # Store updated values back to global GPU memory
    tl.store(velocities_ptr + offsets, vel_new, mask=mask)

class LangevinIntegrator:
    def halfstep1(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        noise: torch.Tensor,
        gamma: float,
        kT: float,
        dt: float,
    ) -> None:
        # Ensure tensors are contiguous and on GPU
        assert positions.is_contiguous() == True and positions.is_cuda
        assert velocities.is_contiguous() == True and velocities.is_cuda
        assert forces.is_contiguous() == True and forces.is_cuda
        assert masses.is_contiguous() == True and masses.is_cuda
        assert noise.is_contiguous() == True and noise.is_cuda

        # Get the number of floats (atoms * 3)
        num_elements = positions.numel()


        # Launch the CUDA kernel to perform the Euler integration step
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

        # Triton automatically extracts the stream and .data_ptr() from PyTorch tensors
        _timer.start("integrator_halfstep1", TimerClass.Compute, Where.Device)
        langevin_halfstep1_kernel[grid](
            positions,
            velocities,
            forces,
            masses,
            noise,
            gamma,
            kT,
            dt,
            num_elements,
            BLOCK_SIZE=256  # Passed as a meta-parameter for tl.constexpr
        )
        _timer.stop()

    def halfstep2(
        self,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        noise: torch.Tensor,
        gamma: float,
        kT: float,
        dt: float,
    ) -> None:
        # Ensure tensors are contiguous and on GPU
        assert velocities.is_contiguous() == True and velocities.is_cuda
        assert forces.is_contiguous() == True and forces.is_cuda
        assert masses.is_contiguous() == True and masses.is_cuda
        assert noise.is_contiguous() == True and noise.is_cuda

        # Get the number of floats (atoms * 3)
        num_elements = velocities.numel()


        # Launch the CUDA kernel to perform the Euler integration step
        BLOCK_SIZE = 256
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

        # Triton automatically extracts the stream and .data_ptr() from PyTorch tensors
        _timer.start("integrator_halfstep2", TimerClass.Compute, Where.Device)
        langevin_halfstep2_kernel[grid](
            velocities,
            forces,
            masses,
            noise,
            gamma,
            kT,
            dt,
            num_elements,
            BLOCK_SIZE=256  # Passed as a meta-parameter for tl.constexpr
        )
        _timer.stop()
