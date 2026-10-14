import torch
from dynamis_integrator.dynamis_integrator_impl import LangevinIntegrator
from dynamis_integrator._data_check import (
    check_devices,
    check_shape_2D,
    check_mass_shape,
    check_shapes_match,
)
from dynamis_integrator.integrator import DistNNPIntegrator
from dynamis_state import DistNNPState
from dynamis_model import DynamisModel
from dynamis_integrator.masses import TORCH_MASSES


class Langevin(DistNNPIntegrator):
    def __init__(self, dt: float, gamma: float, kT: float) -> None:
        super().__init__()
        self.dt = dt
        self.gamma = gamma
        self.kT = kT
        self.cpp_integrator = LangevinIntegrator()
        self.prev_forces = None

    def reset(self) -> None:
        self.prev_forces = None

    def halfstep1_(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        frc: torch.Tensor,
        mass: torch.Tensor,
        noise: torch.Tensor,
        dt: float,
    ) -> None:
        """
        Perform the first half-step of the Langevin (BBK) dynamics method.
        Updates velocities by a half-step including friction and noise, and positions by a full-step.

        Args:
            pos (torch.Tensor): Current positions of shape (N, D).
            vel (torch.Tensor): Current velocities of shape (N, D).
            frc (torch.Tensor): Current forces of shape (N, D).
            mass (torch.Tensor): Masses of shape (N,).
            noise (torch.Tensor): Normally distributed random noise of shape (N, D).
            dt (float): Time step size.
        """
        check_devices(pos, vel, frc, mass, noise)
        check_shape_2D(pos)
        check_mass_shape(mass, pos)
        check_shapes_match(pos, vel, frc, noise)

        self.cpp_integrator.halfstep1(
            pos, vel, frc, mass, noise, self.gamma, self.kT, dt
        )

    def halfstep2_(
        self,
        vel: torch.Tensor,
        frc: torch.Tensor,
        mass: torch.Tensor,
        noise: torch.Tensor,
        dt: float,
    ) -> None:
        """
        Perform the second half-step of the Langevin (BBK) dynamics method.
        Completes the velocity update for the full time step using new forces and friction/noise.

        Args:
            vel (torch.Tensor): Half-step velocities of shape (N, D).
            frc (torch.Tensor): New forces (at updated positions) of shape (N, D).
            mass (torch.Tensor): Masses of shape (N,).
            noise (torch.Tensor): Normally distributed random noise of shape (N, D).
            dt (float): Time step size.
        """
        check_devices(vel, frc, mass, noise)
        check_shape_2D(vel)
        check_mass_shape(mass, vel)
        check_shapes_match(vel, frc, noise)
        self.cpp_integrator.halfstep2(vel, frc, mass, noise, self.gamma, self.kT, dt)

    def step(self, dist_state: DistNNPState, model: DynamisModel, dt=None):
        """
        Update positions and velocities based on the Langevin integration method (BBK scheme).

        Args:
            state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DynamisModel): The potential model used to compute forces based on the current state.
            dt (float, optional): Optional override for the time step size.
        """

        state = dist_state.local_molecular_info
        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt

        # Langevin requires normally distributed noise for thermal kicks
        noise1 = torch.randn_like(state.atom_velocities)
        atom_masses = TORCH_MASSES.to(state.atom_positions.device)[
            state.atom_types.reshape(-1).to(torch.long)
        ]

        if self.prev_forces is None:
            _, atom_forces = self._compute_force(dist_state, model)
        else:
            atom_forces = self.prev_forces

        # Timed in C++ (langevin_integrator.cu), nested inside integrator_step
        self.halfstep1_(
            state.atom_positions,
            state.atom_velocities,
            atom_forces,
            atom_masses,
            noise1,
            step_size,
        )

        # Recompute forces at new positions
        energy, atom_forces = self._compute_force(dist_state, model)
        self.prev_forces = atom_forces

        # Second random noise draw for the second half-step
        noise2 = torch.randn_like(state.atom_velocities)
        # Timed in C++ (langevin_integrator.cu), nested inside integrator_step
        self.halfstep2_(
            state.atom_velocities,
            atom_forces,
            atom_masses,
            noise2,
            step_size,
        )
        return energy, atom_forces
