import torch
from distnnp_integrator.distnnp_integrator_impl import VerletIntegrator
from distnnp_integrator._data_check import (
    check_devices,
    check_shape_2D,
    check_mass_shape,
    check_shapes_match,
)
from distnnp_integrator.integrator import DistNNPIntegrator
from distnnp_state import DistNNPState
from distnnp_model import DistNNPModel
from distnnp_integrator.masses import TORCH_MASSES


class Verlet(DistNNPIntegrator):
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.cpp_integrator = VerletIntegrator()
        self.prev_forces = None

    def halfstep1_(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        frc: torch.Tensor,
        mass: torch.Tensor,
        dt: float,
    ) -> None:
        """
        Perform the first half-step of the Velocity Verlet method.
        Updates velocities by a half-step and positions by a full-step.

        Args:
            pos (torch.Tensor): Current positions of shape (N, D).
            vel (torch.Tensor): Current velocities of shape (N, D).
            frc (torch.Tensor): Current forces of shape (N, D).
            mass (torch.Tensor): Masses of shape (N,).
            dt (float): Time step size.
        """
        check_devices(pos, vel, frc, mass)
        check_shape_2D(pos)
        check_mass_shape(mass, pos)
        check_shapes_match(pos, vel, frc)
        self.cpp_integrator.halfstep1(pos, vel, frc, mass, dt)

    def halfstep2_(
        self,
        vel: torch.Tensor,
        frc: torch.Tensor,
        mass: torch.Tensor,
        dt: float,
    ) -> None:
        """
        Perform the second half-step of the Velocity Verlet method.
        Updates velocities to the full time step using newly computed forces.

        Args:
            vel (torch.Tensor): Half-step velocities of shape (N, D).
            frc (torch.Tensor): New forces (at updated positions) of shape (N, D).
            mass (torch.Tensor): Masses of shape (N,).
            dt (float): Time step size.
        """
        check_devices(vel, frc, mass)
        check_shape_2D(vel)
        check_mass_shape(mass, vel)
        check_shapes_match(vel, frc)

        self.cpp_integrator.halfstep2(vel, frc, mass, dt)

    def step(self, dist_state: DistNNPState, model: DistNNPModel, dt=None):
        """
        Update positions and velocities based on the Velocity Verlet integration method.

        Args:
            dist_state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DistNNPModel): The potential model used to compute forces based on the current state.
            dt (float, optional): Optional override for the time step size.
        """
        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt

        state = dist_state.local_molecular_info
        atom_masses = TORCH_MASSES.to(state.atom_positions.device)[state.atom_types.reshape(-1).to(torch.long)]

        # 0. Compute forces
        if self.prev_forces is None:
            _, atom_forces = model(dist_state)
        else:
            state.atom_forces = self.prev_forces
    
        # 1. Update v to half-step, and pos to full-step
        self.halfstep1_(
            state.atom_positions,
            state.atom_velocities,
            state.atom_forces,
            atom_masses,
            step_size,
        )

        # 2. Recompute forces at new positions
        energy, new_forces = model(dist_state)
    
        self.prev_forces = new_forces

        # 3. Complete the velocity full-step update
        self.halfstep2_(
            state.atom_velocities, new_forces, atom_masses, step_size
        )
