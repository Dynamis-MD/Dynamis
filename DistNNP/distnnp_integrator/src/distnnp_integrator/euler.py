import torch
from distnnp_integrator.distnnp_integrator_impl import EulerIntegrator
from distnnp_integrator._data_check import (
    check_devices,
    check_shape_2D,
    check_mass_shape,
    check_shapes_match,
)
from distnnp_integrator.integrator import DistNNPIntegrator
from distnnp_state import DistNNPState, Timer
from distnnp_model import DistNNPModel
from distnnp_integrator.masses import TORCH_MASSES

class Euler(DistNNPIntegrator):
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self._timer = Timer()

    @staticmethod
    def step_(
        pos: torch.Tensor,
        vel: torch.Tensor,
        frc: torch.Tensor,
        mass: torch.Tensor,
        dt: float,
    ) -> None:
        """
        Update positions and velocities using the Euler method.

        Args:
            pos (torch.Tensor): Current positions of shape (N, D).
            vel (torch.Tensor): Current velocities of shape (N, D).
            frc (torch.Tensor): Current forces of shape (N, D).
            mass (torch.Tensor): Masses of shape (N,).
            dt (float): Time step size.
        """
        # Do python-side checks before calling the C++ implementation

        check_devices(pos, vel, frc, mass)
        check_shape_2D(pos)
        check_mass_shape(mass, pos)
        check_shapes_match(pos, vel, frc)

        EulerIntegrator().step(pos, vel, frc, mass, dt)

    def reset(self) -> None:
        pass

    def step(self, state: DistNNPState, model: DistNNPModel, dt=None):
        """
        Update positions and velocities based on the Euler integration method.

        Args:
            state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DistNNPModel): The potential model used to compute forces based on the current state.
        """
        molecular_info = state.local_molecular_info

        energy, new_forces = self._compute_force(state, model)

        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt

        atom_masses = TORCH_MASSES.to(molecular_info.atom_positions.device)[molecular_info.atom_types.reshape(-1).to(torch.long)]

        # integrator_step is timed in C++ (euler_integrator.cu)
        self.step_(
            molecular_info.atom_positions,
            molecular_info.atom_velocities,
            new_forces,
            atom_masses,
            step_size,
        )
        return energy, new_forces
