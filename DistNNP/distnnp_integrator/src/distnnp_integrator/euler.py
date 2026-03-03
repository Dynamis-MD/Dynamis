import torch
from distnnp_integrator.distnnp_integrator_impl import EulerIntegrator
from distnnp_integrator._data_check import check_devices, check_shapes
from distnnp_integrator.integrator import DistNNPIntegrator


class Euler(DistNNPIntegrator):
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt

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
        check_shapes(pos, vel, frc, mass)

        EulerIntegrator().step(pos, vel, frc, mass, dt)

    def step(self, state, model, dt=None):
        """
        Update positions and velocities based on the Euler integration method.

        Args:
            state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DistNNPModel): The potential model used to compute forces based on the current state.
        """
        # Compute forces using the provided model
        energy, forces = model(state)
        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt

        # Update positions and velocities using the Euler method
        self.step_(state.pos, state.vel, forces, state.mass, step_size)
