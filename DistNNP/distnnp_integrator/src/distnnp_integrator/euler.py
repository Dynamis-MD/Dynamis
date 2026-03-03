import torch
from distnnp_integrator.distnnp_integrator_impl import EulerIntegrator
from distnnp_integrator._data_check import check_devices, check_shapes


class EulerIntegrator:
    @staticmethod
    def step(
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
