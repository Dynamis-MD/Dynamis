import torch
from distnnp_integrator.distnnp_integrator_impl import LangevinIntegrator
from distnnp_integrator._data_check import check_devices, check_shapes
from distnnp_integrator.integrator import DistNNPIntegrator


class Langevin(DistNNPIntegrator):
    def __init__(self, dt: float, gamma: float, kT: float) -> None:
        super().__init__()
        self.dt = dt
        self.gamma = gamma
        self.kT = kT
        self.cpp_integrator = LangevinIntegrator()

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
        check_shapes(pos, vel, frc, mass, noise)
        self.cpp_integrator.halfstep1(pos, vel, frc, mass, noise, self.gamma, self.kT, dt)

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
        check_shapes(vel, frc, mass, noise)
        self.cpp_integrator.halfstep2(vel, frc, mass, noise, self.gamma, self.kT, dt)

    def step(self, state, model, dt=None):
        """
        Update positions and velocities based on the Langevin integration method (BBK scheme).

        Args:
            state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DistNNPModel): The potential model used to compute forces based on the current state.
            dt (float, optional): Optional override for the time step size.
        """
        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt
            
        # Langevin requires normally distributed noise for thermal kicks
        noise1 = torch.randn_like(state.vel)
        self.halfstep1_(state.pos, state.vel, state.forces, state.mass, noise1, step_size)
        
        # Recompute forces at new positions
        energy, new_forces = model(state)
        state.forces = new_forces
        
        # Second random noise draw for the second half-step
        noise2 = torch.randn_like(state.vel)
        self.halfstep2_(state.vel, state.forces, state.mass, noise2, step_size)
