import torch
from distnnp_integrator.distnnp_integrator_impl import VerletIntegrator
from distnnp_integrator._data_check import check_devices, check_shapes
from distnnp_integrator.integrator import DistNNPIntegrator


class Verlet(DistNNPIntegrator):
    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.cpp_integrator = VerletIntegrator()

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
        check_shapes(pos, vel, frc, mass)
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
        check_shapes(vel, frc, mass)
        self.cpp_integrator.halfstep2(vel, frc, mass, dt)

    def step(self, state, model, dt=None):
        """
        Update positions and velocities based on the Velocity Verlet integration method.

        Args:
            state (DistNNPState): The current state of the system, containing positions, velocities, forces, and masses.
            model (DistNNPModel): The potential model used to compute forces based on the current state.
            dt (float, optional): Optional override for the time step size.
        """
        if dt is not None:
            step_size = dt
        else:
            step_size = self.dt
            
        # 1. Update v to half-step, and pos to full-step
        self.halfstep1_(state.pos, state.vel, state.forces, state.mass, step_size)
        
        # 2. Recompute forces at new positions
        energy, new_forces = model(state)
        state.forces = new_forces
        
        # 3. Complete the velocity full-step update
        self.halfstep2_(state.vel, state.forces, state.mass, step_size)
