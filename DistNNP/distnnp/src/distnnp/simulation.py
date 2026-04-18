from __future__ import annotations

from typing import Optional, Tuple

import torch

from distnnp_model import DistNNPModel
from distnnp_integrator.integrator import DistNNPIntegrator
from distnnp_state import DistNNPState, SimulationConfig


class SimState:
    """Python bridge between C++ DistNNPState and the integrator/model interface.

    The C++ DistNNPState exposes only dist_env, global_molecular_info, and
    local_partition_info.  Integrators and models additionally require pos, vel,
    forces, mass, and simulation_config.  SimState adds these as properties /
    stored fields while keeping the underlying C++ object intact.
    """

    def __init__(
        self,
        dist_state: DistNNPState,
        sim_config: SimulationConfig,
        forces: torch.Tensor,
        mass: torch.Tensor,
    ) -> None:
        self._dist_state = dist_state
        self._sim_config = sim_config
        # Mutable fields that integrators read and update each step.
        self.forces = forces
        self.mass = mass

    # ------------------------------------------------------------------
    # Convenience accessors the integrators and models expect
    # ------------------------------------------------------------------

    @property
    def pos(self) -> torch.Tensor:
        """Local atom positions — alias for local_partition_info.local_atom_positions."""
        return self._dist_state.local_partition_info.local_atom_positions

    @property
    def vel(self) -> torch.Tensor:
        """Local atom velocities — alias for local_partition_info.local_atom_velocities."""
        return self._dist_state.local_partition_info.local_atom_velocities

    @property
    def simulation_config(self) -> SimulationConfig:
        return self._sim_config

    @property
    def local_partition_info(self):
        return self._dist_state.local_partition_info

    @property
    def dist_env(self):
        return self._dist_state.dist_env

    @property
    def global_molecular_info(self):
        return self._dist_state.global_molecular_info


class _EnergyCapture:
    """Thin model wrapper that caches the last energy returned by __call__.

    Integrators call model(state) internally but return None.  Wrapping the
    model with this class lets Simulation.step() surface the energy without
    calling the model a second time.
    """

    def __init__(self, model: DistNNPModel) -> None:
        self._model = model
        self.last_energy: Optional[torch.Tensor] = None

    def __call__(self, state: SimState, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        energy, forces = self._model(state, *args, **kwargs)
        self.last_energy = energy
        return energy, forces


class Simulation:
    """Orchestrates a single MD simulation on one rank.

    Wraps a SimState, a DistNNPModel, and a DistNNPIntegrator.  Each call to
    step() advances the state by one timestep and returns the energy and forces
    computed during that step.

    Args:
        state: Per-rank SimState containing positions, velocities, forces, and masses.
        model: Potential model implementing DistNNPModel.__call__.
        integrator: Time integrator implementing DistNNPIntegrator.step.
    """

    def __init__(
        self,
        state: SimState,
        model: DistNNPModel,
        integrator: DistNNPIntegrator,
    ) -> None:
        self.state = state
        self._capture = _EnergyCapture(model)
        self.integrator = integrator

    @property
    def model(self) -> DistNNPModel:
        return self._capture._model

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the simulation by one timestep.

        Returns:
            (energy, forces) computed during this step.  Energy is a scalar
            tensor (rank-0); forces has shape (n_local_atoms, 3).
        """
        self.integrator.step(self.state, self._capture)
        return self._capture.last_energy, self.state.forces
