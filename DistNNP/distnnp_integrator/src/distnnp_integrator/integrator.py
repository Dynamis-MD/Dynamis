from abc import ABC, abstractmethod
from distnnp_state import DistNNPState, TimerClass, Where
import torch
from typing import Tuple


class DistNNPIntegrator(ABC):
    @abstractmethod
    def step(self, state: DistNNPState, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update positions and velocities based on the specific integration method.

        This method should be implemented by subclasses to perform the actual integration step.
        """
        raise NotImplementedError("Subclasses must implement the step method.")

    def _compute_force(self, dist_state: DistNNPState, model) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the potential model on the current state and return energy and forces.

        Args:
            dist_state (DistNNPState): The current state of the system.
            model (DistNNPModel): The potential model used to compute forces.
        """
        device_index = dist_state.local_molecular_info.atom_positions.device.index
        self._timer.start("model_forward", TimerClass.Compute, Where.Device, device_index)
        energy, atom_forces = model(dist_state)
        self._timer.stop()
        return energy, atom_forces

    def reset(self) -> None:
        pass
