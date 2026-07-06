from abc import ABC, abstractmethod
from distnnp_state import DistNNPState
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
