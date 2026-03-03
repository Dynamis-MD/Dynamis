from abc import ABC, abstractmethod
from distnnp_state import DistNNPState
from distnnp_model import DistNNPModel


class DistNNPIntegrator(ABC):
    @abstractmethod
    def step(self, state: DistNNPState, *args, **kwargs) -> None:
        """
        Update positions and velocities based on the specific integration method.

        This method should be implemented by subclasses to perform the actual integration step.
        """
        raise NotImplementedError("Subclasses must implement the step method.")
