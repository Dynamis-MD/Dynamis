from abc import ABC, abstractmethod
from typing import Any
from distnnp_state import DistNNPState
from typing import Tuple, Union
import torch


class DistNNPModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, distributed_state: DistNNPState, *args: Any, **kwds: Any
    ) -> Tuple[Union[float, torch.Tensor], torch.Tensor]:
        """
        Compute the potential energy and forces for a given state.

        This method should be implemented by subclasses to perform the actual computation.

        Args:
            distributed_state (DistNNPState): The current state of the distributed system.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - Energy (torch.Tensor or float): The total potential energy.
                - Forces (torch.Tensor): The forces acting on the atoms/particles.
        """
        ...
