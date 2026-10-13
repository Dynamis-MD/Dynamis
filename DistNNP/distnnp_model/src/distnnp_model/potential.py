import functools
from abc import ABC, abstractmethod
from typing import Any
from dynamis_state import (
    DistNNPState,
    PartitionMessagePassingInfo,
    Timer,
    TimerClass,
    Where,
)
from typing import Tuple, Union
import torch


def _timed_model_forward(fn):
    """Where.Both: the region is Python-orchestrated but launches async CUDA
    work, so either timer alone would miss half the cost."""

    @functools.wraps(fn)
    def wrapper(self, distributed_state, *args, **kwargs):
        timer = Timer()
        timer.start("model_forward", TimerClass.Compute, Where.Both)
        try:
            return fn(self, distributed_state, *args, **kwargs)
        finally:
            timer.stop()

    wrapper._distnnp_timed = True
    return wrapper


class DistNNPModel(ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Wraps __call__ per-subclass since it's abstract on the base class.
        call = cls.__dict__.get("__call__")
        if call is None or getattr(call, "_distnnp_timed", False):
            return
        cls.__call__ = _timed_model_forward(call)

    @abstractmethod
    def __call__(
        self, distributed_state: DistNNPState, *args: Any, **kwds: Any
    ) -> Tuple[Union[float, torch.Tensor], torch.Tensor]:
        """
        Compute the potential energy and forces for a given state.

        This method should be implemented by subclasses to perform the actual computation.

        Subclasses do not need to add timing: DistNNPModel.__init_subclass__ wraps this
        method so every call emits a `model_forward` record.

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

    @abstractmethod
    def initialize_buffers(
        self, partition_info: PartitionMessagePassingInfo, device: torch.device
    ) -> None:
        """
        Initialize any necessary buffers or internal state based on the distributed state.

        This method should be implemented by subclasses to set up any required data structures
        or buffers that depend on the current partitioning of the system.

        Args:
            distributed_state (PartitionMessagePassingInfo): The current state of the distributed system.
        """
        ...
