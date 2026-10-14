import functools
from abc import ABC, abstractmethod
from dynamis_state import DynamisState, Timer, TimerClass, Where
import torch
from typing import Tuple


def _timed_integrator_step(fn):
    """Where.Both: host span is Python orchestration cost, device span is the
    launched GPU work."""

    @functools.wraps(fn)
    def wrapper(self, state, *args, **kwargs):
        timer = getattr(self, "_timer", None)
        if timer is None:
            timer = Timer()
            self._timer = timer
        timer.start("integrator_step", TimerClass.Compute, Where.Both)
        try:
            return fn(self, state, *args, **kwargs)
        finally:
            timer.stop()

    wrapper._dynamis_timed = True
    return wrapper


class DynamisIntegrator(ABC):
    def __init__(self) -> None:
        self._timer = Timer()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Wraps step() here so new integrators are timed by construction, not convention.
        step = cls.__dict__.get("step")
        if step is None or getattr(step, "_dynamis_timed", False):
            return
        cls.step = _timed_integrator_step(step)

    @abstractmethod
    def step(
        self, state: DynamisState, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update positions and velocities based on the specific integration method.

        This method should be implemented by subclasses to perform the actual integration step.

        Subclasses do not need to add timing: DynamisIntegrator.__init_subclass__ wraps
        this method so every call emits an `integrator_step` record.
        """
        raise NotImplementedError("Subclasses must implement the step method.")

    def _compute_force(
        self, dist_state: DynamisState, model
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the potential model on the current state and return energy and forces.

        Not timed here: `model_forward` is instrumented inside dynamis_model
        (DynamisModel.__init_subclass__), so a model called through any other path is
        measured too.

        Args:
            dist_state (DynamisState): The current state of the system.
            model (DynamisModel): The potential model used to compute forces.
        """
        return model(dist_state)

    def reset(self) -> None:
        pass
