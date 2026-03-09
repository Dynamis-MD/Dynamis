from __future__ import annotations
import distnnp_state.distnnp_state_impl
import torch
__all__: list[str] = ['FeatureExchanger', 'push_buffer_create']
class FeatureExchanger:
    def __init__(self) -> None:
        ...
    def feature_push_backward(self, arg0: torch.Tensor, arg1: tuple[int, ...], arg2: distnnp_state.distnnp_state_impl.PartitionMessagePassingInfo) -> torch.Tensor:
        """
        Push feature gradients backward to neighbors
        """
    def feature_push_forward(self, arg0: torch.Tensor, arg1: distnnp_state.distnnp_state_impl.PartitionMessagePassingInfo) -> torch.Tensor:
        """
        Push features forward to neighbors
        """
def push_buffer_create(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
    """
    Gather rows into buffer by index
    """
