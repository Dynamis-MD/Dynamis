from __future__ import annotations
import collections.abc
import distnnp_state.distnnp_state_impl
import torch
import typing
__all__: list[str] = ['FeatureExchanger', 'push_buffer_create']
class FeatureExchanger:
    @staticmethod
    def allocate_ipc_buffer(rows: typing.SupportsInt, cols: typing.SupportsInt, device_index: typing.SupportsInt) -> torch.Tensor:
        """
        Allocate a tensor via raw cudaMalloc (required for IPC)
        """
    def __init__(self) -> None:
        ...
    def ce_threshold_bytes(self) -> int:
        ...
    def create_and_export_event(self) -> list[int]:
        """
        Create interprocess event and return IPC handle (64 bytes)
        """
    def export_ipc_handle(self, buffer: torch.Tensor) -> list[int]:
        """
        Export CUDA IPC handle for a buffer as bytes
        """
    def feature_push_backward(self, grad_output: torch.Tensor, grad_input_shape: tuple[int, ...], partition_info: distnnp_state.distnnp_state_impl.PartitionMessagePassingInfo) -> torch.Tensor:
        """
        Push feature gradients backward to neighbors
        """
    def feature_push_forward(self, local_features: torch.Tensor, partition_info: distnnp_state.distnnp_state_impl.PartitionMessagePassingInfo) -> torch.Tensor:
        """
        Push features forward to neighbors
        """
    def is_ipc_setup(self) -> bool:
        ...
    def record_push_complete(self) -> None:
        """
        Record completion event on default stream
        """
    def set_ce_threshold_bytes(self, threshold: typing.SupportsInt) -> None:
        ...
    def setup_events(self, all_event_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]) -> None:
        """
        Open remote IPC events for stream-level sync
        """
    def setup_ipc(self, x_buffer: torch.Tensor, grad_recv_buffer: torch.Tensor, all_x_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], all_grad_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], send_offsets: torch.Tensor, send_local_indices: torch.Tensor, recv_offsets: torch.Tensor, all_recv_offsets: torch.Tensor, all_num_local: torch.Tensor, num_local: typing.SupportsInt, feature_dim: typing.SupportsInt, rank: typing.SupportsInt, world_size: typing.SupportsInt) -> None:
        """
        Open remote IPC handles and build push targets
        """
    def teardown(self) -> None:
        ...
    def wait_for_pushes(self) -> None:
        """
        Wait for all remote pushes to complete on default stream
        """
def push_buffer_create(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
    """
    Gather rows into buffer by index
    """
