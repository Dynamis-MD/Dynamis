from __future__ import annotations
import collections.abc
import distnnp_state.distnnp_state_impl
import torch
import typing

__all__: list[str] = ['FeatureExchanger', 'IPCExchanger', 'push_buffer_create', 'supports_fabric_handles']

def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool:
    """
    Check if fabric handles (multi-node NVLink / NVL72) are supported
    """

def push_buffer_create(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
    """
    Gather rows into buffer by index
    """

class IPCExchanger:
    @staticmethod
    def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool:
        """
        Check if fabric handles are supported on the device
        """
    @staticmethod
    def allocate_ipc_buffer(rows: typing.SupportsInt, cols: typing.SupportsInt, device_index: typing.SupportsInt) -> torch.Tensor:
        """
        Allocate a tensor via Fabric VMM or raw cudaMalloc (required for IPC/Fabric)
        """
    def __init__(self) -> None:
        ...
    def export_ipc_handle(self, buffer: torch.Tensor) -> list[int]:
        """
        Export CUDA IPC or Fabric handle for a buffer as bytes
        """
    def setup(self, x_buffer: torch.Tensor, grad_recv_buffer: torch.Tensor, all_x_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], all_grad_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], send_offsets: torch.Tensor, send_local_indices: torch.Tensor, recv_offsets: torch.Tensor, all_recv_offsets: torch.Tensor, all_num_local: torch.Tensor, num_local: typing.SupportsInt, feature_dim: typing.SupportsInt, rank: typing.SupportsInt, world_size: typing.SupportsInt) -> None:
        ...
    def push_forward(self, x_buffer: torch.Tensor) -> None:
        ...
    def push_backward(self, grad_at_ghost: torch.Tensor) -> None:
        ...
    def create_and_export_sync_handle(self) -> list[int]:
        ...
    def setup_sync(self, all_sync_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]) -> None:
        ...
    def record_push_complete(self) -> None:
        ...
    def wait_for_pushes(self) -> None:
        ...
    def teardown(self) -> None:
        ...
    def is_setup(self) -> bool:
        ...
    def use_fabric(self) -> bool:
        ...

class FeatureExchanger:
    @staticmethod
    def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool:
        """
        Check if fabric handles are supported on the device
        """
    @staticmethod
    def allocate_ipc_buffer(rows: typing.SupportsInt, cols: typing.SupportsInt, device_index: typing.SupportsInt) -> torch.Tensor:
        """
        Allocate a tensor via Fabric VMM or raw cudaMalloc (required for IPC/Fabric)
        """
    def __init__(self) -> None:
        ...
    def ce_threshold_bytes(self) -> int:
        ...
    def create_and_export_sync_handle(self) -> list[int]:
        """
        Allocate IPC/Fabric sync counter and return its handle bytes
        """
    def export_ipc_handle(self, buffer: torch.Tensor) -> list[int]:
        """
        Export CUDA IPC or Fabric handle for a buffer as bytes
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
    def use_fabric(self) -> bool:
        ...
    def record_push_complete(self) -> None:
        """
        Bump sync counter on default stream after push completes
        """
    def set_ce_threshold_bytes(self, threshold: typing.SupportsInt) -> None:
        ...
    def setup_sync(self, all_sync_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]) -> None:
        """
        Open remote IPC/Fabric sync counters for stream-level push synchronization
        """
    def setup_ipc(self, x_buffer: torch.Tensor, grad_recv_buffer: torch.Tensor, all_x_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], all_grad_handles: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], send_offsets: torch.Tensor, send_local_indices: torch.Tensor, recv_offsets: torch.Tensor, all_recv_offsets: torch.Tensor, all_num_local: torch.Tensor, num_local: typing.SupportsInt, feature_dim: typing.SupportsInt, rank: typing.SupportsInt, world_size: typing.SupportsInt) -> None:
        """
        Open remote IPC/Fabric handles and build push targets
        """
    def teardown(self) -> None:
        ...
    def wait_for_pushes(self) -> None:
        """
        Wait on default stream until all peers' sync counters reach current step
        """
