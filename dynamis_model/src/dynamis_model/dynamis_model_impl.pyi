from __future__ import annotations
import collections.abc
import dynamis_state.dynamis_state_impl
import torch
import typing

__all__: list[str] = [
    "CommBackend",
    "FeatureExchanger",
    "IPCExchanger",
    "push_buffer_create",
    "supports_fabric_handles",
]

def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool:
    """
    Check if fabric handles (multi-node NVLink / NVL72) are supported
    """

def push_buffer_create(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor
) -> torch.Tensor:
    """
    Gather rows into buffer by index
    """

class CommBackend:
    pass

class IPCExchanger(CommBackend):
    @staticmethod
    def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool:
        """
        Check if fabric handles are supported on the device
        """

    @staticmethod
    def allocate_ipc_buffer(
        rows: typing.SupportsInt,
        cols: typing.SupportsInt,
        device_index: typing.SupportsInt,
    ) -> torch.Tensor:
        """
        Allocate a tensor via Fabric VMM or raw cudaMalloc (required for IPC/Fabric)
        """

    def __init__(self) -> None: ...
    def export_ipc_handle(self, buffer: torch.Tensor) -> list[int]:
        """
        Export CUDA IPC or Fabric handle for a buffer as bytes
        """

    def setup(
        self,
        x_buffer: torch.Tensor,
        grad_recv_buffer: torch.Tensor,
        all_x_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
        all_grad_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
        send_offsets: torch.Tensor,
        send_local_indices: torch.Tensor,
        recv_offsets: torch.Tensor,
        all_recv_offsets: torch.Tensor,
        all_num_local: torch.Tensor,
        num_local: typing.SupportsInt,
        feature_dim: typing.SupportsInt,
        rank: typing.SupportsInt,
        world_size: typing.SupportsInt,
    ) -> None: ...
    def push_forward(self, x_buffer: torch.Tensor) -> None: ...
    def push_backward(self, grad_at_ghost: torch.Tensor) -> None: ...
    def create_and_export_sync_handle(self) -> list[int]: ...
    def setup_sync(
        self,
        all_sync_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
    ) -> None: ...
    def record_push_complete(self) -> None: ...
    def wait_for_pushes(self) -> None: ...
    def set_watchdog_flags(
        self, local_flag: torch.Tensor, global_flag: torch.Tensor
    ) -> None: ...
    def clear_watchdog_flags(self) -> None: ...
    def teardown(self) -> None: ...
    def is_setup(self) -> bool: ...
    def use_fabric(self) -> bool: ...

class FeatureExchanger:
    @staticmethod
    def supports_fabric_handles(device_id: typing.SupportsInt = -1) -> bool: ...
    @staticmethod
    def allocate_ipc_buffer(
        rows: typing.SupportsInt,
        cols: typing.SupportsInt,
        device_index: typing.SupportsInt,
    ) -> torch.Tensor: ...
    def __init__(self, backend_type: str = "ipc") -> None: ...
    def prepare_setup(
        self,
        partition_info: dynamis_state.dynamis_state_impl.PartitionMessagePassingInfo,
        max_feature_dim: typing.SupportsInt,
        rank: typing.SupportsInt,
        world_size: typing.SupportsInt,
        device_index: typing.SupportsInt,
    ) -> list[int]: ...
    def complete_setup(
        self,
        all_rank_metadata: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
    ) -> None: ...
    @typing.overload
    def feature_push_forward(self, local_features: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def feature_push_forward(
        self,
        local_features: torch.Tensor,
        partition_info: dynamis_state.dynamis_state_impl.PartitionMessagePassingInfo,
    ) -> torch.Tensor: ...
    @typing.overload
    def feature_push_backward(self, grad_output: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def feature_push_backward(
        self,
        grad_output: torch.Tensor,
        grad_input_shape: tuple[int, ...],
        partition_info: dynamis_state.dynamis_state_impl.PartitionMessagePassingInfo,
    ) -> torch.Tensor: ...
    def ce_threshold_bytes(self) -> int: ...
    def set_ce_threshold_bytes(self, threshold: typing.SupportsInt) -> None: ...
    def create_and_export_sync_handle(self) -> list[int]: ...
    def export_ipc_handle(self, buffer: torch.Tensor) -> list[int]: ...
    def setup_sync(
        self,
        all_sync_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
    ) -> None: ...
    def setup_ipc(
        self,
        x_buffer: torch.Tensor,
        grad_recv_buffer: torch.Tensor,
        all_x_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
        all_grad_handles: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
        send_offsets: torch.Tensor,
        send_local_indices: torch.Tensor,
        recv_offsets: torch.Tensor,
        all_recv_offsets: torch.Tensor,
        all_num_local: torch.Tensor,
        num_local: typing.SupportsInt,
        feature_dim: typing.SupportsInt,
        rank: typing.SupportsInt,
        world_size: typing.SupportsInt,
    ) -> None: ...
    def teardown(self) -> None: ...
    def is_setup(self) -> bool: ...
    def is_ipc_setup(self) -> bool: ...
    def use_fabric(self) -> bool: ...
    def max_feature_dim(self) -> int: ...
    def num_local(self) -> int: ...
    def record_push_complete(self) -> None: ...
    def wait_for_pushes(self) -> None: ...
    def set_watchdog_flags(
        self, local_flag: torch.Tensor, global_flag: torch.Tensor
    ) -> None: ...
    def clear_watchdog_flags(self) -> None: ...
