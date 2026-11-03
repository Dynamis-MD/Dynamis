from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from dynamis_model import FeatureExchanger
from dynamis_model.comm import drift_flags
from dynamis_state import DynamisState, PartitionMessagePassingInfo

if TYPE_CHECKING:
    pass


class Exchanger(ABC):
    """Base interface for halo feature exchangers in Dynamis.

    A single Exchanger instance per model is sized up to ``max_feature_dim`` and
    handles all message widths ``1 <= F <= max_feature_dim`` (e.g., 1D atom types,
    3D positions, and high-dimensional GNN latent features across all layers).
    """

    def __init__(self, max_feature_dim: int | None = None) -> None:
        super().__init__()
        self.max_feature_dim: int | None = max_feature_dim
        self.is_ipc_setup: bool = False

    @property
    def is_setup(self) -> bool:
        return self.is_ipc_setup

    def _resolve_max_feature_dim(
        self,
        feature_dim: int | None = None,
        max_feature_dim: int | None = None,
    ) -> int:
        resolved = max_feature_dim or feature_dim or self.max_feature_dim
        if resolved is None or int(resolved) <= 0:
            raise ValueError(
                "A positive feature_dim / max_feature_dim must be provided to setup_exchanger."
            )
        self.max_feature_dim = max(int(resolved), int(self.max_feature_dim or 0))
        return self.max_feature_dim

    @abstractmethod
    def setup_exchanger(
        self,
        feature_dim: int | None = None,
        partition_info: PartitionMessagePassingInfo | None = None,
        device: torch.device | str | None = None,
        *,
        max_feature_dim: int | None = None,
    ) -> None:
        """Allocate and connect communication buffers for the current partition topology.

        Args:
            feature_dim: Maximum feature dimension to support (alias for ``max_feature_dim``).
            partition_info: Partition halo metadata containing CSR offsets and send indices.
            device: CUDA device on which exchanged tensors reside.
            max_feature_dim: Optional keyword alias for ``feature_dim``.
        """
        ...

    def setup(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device | str,
        max_feature_dim: int | None = None,
    ) -> None:
        """Ergonomic alias for :meth:`setup_exchanger`."""
        self.setup_exchanger(
            feature_dim=max_feature_dim,
            partition_info=partition_info,
            device=device,
            max_feature_dim=max_feature_dim,
        )

    @abstractmethod
    def reset_buffers(self) -> None:
        """Release internal communication buffers and reset setup state."""
        ...

    def teardown(self) -> None:
        """Tear down communication resources."""
        self.reset_buffers()

    @abstractmethod
    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Backend-specific implementation of forward feature push."""
        ...

    @abstractmethod
    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Backend-specific implementation of backward feature gradient push."""
        ...

    @staticmethod
    def _is_single_rank() -> bool:
        return dist.is_initialized() and dist.get_world_size() == 1

    def feature_push_forward(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the forward feature push operation for a 2D ``[N_local, F]`` tensor."""
        if self._is_single_rank():
            return features_to_push

        feature_dim = int(features_to_push.shape[-1])
        if (
            not self.is_ipc_setup
            or self.max_feature_dim is None
            or feature_dim > self.max_feature_dim
        ):
            target_dim = max(feature_dim, self.max_feature_dim or 0)
            self.setup_exchanger(
                target_dim, partition_info, features_to_push.device
            )
            self.is_ipc_setup = True

        return self.feature_push_impl(features_to_push, partition_info)

    def feature_push_backward(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the backward pass for the feature push operation."""
        if self._is_single_rank():
            return grad_output

        if not self.is_ipc_setup:
            raise RuntimeError(
                "Exchanger is not setup for backward pass. Call feature_push_forward first."
            )
        return self.feature_push_backward_impl(
            grad_output, input_feature_shape, partition_info
        )

    def push(
        self,
        features_to_push: torch.Tensor,
        state: DynamisState,
    ) -> torch.Tensor:
        """High-level differentiable halo exchange for 1D, 2D, or ND per-atom tensors.

        Automatically handles:
          - 1D tensors ``[N_local]`` (e.g., integer ``atom_types``): unsqueezes to ``[N_local, 1]``,
            casts to ``float32`` for transport, and restores the original 1D shape and dtype.
          - 2D tensors ``[N_local, F]`` (e.g., positions or flat node embeddings).
          - Higher-rank tensors ``[N_local, D1, D2, ...]`` (e.g., spherical harmonic channels):
            flattens trailing dimensions to ``[N_local, D1 * D2 * ...]`` and reshapes on return.
        """
        from dynamis_model.comm.feature_push import FeaturePushToExchange

        orig_dtype = features_to_push.dtype
        orig_shape = features_to_push.shape
        orig_ndim = features_to_push.dim()

        if orig_ndim == 0:
            raise ValueError("Cannot push a 0-D scalar tensor.")

        n_local = orig_shape[0]
        if orig_ndim == 1:
            tensor_2d = features_to_push.unsqueeze(-1)
        elif orig_ndim == 2:
            tensor_2d = features_to_push
        else:
            flat_dim = math.prod(orig_shape[1:])
            tensor_2d = features_to_push.reshape(n_local, flat_dim)

        if tensor_2d.dtype != torch.float32:
            tensor_2d = tensor_2d.to(torch.float32)

        exchanged_2d = FeaturePushToExchange.apply(tensor_2d, state, self)

        if exchanged_2d.dtype != orig_dtype:
            exchanged_2d = exchanged_2d.to(orig_dtype)

        n_total = exchanged_2d.shape[0]
        if orig_ndim == 1:
            return exchanged_2d.reshape(n_total)
        if orig_ndim == 2:
            return exchanged_2d
        return exchanged_2d.reshape(n_total, *orig_shape[1:])

    __call__ = push


class PythonNCCLExchanger(Exchanger):
    """Pure-Python fallback exchanger using ``torch.distributed.all_to_all_single``."""

    def __init__(self, max_feature_dim: int | None = None) -> None:
        super().__init__(max_feature_dim=max_feature_dim)
        self.send_buffer: torch.Tensor | None = None
        self.grad_recv_buffer: torch.Tensor | None = None

    def setup_exchanger(
        self,
        feature_dim: int | None = None,
        partition_info: PartitionMessagePassingInfo | None = None,
        device: torch.device | str | None = None,
        *,
        max_feature_dim: int | None = None,
    ) -> None:
        assert dist.is_initialized(), "Distributed process group is not initialized"
        if feature_dim is not None or max_feature_dim is not None or self.max_feature_dim is not None:
            self._resolve_max_feature_dim(feature_dim, max_feature_dim)
        self.is_ipc_setup = True

    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        device = features_to_push.device
        dtype = features_to_push.dtype
        n_local = features_to_push.shape[0]
        feature_dim = features_to_push.shape[1]
        world_size = dist.get_world_size()

        send_offsets = partition_info.send_offsets.cpu().tolist()
        recv_offsets = partition_info.recv_offsets.cpu().tolist()

        send_splits = [send_offsets[r + 1] - send_offsets[r] for r in range(world_size)]
        recv_splits = [recv_offsets[r + 1] - recv_offsets[r] for r in range(world_size)]

        total_send = send_offsets[-1]
        total_recv = recv_offsets[-1]

        output = torch.empty(
            (n_local + total_recv, feature_dim),
            dtype=dtype,
            device=device,
        )
        output[:n_local].copy_(features_to_push)

        if total_send > 0:
            send_local_indices = partition_info.send_local_indices.to(
                device=device, dtype=torch.int64
            )
            alloc_dim = max(feature_dim, self.max_feature_dim or feature_dim)
            if (
                self.send_buffer is None
                or self.send_buffer.numel() < total_send * feature_dim
                or self.send_buffer.dtype != dtype
                or self.send_buffer.device != device
            ):
                self.send_buffer = torch.empty(
                    (total_send, alloc_dim), dtype=dtype, device=device
                )
            send_tensor = self.send_buffer.view(-1)[: total_send * feature_dim].view(
                total_send, feature_dim
            )
            torch.index_select(
                features_to_push,
                dim=0,
                index=send_local_indices,
                out=send_tensor,
            )
        else:
            send_tensor = torch.empty((0, feature_dim), dtype=dtype, device=device)

        dist.all_to_all_single(
            output=output[n_local:],
            input=send_tensor,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        return output

    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        device = grad_output.device
        dtype = grad_output.dtype
        n_local = input_feature_shape[0]
        feature_dim = input_feature_shape[1]
        world_size = dist.get_world_size()

        send_offsets = partition_info.send_offsets.cpu().tolist()
        recv_offsets = partition_info.recv_offsets.cpu().tolist()

        send_splits = [send_offsets[r + 1] - send_offsets[r] for r in range(world_size)]
        recv_splits = [recv_offsets[r + 1] - recv_offsets[r] for r in range(world_size)]

        total_send = send_offsets[-1]
        total_recv = recv_offsets[-1]

        grad_local = grad_output[:n_local]
        grad_ghost = grad_output[n_local : n_local + total_recv]

        if not grad_ghost.is_contiguous():
            grad_ghost = grad_ghost.contiguous()

        if total_send > 0:
            alloc_dim = max(feature_dim, self.max_feature_dim or feature_dim)
            if (
                self.grad_recv_buffer is None
                or self.grad_recv_buffer.numel() < total_send * feature_dim
                or self.grad_recv_buffer.dtype != dtype
                or self.grad_recv_buffer.device != device
            ):
                self.grad_recv_buffer = torch.empty(
                    (total_send, alloc_dim), dtype=dtype, device=device
                )
            grad_recv = self.grad_recv_buffer.view(-1)[: total_send * feature_dim].view(
                total_send, feature_dim
            )
        else:
            grad_recv = torch.empty((0, feature_dim), dtype=dtype, device=device)

        dist.all_to_all_single(
            output=grad_recv,
            input=grad_ghost,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )

        grad_input = grad_local.clone()
        if total_send > 0:
            send_local_indices = partition_info.send_local_indices.to(
                device=device, dtype=torch.int64
            )
            grad_input.index_add_(0, send_local_indices, grad_recv)

        return grad_input

    def reset_buffers(self) -> None:
        if self.send_buffer is not None:
            del self.send_buffer
            self.send_buffer = None
        if self.grad_recv_buffer is not None:
            del self.grad_recv_buffer
            self.grad_recv_buffer = None
        self.is_ipc_setup = False


class GPUDirectExchanger(Exchanger):
    """C++-backed halo exchanger using one-sided GPUDirect / NVLink / Fabric transport."""

    def __init__(
        self,
        max_feature_dim: int | None = None,
        backend: str = "ipc",
    ) -> None:
        super().__init__(max_feature_dim=max_feature_dim)
        self.exchanger = FeatureExchanger(backend)
        self.send_buffer = None
        self.grad_recv_buffer = None
        self._wired_flags = None

    def reset_buffers(self) -> None:
        """Tear down C++ transport buffers and reset setup state."""
        self.exchanger.teardown()
        self.send_buffer = None
        self.grad_recv_buffer = None
        self._wired_flags = None
        self.is_ipc_setup = False

    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        self._wire_drift_flags(features_to_push.device)
        return self.exchanger.feature_push_forward(features_to_push)

    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        self._wire_drift_flags(grad_output.device)
        return self.exchanger.feature_push_backward(grad_output)

    def _wire_drift_flags(self, device: torch.device) -> None:
        """Carry the process-wide drift-watchdog flags for ``device`` on this exchanger's
        push handshake, or stop carrying them if they were disabled (see drift_flags)."""
        flags = drift_flags.get(device)
        if flags is self._wired_flags:
            return
        if flags is None:
            self.exchanger.clear_watchdog_flags()
        else:
            self.exchanger.set_watchdog_flags(flags.local, flags.global_)
            flags.n_wired += 1
        self._wired_flags = flags

    def setup_exchanger(
        self,
        feature_dim: int | None = None,
        partition_info: PartitionMessagePassingInfo | None = None,
        device: torch.device | str | None = None,
        *,
        max_feature_dim: int | None = None,
    ) -> None:
        """Initialize the C++ feature exchanger with a single all-gather collective."""
        assert dist.is_initialized(), "Distributed process group is not initialized"

        resolved_max_dim = self._resolve_max_feature_dim(feature_dim, max_feature_dim)

        if self._is_single_rank():
            self.is_ipc_setup = True
            return

        if partition_info is None:
            raise ValueError("partition_info is required for GPUDirectExchanger.setup_exchanger")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        resolved_device = torch.device(device) if device is not None else torch.device("cuda", rank)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        device_index = resolved_device.index if resolved_device.index is not None else rank

        # Step 1: Allocate 2-slot max_feature_dim buffers and pack local metadata in C++
        local_meta_bytes = self.exchanger.prepare_setup(
            partition_info,
            resolved_max_dim,
            rank,
            world_size,
            device_index,
        )

        # Step 2: Single collective to exchange all handles, sync counters, and CSR offsets
        local_meta_tensor = torch.tensor(
            local_meta_bytes, dtype=torch.uint8, device=resolved_device
        )
        all_meta_tensor = torch.empty(
            (world_size, local_meta_tensor.numel()),
            dtype=torch.uint8,
            device=resolved_device,
        )
        dist.all_gather_into_tensor(all_meta_tensor, local_meta_tensor)

        # Step 3: Open remote handles and build push targets in C++
        self.exchanger.complete_setup(all_meta_tensor.cpu().tolist())
        self.exchanger.clear_watchdog_flags()
        self._wired_flags = None
        self.is_ipc_setup = True
