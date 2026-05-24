import torch.distributed as dist
import torch
from distnnp_model import FeatureExchanger
from distnnp_state import PartitionMessagePassingInfo
from abc import ABC, abstractmethod


class Exchanger(ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.is_ipc_setup = False

    @abstractmethod
    def setup_exchanger(
        self,
        feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ):
        """Helper function to setup the feature exchanger for testing.

        This function performs the necessary steps to initialize the feature exchanger,
        including allocating IPC buffers, exchanging handles, and setting up events.

        Args:
            exchanger (FeatureExchanger): The feature exchanger instance to setup.
            local_features (torch.Tensor): The local features tensor for the current rank.
            partition_info: The partition information containing send/recv details.
        """
        ...

    @abstractmethod
    def reset_buffers(self):
        """Reset the internal buffers of the exchanger to prepare for the next communication round."""
        ...

    @abstractmethod
    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Low-level implementation of the feature push operation.

        This method should be implemented by subclasses to perform the actual communication
        using the underlying communication engine (e.g., CUDA IPC, NCCL, MPI).

        Args:
            features_to_push (torch.Tensor): The features that need to be pushed to neighboring partitions.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation.
        """
        ...

    @abstractmethod
    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Low-level implementation of the backward pass for the feature push operation.

        This method should be implemented by subclasses to perform the actual communication
        for the backward pass using the underlying communication engine.

        Args:
            grad_output (torch.Tensor): The gradients received from the neighboring partitions for the features that were
                pushed in the forward pass.
            input_feature_shape (torch.Size): The shape of the input features that were pushed in the forward pass.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The gradients for the input features that were pushed in the forward pass.
        """
        ...

    @staticmethod
    def _is_single_rank() -> bool:
        return dist.is_initialized() and dist.get_world_size() == 1

    def feature_push_forward(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the forward feature push operation.

        Args:
            features_to_push (torch.Tensor): The features that need to be pushed to neighboring partitions.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation.
        """
        # With a single rank there are no peers, no halo, and a 0-byte
        # grad_recv_buffer that cudaIpcGetMemHandle would reject. Skip IPC
        # entirely and act as identity.
        if self._is_single_rank():
            return features_to_push

        if not self.is_ipc_setup:
            feature_dim = features_to_push.shape[-1]
            device = features_to_push.device

            self.setup_exchanger(feature_dim, partition_info, device)
            self.is_ipc_setup = True

        return self.feature_push_impl(features_to_push, partition_info)

    def feature_push_backward(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the backward pass for the feature push operation.

        Args:
            grad_output (torch.Tensor): The gradients received from the neighboring partitions for the features that were
                pushed in the forward pass.
            input_feature_shape (torch.Size): The shape of the input features that were pushed in the forward pass.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The gradients for the input features that were pushed in the forward pass.
        """
        if self._is_single_rank():
            return grad_output

        if not self.is_ipc_setup:
            raise RuntimeError(
                "Exchanger is not setup for backward pass. Call feature_push_forward first."
            )
        return self.feature_push_backward_impl(
            grad_output, input_feature_shape, partition_info
        )


class GPUDirectExchanger(Exchanger):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.exchanger = FeatureExchanger()
        self.is_ipc_setup = False
        self.send_buffer = None
        self.grad_recv_buffer = None

    def reset_buffers(self):
        """Reset the internal buffers of the exchanger to prepare for the next communication round."""
        if self.send_buffer is not None:
            del self.send_buffer
            self.send_buffer = None
        if self.grad_recv_buffer is not None:
            del self.grad_recv_buffer
            self.grad_recv_buffer = None
        self.is_ipc_setup = False

    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Low-level implementation of the feature push operation using GPU Direct.

        Args:
            features_to_push (torch.Tensor): The features that need to be pushed to neighboring partitions.
            partition_info: The partition information containing send/recv details.
        Returns:
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation.
        """
        # The actual implementation of the feature push using GPU Direct would go here.
        # This is a placeholder to illustrate where the communication logic would be implemented.
        return self.exchanger.feature_push_forward(features_to_push, partition_info)

    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Low-level implementation of the backward pass for the feature push operation using GPU Direct.

        Args:
            grad_output (torch.Tensor): The gradients received from the neighboring partitions for the features that were
                pushed in the forward pass.
            input_feature_shape (torch.Size): The shape of the input features that were pushed in the forward pass.
            partition_info: The partition information containing send/recv details.
        Returns:
            torch.Tensor: The gradients for the input features that were pushed in the forward pass.
        """
        # The actual implementation of the backward pass for the feature push using GPU Direct would go here.
        # This is a placeholder to illustrate where the communication logic for the backward pass would be implemented.
        return self.exchanger.feature_push_backward(
            grad_output, input_feature_shape, partition_info
        )

    def setup_exchanger(
        self,
        feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ):
        """Helper function to setup the feature exchanger for testing.

        This function performs the necessary steps to initialize the feature exchanger,
        including allocating IPC buffers, exchanging handles, and setting up events.

        Args:
            exchanger (FeatureExchanger): The feature exchanger instance to setup.
            local_features (torch.Tensor): The local features tensor for the current rank.
            partition_info: The partition information containing send/recv details.
        """

        assert dist.is_initialized(), "Distributed process group is not initialized"

        # With a single rank there are no peers, no halo, and a 0-byte
        # grad_recv_buffer that cudaIpcGetMemHandle would reject. Skip IPC
        # setup entirely; feature_push_forward/backward act as identity.
        if self._is_single_rank():
            self.is_ipc_setup = True
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        n_local = len(partition_info.partition_indices)

        send_offsets = partition_info.send_offsets
        recv_offsets = partition_info.recv_offsets
        
        total_send = int(send_offsets[-1].item())
        total_recv = int(recv_offsets[-1].item())

        # --- Allocate IPC buffers ---
        x_buffer = FeatureExchanger.allocate_ipc_buffer(
            n_local + total_recv, feature_dim, rank
        )
        grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(
            total_send, feature_dim, rank
        )

        # --- Exchange IPC handles via NCCL broadcast ---
        x_handle = self.exchanger.export_ipc_handle(x_buffer)
        grad_handle = self.exchanger.export_ipc_handle(grad_recv_buffer)

        handle_size = len(x_handle)
        x_handle_tensor = torch.tensor(x_handle, dtype=torch.uint8, device=device)
        grad_handle_tensor = torch.tensor(grad_handle, dtype=torch.uint8, device=device)

        all_x_handles_tensor = [
            torch.zeros(handle_size, dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]
        all_grad_handles_tensor = [
            torch.zeros(handle_size, dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]

        dist.all_gather(all_x_handles_tensor, x_handle_tensor)
        dist.all_gather(all_grad_handles_tensor, grad_handle_tensor)

        all_x_handles = [t.cpu().tolist() for t in all_x_handles_tensor]
        all_grad_handles = [t.cpu().tolist() for t in all_grad_handles_tensor]

        # Gather recv_offsets and num_local from all ranks

        # Local indices of the atoms this rank pushes to its peers, in CSR order
        # matching send_offsets. build_halo_maps populates this; using a zero
        # tensor here would make every send push local atom 0.
        send_local_indices = partition_info.send_local_indices.to(
            dtype=torch.int64, device=device
        )

        all_recv_offsets_list = [
            torch.zeros(world_size + 1, dtype=torch.int64, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(all_recv_offsets_list, recv_offsets)
        all_recv_offsets = torch.stack(all_recv_offsets_list)  # [W, W+1]

        all_num_local_list = [
            torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
        ]
        dist.all_gather(
            all_num_local_list,
            torch.tensor([n_local], dtype=torch.int64, device=device),
        )
        all_num_local = torch.cat(all_num_local_list)  # [W]

        # --- Setup IPC ---
        self.exchanger.setup_ipc(
            x_buffer,
            grad_recv_buffer,
            all_x_handles,
            all_grad_handles,
            send_offsets,
            send_local_indices,
            recv_offsets,
            all_recv_offsets,
            all_num_local,
            n_local,
            feature_dim,
            rank,
            world_size,
        )

        # --- Setup push-sync counter ---
        sync_handle = self.exchanger.create_and_export_sync_handle()
        sync_handle_tensor = torch.tensor(
            sync_handle, dtype=torch.uint8, device=device
        )
        all_sync_handles_tensor = [
            torch.zeros(
                len(sync_handle), dtype=torch.uint8, device=device
            )
            for _ in range(world_size)
        ]
        dist.all_gather(all_sync_handles_tensor, sync_handle_tensor)
        all_sync_handles = [t.cpu().tolist() for t in all_sync_handles_tensor]
        self.exchanger.setup_sync(all_sync_handles)
