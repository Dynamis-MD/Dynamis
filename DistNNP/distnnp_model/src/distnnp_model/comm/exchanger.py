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
        

        # Buffer sizes must come from this rank's own CSR offsets, which
        # build_halo_maps fills locally and correctly. The global_send_recv_matrix
        # is only populated for this rank's own row (it is never all-reduced in the
        # pipeline), so sum(dim=0)[rank] wrongly returns num_local instead of the
        # true ghost-recv count — under-sizing the ghost region and causing IPC
        # peers to overflow halo_exchanger_buffer (and, at W=1, appending num_local phantom
        # rows that corrupt the graph).
        total_recv = int(recv_offsets[-1].item())  # ghost atoms this rank receives
        total_send = int(send_offsets[-1].item())  # atoms this rank sends to peers

        # --- Allocate IPC buffers ---
        # cudaIpcGetMemHandle rejects a 0-byte allocation, so allocate at least
        # one row for the handle; CSR offsets bound all real reads/writes, so the
        # padding row is never touched (matters at W=1 where total_send == 0).
        halo_exchanger_buffer = FeatureExchanger.allocate_ipc_buffer(
            max(1, n_local + total_recv), feature_dim, rank
        )
        self.send_buffer = halo_exchanger_buffer
        grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(
            max(1, total_send), feature_dim, rank
        )
        self.grad_recv_buffer = grad_recv_buffer

        # --- Exchange IPC handles via NCCL broadcast ---
        halo_exchanger_handle = self.exchanger.export_ipc_handle(halo_exchanger_buffer)
        grad_handle = self.exchanger.export_ipc_handle(grad_recv_buffer)

        handle_size = len(halo_exchanger_handle)
        halo_exchanger_handle_tensor = torch.tensor(halo_exchanger_handle, dtype=torch.uint8, device=device)
        grad_handle_tensor = torch.tensor(grad_handle, dtype=torch.uint8, device=device)

        all_halo_exchanger_handles_tensor = [
            torch.zeros(handle_size, dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]
        all_grad_handles_tensor = [
            torch.zeros(handle_size, dtype=torch.uint8, device=device)
            for _ in range(world_size)
        ]

        dist.all_gather(all_halo_exchanger_handles_tensor, halo_exchanger_handle_tensor)
        dist.all_gather(all_grad_handles_tensor, grad_handle_tensor)

        all_halo_exchanger_handles = [t.cpu().tolist() for t in all_halo_exchanger_handles_tensor]
        all_grad_handles = [t.cpu().tolist() for t in all_grad_handles_tensor]

        # Gather recv_offsets and num_local from all ranks

        # Local indices of the atoms this rank pushes to its peers, in CSR order
        # matching send_offsets. build_halo_maps populates this; using a zero
        # tensor here would make every send push local atom 0.
        send_local_indices = partition_info.send_local_indices.to(
            device=device, dtype=torch.int64
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
            halo_exchanger_buffer,
            grad_recv_buffer,
            all_halo_exchanger_handles,
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
        self.is_ipc_setup = True

class NCCLExchanger(Exchanger):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def setup_exchanger(
        self,
        local_features: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ):
        """torch.distributed handles NCCL setup; no additional state needed."""
        assert dist.is_initialized(), "Distributed process group is not initialized"

    def reset_buffers(self):
        """No persistent buffers to reset for NCCL."""
        pass

    def feature_push_impl(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Exchange halo features via NCCL all-to-all.

        Gathers atoms at send_local_indices, performs an all_to_all_single
        exchange, and returns the full [N_local + N_ghost, F] tensor with
        the ghost region populated.

        Args:
            features_to_push: [N_local, F] local atom features.
            partition_info: CSR send/recv maps for this rank.
        Returns:
            [N_local + N_ghost, F] tensor: local features concatenated with
            received ghost features ordered by source rank.
        """
        world_size = dist.get_world_size()
        F = features_to_push.shape[1]
        device = features_to_push.device

        send_offsets = partition_info.send_offsets.cpu()
        recv_offsets = partition_info.recv_offsets.cpu()

        send_counts = [int(send_offsets[r + 1] - send_offsets[r]) for r in range(world_size)]
        recv_counts = [int(recv_offsets[r + 1] - recv_offsets[r]) for r in range(world_size)]
        total_recv = int(recv_offsets[-1])

        # Pack atoms to send in CSR rank order.
        send_buf = features_to_push[partition_info.send_local_indices]  # [total_send, F]
        recv_buf = torch.empty(total_recv, F, dtype=features_to_push.dtype, device=device)

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )

        return torch.cat([features_to_push, recv_buf], dim=0)

    def feature_push_backward_impl(
        self,
        grad_output: torch.Tensor,
        input_feature_shape: torch.Size,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Reverse NCCL all-to-all to propagate ghost gradients back to owners.

        Sends ghost gradients back to the rank that originally provided each
        atom, then scatter-adds the received gradients into the local gradient
        tensor at the positions that were pushed in the forward pass.

        Args:
            grad_output: [N_local + N_ghost, F] gradient at the forward output.
            input_feature_shape: Shape of the forward input, i.e. [N_local, F].
            partition_info: CSR send/recv maps for this rank.
        Returns:
            [N_local, F] gradient for the forward input features.
        """
        world_size = dist.get_world_size()
        n_local = input_feature_shape[0]
        F = grad_output.shape[1]
        device = grad_output.device

        send_offsets = partition_info.send_offsets.cpu()
        recv_offsets = partition_info.recv_offsets.cpu()

        send_counts = [int(send_offsets[r + 1] - send_offsets[r]) for r in range(world_size)]
        recv_counts = [int(recv_offsets[r + 1] - recv_offsets[r]) for r in range(world_size)]
        total_send = int(send_offsets[-1])

        # Ghost grads to return to atom owners (forward recv → backward send).
        grad_ghost = grad_output[n_local:].contiguous()  # [total_recv, F]
        grad_owner = torch.empty(total_send, F, dtype=grad_output.dtype, device=device)

        # Roles are reversed vs forward: we send recv_counts[r] to rank r
        # and receive send_counts[r] from rank r.
        dist.all_to_all_single(
            grad_owner,
            grad_ghost,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
        )

        # Gradient for local atoms: identity path + accumulated peer contributions.
        grad_input = grad_output[:n_local].clone()
        grad_input.index_add_(0, partition_info.send_local_indices, grad_owner)

        return grad_input
