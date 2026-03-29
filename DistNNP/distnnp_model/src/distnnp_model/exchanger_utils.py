import torch.distributed as dist
import torch
from distnnp_model import FeatureExchanger
from distnnp_state import PartitionMessagePassingInfo


def setup_exchanger(
    exchanger: FeatureExchanger,
    local_features: torch.Tensor,
    partition_info: PartitionMessagePassingInfo,
):
    """Helper function to setup the feature exchanger for testing.

    This function performs the necessary steps to initialize the feature exchanger,
    including allocating IPC buffers, exchanging handles, and setting up events.

    Args:
        exchanger (FeatureExchanger): The feature exchanger instance to setup.
        local_features (torch.Tensor): The local features tensor for the current rank.
        partition_info: The partition information containing send/recv details.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_local = local_features.shape[0]
    F = local_features.shape[1]

    total_recv = int(partition_info.global_send_recv_matrix.sum(dim=0)[rank].item())
    total_send = int(partition_info.global_send_recv_matrix.sum(dim=1)[rank].item())

    send_offsets = partition_info.send_offsets
    recv_offsets = partition_info.recv_offsets

    # --- Allocate IPC buffers ---
    x_buffer = FeatureExchanger.allocate_ipc_buffer(n_local + total_recv, F, rank)
    grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(total_send, F, rank)

    # --- Exchange IPC handles via NCCL broadcast ---
    x_handle = exchanger.export_ipc_handle(x_buffer)
    grad_handle = exchanger.export_ipc_handle(grad_recv_buffer)

    handle_size = len(x_handle)
    x_handle_tensor = torch.tensor(
        x_handle, dtype=torch.uint8, device=local_features.device
    )
    grad_handle_tensor = torch.tensor(
        grad_handle, dtype=torch.uint8, device=local_features.device
    )

    all_x_handles_tensor = [
        torch.zeros(handle_size, dtype=torch.uint8, device=local_features.device)
        for _ in range(world_size)
    ]
    all_grad_handles_tensor = [
        torch.zeros(handle_size, dtype=torch.uint8, device=local_features.device)
        for _ in range(world_size)
    ]

    dist.all_gather(all_x_handles_tensor, x_handle_tensor)
    dist.all_gather(all_grad_handles_tensor, grad_handle_tensor)

    all_x_handles = [t.cpu().tolist() for t in all_x_handles_tensor]
    all_grad_handles = [t.cpu().tolist() for t in all_grad_handles_tensor]

    # Gather recv_offsets and num_local from all ranks

    send_local_indices = torch.zeros(
        total_send, dtype=torch.int64, device=local_features.device
    )

    all_recv_offsets_list = [
        torch.zeros(world_size + 1, dtype=torch.int64, device=local_features.device)
        for _ in range(world_size)
    ]
    dist.all_gather(all_recv_offsets_list, recv_offsets)
    all_recv_offsets = torch.stack(all_recv_offsets_list)  # [W, W+
    all_num_local_list = [
        torch.zeros(1, dtype=torch.int64, device=local_features.device)
        for _ in range(world_size)
    ]
    dist.all_gather(
        all_num_local_list,
        torch.tensor([n_local], dtype=torch.int64, device=local_features.device),
    )
    all_num_local = torch.cat(all_num_local_list)  # [W]

    # --- Setup IPC ---
    exchanger.setup_ipc(
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
        F,
        rank,
        world_size,
    )
