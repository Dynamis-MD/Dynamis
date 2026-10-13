"""
Test IPC feature exchange via FeatureExchanger on 2 GPUs.

Uses torch.multiprocessing to spawn 2 processes, each on a different GPU.
Each rank has N_local atoms with F features. CSR halo maps define which
atoms to send/recv. After feature_push_forward, each rank's ghost region
should contain the correct features from the peer.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from dynamis_model.dynamis_model_impl import FeatureExchanger

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs",
)


def _worker(rank, world_size, result_dict):
    """Worker function running on each GPU."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Init process group for IPC handle exchange via broadcast
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29500",
                            rank=rank, world_size=world_size)

    # --- Test parameters ---
    N_local = [50, 60]  # rank 0 has 50 atoms, rank 1 has 60
    F = 16              # feature dimension
    torch.manual_seed(42 + rank)

    # --- Build CSR halo maps ---
    # Rank 0 sends atoms [0,2,4,6,8] to rank 1
    # Rank 1 sends atoms [1,3,5,7,9,11] to rank 0
    if rank == 0:
        send_local_indices = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64, device=device)
        send_offsets = torch.tensor([0, 0, 5], dtype=torch.int64, device=device)  # nothing to self, 5 to rank 1
        recv_offsets = torch.tensor([0, 0, 6], dtype=torch.int64, device=device)  # nothing from self, 6 from rank 1
    else:
        send_local_indices = torch.tensor([1, 3, 5, 7, 9, 11], dtype=torch.int64, device=device)
        send_offsets = torch.tensor([0, 6, 6], dtype=torch.int64, device=device)  # 6 to rank 0, nothing to self
        recv_offsets = torch.tensor([0, 5, 5], dtype=torch.int64, device=device)  # 5 from rank 0, nothing from self

    n_local = N_local[rank]
    total_send = send_offsets[-1].item()
    total_recv = recv_offsets[-1].item()

    # --- Allocate IPC buffers ---
    exchanger = FeatureExchanger()
    x_buffer = FeatureExchanger.allocate_ipc_buffer(n_local + total_recv, F, rank)
    grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(total_send, F, rank)

    # --- Exchange IPC handles via NCCL broadcast ---
    x_handle = exchanger.export_ipc_handle(x_buffer)
    grad_handle = exchanger.export_ipc_handle(grad_recv_buffer)

    # Gather handles: use all_gather on raw bytes via CPU tensor → NCCL
    handle_size = len(x_handle)
    x_handle_tensor = torch.tensor(x_handle, dtype=torch.uint8, device=device)
    grad_handle_tensor = torch.tensor(grad_handle, dtype=torch.uint8, device=device)

    all_x_handles_tensor = [torch.zeros(handle_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    all_grad_handles_tensor = [torch.zeros(handle_size, dtype=torch.uint8, device=device) for _ in range(world_size)]

    dist.all_gather(all_x_handles_tensor, x_handle_tensor)
    dist.all_gather(all_grad_handles_tensor, grad_handle_tensor)

    all_x_handles = [t.cpu().tolist() for t in all_x_handles_tensor]
    all_grad_handles = [t.cpu().tolist() for t in all_grad_handles_tensor]

    # Gather recv_offsets and num_local from all ranks
    all_recv_offsets_list = [torch.zeros(world_size + 1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_recv_offsets_list, recv_offsets)
    all_recv_offsets = torch.stack(all_recv_offsets_list)  # [W, W+1]

    all_num_local_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_num_local_list, torch.tensor([n_local], dtype=torch.int64, device=device))
    all_num_local = torch.cat(all_num_local_list)  # [W]

    # --- Setup IPC ---
    exchanger.setup_ipc(
        x_buffer, grad_recv_buffer,
        all_x_handles, all_grad_handles,
        send_offsets, send_local_indices, recv_offsets,
        all_recv_offsets, all_num_local,
        n_local, F, rank, world_size,
    )

    # Setup push-sync counter
    sync_handle = exchanger.create_and_export_sync_handle()
    sync_handle_tensor = torch.tensor(sync_handle, dtype=torch.uint8, device=device)
    all_sync_handles_tensor = [torch.zeros(len(sync_handle), dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(all_sync_handles_tensor, sync_handle_tensor)
    all_sync_handles = [t.cpu().tolist() for t in all_sync_handles_tensor]
    exchanger.setup_sync(all_sync_handles)

    # --- Create local features ---
    # Use deterministic values: rank * 1000 + atom_index for easy verification
    local_features = torch.zeros(n_local, F, dtype=torch.float32, device=device)
    for i in range(n_local):
        local_features[i, :] = rank * 1000.0 + i

    # --- Forward push ---
    # Create a dummy partition_info (not used by IPC path, but required by interface)
    from dynamis_state import PartitionMessagePassingInfo
    dummy_info = PartitionMessagePassingInfo()

    output = exchanger.feature_push_forward(local_features, dummy_info)

    torch.cuda.synchronize()

    # --- Verify forward ---
    # output should be [n_local + total_recv, F]
    assert output.shape == (n_local + total_recv, F), \
        f"Rank {rank}: expected shape {(n_local + total_recv, F)}, got {output.shape}"

    # Local region should match local_features
    local_ok = torch.allclose(output[:n_local], local_features)

    # Ghost region should contain peer's features at the sent indices
    ghost = output[n_local:]
    if rank == 0:
        # Ghost from rank 1: atoms [1,3,5,7,9,11] with values 1000+idx
        expected_ghost = torch.zeros(6, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([1, 3, 5, 7, 9, 11]):
            expected_ghost[i, :] = 1000.0 + idx
    else:
        # Ghost from rank 0: atoms [0,2,4,6,8] with values 0+idx
        expected_ghost = torch.zeros(5, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([0, 2, 4, 6, 8]):
            expected_ghost[i, :] = 0.0 + idx

    ghost_ok = torch.allclose(ghost, expected_ghost)

    # --- Test backward ---
    # Create gradient at output (all ones for simplicity)
    grad_output = torch.ones(n_local + total_recv, F, dtype=torch.float32, device=device)

    grad_input = exchanger.feature_push_backward(
        grad_output,
        [n_local, F],
        dummy_info,
    )

    torch.cuda.synchronize()

    # grad_input should be [n_local, F]
    assert grad_input.shape == (n_local, F), \
        f"Rank {rank}: grad_input shape expected {(n_local, F)}, got {grad_input.shape}"

    # Direct gradient is 1.0 for all local atoms.
    # Atoms that were sent also receive gradient 1.0 from the peer.
    # So atoms at send_local_indices should have gradient 2.0, others 1.0.
    expected_grad = torch.ones(n_local, F, dtype=torch.float32, device=device)
    send_idx_cpu = send_local_indices.cpu()
    for idx in send_idx_cpu:
        expected_grad[idx.item(), :] = 2.0

    grad_ok = torch.allclose(grad_input, expected_grad)

    exchanger.teardown()
    dist.destroy_process_group()

    result_dict[rank] = {
        "local_ok": local_ok,
        "ghost_ok": ghost_ok,
        "grad_ok": grad_ok,
        "output_shape": tuple(output.shape),
        "grad_shape": tuple(grad_input.shape),
    }


def test_ipc_exchange_2gpu():
    """Test FeatureExchanger forward and backward on 2 GPUs."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(_worker, args=(world_size, result_dict), nprocs=world_size, join=True)

    for rank in range(world_size):
        r = result_dict[rank]
        assert r["local_ok"], f"Rank {rank}: local features mismatch after forward"
        assert r["ghost_ok"], f"Rank {rank}: ghost features mismatch after forward"
        assert r["grad_ok"], f"Rank {rank}: gradient mismatch after backward"


if __name__ == "__main__":
    test_ipc_exchange_2gpu()
