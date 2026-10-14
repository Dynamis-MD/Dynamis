"""
Test GPUDirectExchanger forward and backward passes on 2 GPUs.

Uses torch.multiprocessing to spawn 2 processes, each on a different GPU.
Each rank has N_local atoms with F features. A PartitionMessagePassingInfo
struct carries the CSR halo maps. After feature_push_forward, each rank's
ghost region should contain the correct features from the peer; after
feature_push_backward, gradients should accumulate correctly.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from dynamis_state import PartitionMessagePassingInfo
from dynamis_model.comm.exchanger import GPUDirectExchanger

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs",
)

# Local atom count per rank for the 2-rank test topology.
N_LOCAL = (50, 60)


def _build_partition_info(rank, world_size, device):
    """Build a PartitionMessagePassingInfo for a 2-rank test topology.

    Rank 0 sends atoms [0,2,4,6,8]    (5 atoms) to rank 1.
    Rank 1 sends atoms [1,3,5,7,9,11] (6 atoms) to rank 0.

    The global_send_recv_matrix is fully populated (as if all-reduced):
        matrix[r][c] = number of atoms rank r sends to rank c
    """
    info = PartitionMessagePassingInfo()

    # Local (owned) atom indices for this partition. Production partitioners
    # always populate this; GPUDirectExchanger.setup_exchanger reads
    # len(partition_indices) to size the IPC halo buffer.
    info.partition_indices = torch.arange(
        N_LOCAL[rank], dtype=torch.int64, device=device
    )

    # [world_size, world_size]: matrix[sender][receiver] = atom count
    gsrm = torch.zeros(world_size, world_size, dtype=torch.int64, device=device)
    gsrm[0, 1] = 5  # rank 0 → rank 1
    gsrm[1, 0] = 6  # rank 1 → rank 0
    info.global_send_recv_matrix = gsrm

    if rank == 0:
        # send_offsets[r] = start index in send_local_indices for data to rank r
        info.send_offsets = torch.tensor([0, 0, 5], dtype=torch.int64, device=device)
        info.send_local_indices = torch.tensor(
            [0, 2, 4, 6, 8], dtype=torch.int64, device=device
        )
        info.recv_offsets = torch.tensor([0, 0, 6], dtype=torch.int64, device=device)
    else:
        info.send_offsets = torch.tensor([0, 6, 6], dtype=torch.int64, device=device)
        info.send_local_indices = torch.tensor(
            [1, 3, 5, 7, 9, 11], dtype=torch.int64, device=device
        )
        info.recv_offsets = torch.tensor([0, 5, 5], dtype=torch.int64, device=device)

    return info


def _worker_forward_backward(rank, world_size, result_dict):
    """Worker: test GPUDirectExchanger forward and backward passes."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )

    F = 16
    n_local = N_LOCAL[rank]

    partition_info = _build_partition_info(rank, world_size, device)

    total_recv = int(partition_info.global_send_recv_matrix.sum(dim=0)[rank].item())

    # Features: deterministic values so we can verify ghost correctness.
    # atom i on rank r has value r * 1000 + i in every feature channel.
    local_features = torch.zeros(n_local, F, dtype=torch.float32, device=device)
    for i in range(n_local):
        local_features[i, :] = rank * 1000.0 + i

    exchanger = GPUDirectExchanger()

    # --- Forward pass ---
    output = exchanger.feature_push_forward(local_features, partition_info)
    torch.cuda.synchronize()

    # Output shape must be [n_local + total_recv, F]
    expected_shape = (n_local + total_recv, F)
    shape_ok = output.shape == torch.Size(expected_shape)

    # Local region must be unchanged
    local_ok = torch.allclose(output[:n_local], local_features)

    # Ghost region: each ghost slot should contain the sender's value
    ghost = output[n_local:]
    if rank == 0:
        # Received atoms [1,3,5,7,9,11] from rank 1 (value = 1000 + idx)
        expected_ghost = torch.zeros(6, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([1, 3, 5, 7, 9, 11]):
            expected_ghost[i, :] = 1000.0 + idx
    else:
        # Received atoms [0,2,4,6,8] from rank 0 (value = 0 + idx)
        expected_ghost = torch.zeros(5, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([0, 2, 4, 6, 8]):
            expected_ghost[i, :] = float(idx)

    ghost_ok = torch.allclose(ghost, expected_ghost)

    # --- Backward pass ---
    # Gradient of 1.0 everywhere at the output.
    grad_output = torch.ones(
        n_local + total_recv, F, dtype=torch.float32, device=device
    )

    grad_input = exchanger.feature_push_backward(
        grad_output,
        [n_local, F],
        partition_info,
    )
    torch.cuda.synchronize()

    # grad_input must be [n_local, F]
    grad_shape_ok = grad_input.shape == torch.Size((n_local, F))

    # Each atom that was sent also receives a gradient of 1.0 from the peer,
    # so atoms in send_local_indices accumulate gradient 2.0 while others stay 1.0.
    expected_grad = torch.ones(n_local, F, dtype=torch.float32, device=device)
    for idx in partition_info.send_local_indices.cpu().tolist():
        expected_grad[idx, :] = 2.0
    grad_ok = torch.allclose(grad_input, expected_grad)

    exchanger.exchanger.teardown()
    dist.destroy_process_group()

    result_dict[rank] = {
        "shape_ok": shape_ok,
        "local_ok": local_ok,
        "ghost_ok": ghost_ok,
        "grad_shape_ok": grad_shape_ok,
        "grad_ok": grad_ok,
        "output_shape": tuple(output.shape),
        "grad_shape": tuple(grad_input.shape),
    }


def _worker_reset_buffers(rank, world_size, result_dict):
    """Worker: verify that reset_buffers clears state and allows re-setup."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size,
    )

    F = 8
    n_local = N_LOCAL[rank]

    partition_info = _build_partition_info(rank, world_size, device)
    total_recv = int(partition_info.global_send_recv_matrix.sum(dim=0)[rank].item())

    local_features = torch.ones(n_local, F, dtype=torch.float32, device=device)

    exchanger = GPUDirectExchanger()

    # First forward pass triggers setup
    out1 = exchanger.feature_push_forward(local_features, partition_info)
    torch.cuda.synchronize()
    assert (
        exchanger.is_ipc_setup
    ), "Exchanger should be marked as set up after first forward"

    # reset_buffers must clear the setup flag and internal buffers
    exchanger.reset_buffers()
    assert (
        not exchanger.is_ipc_setup
    ), "Exchanger should not be set up after reset_buffers"
    assert exchanger.send_buffer is None, "send_buffer should be None after reset"
    assert (
        exchanger.grad_recv_buffer is None
    ), "grad_recv_buffer should be None after reset"

    # Second forward pass must re-trigger setup and produce identical results
    out2 = exchanger.feature_push_forward(local_features, partition_info)
    torch.cuda.synchronize()
    assert exchanger.is_ipc_setup, "Exchanger should be re-set-up after second forward"

    shapes_match = out1.shape == out2.shape
    values_match = torch.allclose(out1, out2)

    exchanger.exchanger.teardown()
    dist.destroy_process_group()

    result_dict[rank] = {
        "shapes_match": shapes_match,
        "values_match": values_match,
        "output_shape": tuple(out1.shape),
        "expected_shape": (n_local + total_recv, F),
    }


def _worker_backward_without_forward(rank, world_size, result_dict):
    """Worker: verify that calling backward before forward raises RuntimeError."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29502",
        rank=rank,
        world_size=world_size,
    )

    F = 8
    n_local = N_LOCAL[rank]
    total_recv = 6 if rank == 0 else 5

    partition_info = _build_partition_info(rank, world_size, device)

    exchanger = GPUDirectExchanger()
    grad_output = torch.ones(
        n_local + total_recv, F, dtype=torch.float32, device=device
    )

    raised = False
    try:
        exchanger.feature_push_backward(grad_output, [n_local, F], partition_info)
    except RuntimeError:
        raised = True

    dist.destroy_process_group()
    result_dict[rank] = {"raised": raised}


# ---------------------------------------------------------------------------
# Test entry points
# ---------------------------------------------------------------------------


def test_gpu_direct_exchanger_forward_backward_2gpu():
    """GPUDirectExchanger forward and backward produce correct values on 2 GPUs."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_forward_backward,
        args=(world_size, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = result_dict[rank]
        assert r["shape_ok"], f"Rank {rank}: wrong output shape {r['output_shape']}"
        assert r["local_ok"], f"Rank {rank}: local features mutated after forward"
        assert r["ghost_ok"], f"Rank {rank}: ghost features incorrect after forward"
        assert r[
            "grad_shape_ok"
        ], f"Rank {rank}: wrong gradient shape {r['grad_shape']}"
        assert r["grad_ok"], f"Rank {rank}: gradient values incorrect after backward"


def test_gpu_direct_exchanger_reset_buffers_2gpu():
    """reset_buffers clears IPC state; a subsequent forward re-initialises correctly."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_reset_buffers,
        args=(world_size, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = result_dict[rank]
        assert r[
            "shapes_match"
        ], f"Rank {rank}: output shape changed after reset and re-forward"
        assert r[
            "values_match"
        ], f"Rank {rank}: output values differ after reset and re-forward"
        assert (
            r["output_shape"] == r["expected_shape"]
        ), f"Rank {rank}: expected shape {r['expected_shape']}, got {r['output_shape']}"


def test_gpu_direct_exchanger_backward_without_forward_raises_2gpu():
    """Calling feature_push_backward before feature_push_forward raises RuntimeError."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_backward_without_forward,
        args=(world_size, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        assert result_dict[rank][
            "raised"
        ], f"Rank {rank}: expected RuntimeError when calling backward before forward"


if __name__ == "__main__":
    test_gpu_direct_exchanger_forward_backward_2gpu()
    test_gpu_direct_exchanger_reset_buffers_2gpu()
    test_gpu_direct_exchanger_backward_without_forward_raises_2gpu()
