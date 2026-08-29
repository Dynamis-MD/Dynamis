"""
Test PythonNCCLExchanger forward and backward passes.

Tests:
1. Multi-GPU NCCL forward and backward passes on 2 GPUs (when CUDA >= 2 GPUs available).
2. Dynamic topology updates across simulation timesteps without recreating the exchanger.
3. Single rank identity behavior.
4. reset_buffers clearing state and allowing re-setup.
5. RuntimeError when backward is invoked before forward.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from distnnp_state import PartitionMessagePassingInfo
from distnnp_model.comm.exchanger import PythonNCCLExchanger


def _build_partition_info(rank: int, world_size: int, device: torch.device, topology_variant: int = 0):
    """Build a PartitionMessagePassingInfo for a 2-rank test topology.

    topology_variant 0:
        Rank 0 sends atoms [0, 2, 4, 6, 8]    (5 atoms) to rank 1.
        Rank 1 sends atoms [1, 3, 5, 7, 9, 11] (6 atoms) to rank 0.

    topology_variant 1 (simulating MD atom movement / repartitioning):
        Rank 0 sends atoms [10, 12, 14]       (3 atoms) to rank 1.
        Rank 1 sends atoms [20, 22, 24, 26]   (4 atoms) to rank 0.
    """
    info = PartitionMessagePassingInfo()

    if topology_variant == 0:
        if rank == 0:
            info.partition_indices = torch.arange(50, dtype=torch.int64, device=device)
            info.send_offsets = torch.tensor([0, 0, 5], dtype=torch.int64, device=device)
            info.send_local_indices = torch.tensor(
                [0, 2, 4, 6, 8], dtype=torch.int64, device=device
            )
            info.recv_offsets = torch.tensor([0, 0, 6], dtype=torch.int64, device=device)
        else:
            info.partition_indices = torch.arange(60, dtype=torch.int64, device=device)
            info.send_offsets = torch.tensor([0, 6, 6], dtype=torch.int64, device=device)
            info.send_local_indices = torch.tensor(
                [1, 3, 5, 7, 9, 11], dtype=torch.int64, device=device
            )
            info.recv_offsets = torch.tensor([0, 5, 5], dtype=torch.int64, device=device)
    else:
        if rank == 0:
            info.partition_indices = torch.arange(50, dtype=torch.int64, device=device)
            info.send_offsets = torch.tensor([0, 0, 3], dtype=torch.int64, device=device)
            info.send_local_indices = torch.tensor(
                [10, 12, 14], dtype=torch.int64, device=device
            )
            info.recv_offsets = torch.tensor([0, 0, 4], dtype=torch.int64, device=device)
        else:
            info.partition_indices = torch.arange(60, dtype=torch.int64, device=device)
            info.send_offsets = torch.tensor([0, 4, 4], dtype=torch.int64, device=device)
            info.send_local_indices = torch.tensor(
                [20, 22, 24, 26], dtype=torch.int64, device=device
            )
            info.recv_offsets = torch.tensor([0, 3, 3], dtype=torch.int64, device=device)

    return info


def _worker_forward_backward(rank: int, world_size: int, backend: str, port: int, result_dict):
    """Worker: test PythonNCCLExchanger forward and backward passes."""
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    N_local = [50, 60]
    F = 16
    n_local = N_local[rank]

    partition_info = _build_partition_info(rank, world_size, device, topology_variant=0)
    total_recv = 6 if rank == 0 else 5

    # Features: atom i on rank r has value r * 1000 + i in every feature channel
    local_features = torch.zeros(n_local, F, dtype=torch.float32, device=device)
    for i in range(n_local):
        local_features[i, :] = rank * 1000.0 + i

    exchanger = PythonNCCLExchanger()

    # --- Forward pass (Topology 0) ---
    output = exchanger.feature_push_forward(local_features, partition_info)
    if backend == "nccl":
        torch.cuda.synchronize()

    expected_shape = (n_local + total_recv, F)
    shape_ok = output.shape == torch.Size(expected_shape)
    local_ok = torch.allclose(output[:n_local], local_features)

    ghost = output[n_local:]
    if rank == 0:
        expected_ghost = torch.zeros(6, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([1, 3, 5, 7, 9, 11]):
            expected_ghost[i, :] = 1000.0 + idx
    else:
        expected_ghost = torch.zeros(5, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([0, 2, 4, 6, 8]):
            expected_ghost[i, :] = float(idx)

    ghost_ok = torch.allclose(ghost, expected_ghost)

    # --- Backward pass (Topology 0) ---
    grad_output = torch.ones(n_local + total_recv, F, dtype=torch.float32, device=device)

    grad_input = exchanger.feature_push_backward(
        grad_output,
        torch.Size([n_local, F]),
        partition_info,
    )
    if backend == "nccl":
        torch.cuda.synchronize()

    grad_shape_ok = grad_input.shape == torch.Size((n_local, F))

    expected_grad = torch.ones(n_local, F, dtype=torch.float32, device=device)
    for idx in partition_info.send_local_indices.cpu().tolist():
        expected_grad[idx, :] = 2.0
    grad_ok = torch.allclose(grad_input, expected_grad)

    # --- Step 2: Dynamic topology update in MD without resetting exchanger ---
    partition_info_step2 = _build_partition_info(rank, world_size, device, topology_variant=1)
    total_recv_step2 = 4 if rank == 0 else 3

    output_step2 = exchanger.feature_push_forward(local_features, partition_info_step2)
    if backend == "nccl":
        torch.cuda.synchronize()

    dynamic_shape_ok = output_step2.shape == torch.Size((n_local + total_recv_step2, F))
    ghost_step2 = output_step2[n_local:]
    if rank == 0:
        expected_ghost_step2 = torch.zeros(4, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([20, 22, 24, 26]):
            expected_ghost_step2[i, :] = 1000.0 + idx
    else:
        expected_ghost_step2 = torch.zeros(3, F, dtype=torch.float32, device=device)
        for i, idx in enumerate([10, 12, 14]):
            expected_ghost_step2[i, :] = float(idx)

    dynamic_ghost_ok = torch.allclose(ghost_step2, expected_ghost_step2)

    grad_output_step2 = torch.ones(n_local + total_recv_step2, F, dtype=torch.float32, device=device)
    grad_input_step2 = exchanger.feature_push_backward(
        grad_output_step2,
        torch.Size([n_local, F]),
        partition_info_step2,
    )
    if backend == "nccl":
        torch.cuda.synchronize()

    expected_grad_step2 = torch.ones(n_local, F, dtype=torch.float32, device=device)
    for idx in partition_info_step2.send_local_indices.cpu().tolist():
        expected_grad_step2[idx, :] = 2.0
    dynamic_grad_ok = torch.allclose(grad_input_step2, expected_grad_step2)

    dist.destroy_process_group()

    result_dict[rank] = {
        "shape_ok": shape_ok,
        "local_ok": local_ok,
        "ghost_ok": ghost_ok,
        "grad_shape_ok": grad_shape_ok,
        "grad_ok": grad_ok,
        "dynamic_shape_ok": dynamic_shape_ok,
        "dynamic_ghost_ok": dynamic_ghost_ok,
        "dynamic_grad_ok": dynamic_grad_ok,
    }


def _worker_reset_buffers(rank: int, world_size: int, backend: str, port: int, result_dict):
    """Worker: verify that reset_buffers clears state and allows re-setup."""
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    N_local = [50, 60]
    F = 8
    n_local = N_local[rank]

    partition_info = _build_partition_info(rank, world_size, device)
    total_recv = 6 if rank == 0 else 5

    local_features = torch.ones(n_local, F, dtype=torch.float32, device=device)

    exchanger = PythonNCCLExchanger()

    # First forward pass triggers setup
    out1 = exchanger.feature_push_forward(local_features, partition_info)
    if backend == "nccl":
        torch.cuda.synchronize()
    assert exchanger.is_ipc_setup, "Exchanger should be marked as set up after first forward"

    # reset_buffers must clear the setup flag and internal buffers
    exchanger.reset_buffers()
    assert not exchanger.is_ipc_setup, "Exchanger should not be set up after reset_buffers"
    assert exchanger.send_buffer is None, "send_buffer should be None after reset"
    assert exchanger.grad_recv_buffer is None, "grad_recv_buffer should be None after reset"

    # Second forward pass must re-trigger setup and produce identical results
    out2 = exchanger.feature_push_forward(local_features, partition_info)
    if backend == "nccl":
        torch.cuda.synchronize()
    assert exchanger.is_ipc_setup, "Exchanger should be re-set-up after second forward"

    shapes_match = out1.shape == out2.shape
    values_match = torch.allclose(out1, out2)

    dist.destroy_process_group()

    result_dict[rank] = {
        "shapes_match": shapes_match,
        "values_match": values_match,
        "output_shape": tuple(out1.shape),
        "expected_shape": (n_local + total_recv, F),
    }


def _worker_backward_without_forward(rank: int, world_size: int, backend: str, port: int, result_dict):
    """Worker: verify that calling backward before forward raises RuntimeError."""
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    N_local = [50, 60]
    F = 8
    n_local = N_local[rank]
    total_recv = 6 if rank == 0 else 5

    partition_info = _build_partition_info(rank, world_size, device)

    exchanger = PythonNCCLExchanger()
    grad_output = torch.ones(n_local + total_recv, F, dtype=torch.float32, device=device)

    raised = False
    try:
        exchanger.feature_push_backward(grad_output, torch.Size([n_local, F]), partition_info)
    except RuntimeError:
        raised = True

    dist.destroy_process_group()
    result_dict[rank] = {"raised": raised}


# ---------------------------------------------------------------------------
# Test entry points
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs",
)
def test_python_nccl_exchanger_forward_backward_2gpu():
    """PythonNCCLExchanger forward and backward produce correct values on 2 GPUs (NCCL)."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_forward_backward,
        args=(world_size, "nccl", 29510, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = result_dict[rank]
        assert r["shape_ok"], f"Rank {rank}: shape mismatch"
        assert r["local_ok"], f"Rank {rank}: local features corrupted"
        assert r["ghost_ok"], f"Rank {rank}: ghost features incorrect"
        assert r["grad_shape_ok"], f"Rank {rank}: grad shape mismatch"
        assert r["grad_ok"], f"Rank {rank}: gradients incorrect"
        assert r["dynamic_shape_ok"], f"Rank {rank}: dynamic shape mismatch"
        assert r["dynamic_ghost_ok"], f"Rank {rank}: dynamic ghost mismatch"
        assert r["dynamic_grad_ok"], f"Rank {rank}: dynamic grad mismatch"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs",
)
def test_python_nccl_exchanger_reset_buffers_2gpu():
    """PythonNCCLExchanger reset_buffers clears state and re-setup succeeds (NCCL)."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_reset_buffers,
        args=(world_size, "nccl", 29511, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        r = result_dict[rank]
        assert r["shapes_match"], f"Rank {rank}: shape mismatch after reset"
        assert r["values_match"], f"Rank {rank}: value mismatch after reset"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs",
)
def test_python_nccl_exchanger_backward_without_forward_2gpu():
    """Calling feature_push_backward before feature_push_forward raises RuntimeError (NCCL)."""
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _worker_backward_without_forward,
        args=(world_size, "nccl", 29512, result_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        assert result_dict[rank]["raised"], f"Rank {rank} did not raise RuntimeError"


def test_python_nccl_exchanger_single_rank():
    """PythonNCCLExchanger acts as identity when world_size == 1 or single rank."""
    exchanger = PythonNCCLExchanger()
    n_local = 10
    F = 16
    features = torch.randn(n_local, F)
    info = PartitionMessagePassingInfo()

    out = exchanger.feature_push_forward(features, info)
    assert torch.equal(out, features)

    grad_out = torch.randn(n_local, F)
    grad_in = exchanger.feature_push_backward(grad_out, torch.Size([n_local, F]), info)
    assert torch.equal(grad_in, grad_out)
