import pytest
import torch
import torch.distributed as dist
from distnnp_model.distnnp_model_impl import FeatureExchanger
from distnnp_state import PartitionMessagePassingInfo

# Requires 2+ GPUs
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs",
)


def _setup():
    """Init torch.distributed and return (exchanger, rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Extract raw ncclComm_t pointer from PyTorch's NCCL backend
    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("cuda"))
    comm_ptr = backend.nccl_comm_pointer()

    exchanger = FeatureExchanger(comm_ptr, rank, world_size)
    return exchanger, rank, world_size


def _make_info(send_offsets, send_local_indices, recv_offsets, recv_ghost_indices):
    """Build a PartitionMessagePassingInfo with just the halo map fields."""
    info = PartitionMessagePassingInfo()
    info.send_offsets = torch.tensor(send_offsets, dtype=torch.int64, device="cuda")
    info.send_local_indices = torch.tensor(send_local_indices, dtype=torch.int64, device="cuda")
    info.recv_offsets = torch.tensor(recv_offsets, dtype=torch.int64, device="cuda")
    info.recv_ghost_indices = torch.tensor(recv_ghost_indices, dtype=torch.int64, device="cuda")
    return info


# Rank 0 sends 2 atoms to rank 1, receives 1 from rank 1
# Verifies output shape is [num_ghost, F]
@pytest.mark.parametrize("F", range(1, 129))
@pytest.mark.parametrize("seed", range(5))
def test_forward_output_shape(F, seed):
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    torch.manual_seed(seed)

    if rank == 0:
        info = _make_info([0, 0, 2], [1, 3], [0, 0, 1], [5])
        expected_ghost = 1
    else:
        info = _make_info([0, 1, 1], [2], [0, 2, 2], [5, 6])
        expected_ghost = 2

    features = torch.randn(5, F, dtype=torch.float32, device="cuda")
    result = exchanger.feature_push_forward(features, info)

    assert result.shape == (expected_ghost, F)
    assert result.dtype == torch.float32
    assert result.device.type == "cuda"

    dist.barrier()


# Rank 0 sends 2 atoms to rank 1, receives 1 from rank 1
# Verifies backward output shape is [num_local, F]
@pytest.mark.parametrize("F", range(1, 129))
@pytest.mark.parametrize("seed", range(5))
def test_backward_output_shape(F, seed):
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    torch.manual_seed(seed)
    num_local = 5

    if rank == 0:
        # Receives 1 ghost in forward, so grad_output is [1, F]
        info = _make_info([0, 0, 2], [1, 3], [0, 0, 1], [5])
        num_ghost = 1
    else:
        # Receives 2 ghosts in forward, so grad_output is [2, F]
        info = _make_info([0, 1, 1], [2], [0, 2, 2], [5, 6])
        num_ghost = 2

    grad_output = torch.randn(num_ghost, F, dtype=torch.float32, device="cuda")
    result = exchanger.feature_push_backward(grad_output, (num_local, F), info)

    assert result.shape == (num_local, F)
    assert result.dtype == torch.float32
    assert result.device.type == "cuda"

    dist.barrier()


# Rank 0 has known features, sends atoms 0 and 2 to rank 1
# Rank 1 verifies it received the exact values
def test_forward_values():
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    if rank == 0:
        features = torch.tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]],
            dtype=torch.float32, device="cuda",
        )
        # Send atoms 0 and 2 to rank 1
        info = _make_info([0, 0, 2], [0, 2], [0, 0, 1], [3])
    else:
        features = torch.tensor(
            [[10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0],
             [16.0, 17.0, 18.0]],
            dtype=torch.float32, device="cuda",
        )
        # Send atom 1 to rank 0
        info = _make_info([0, 1, 1], [1], [0, 2, 2], [3, 4])

    result = exchanger.feature_push_forward(features, info)

    if rank == 0:
        # Rank 0 should receive atom 1 from rank 1: [13, 14, 15]
        expected = torch.tensor([[13.0, 14.0, 15.0]], dtype=torch.float32, device="cuda")
        assert torch.allclose(result, expected)
    else:
        # Rank 1 should receive atoms 0 and 2 from rank 0: [1,2,3] and [7,8,9]
        expected = torch.tensor(
            [[1.0, 2.0, 3.0],
             [7.0, 8.0, 9.0]],
            dtype=torch.float32, device="cuda",
        )
        assert torch.allclose(result, expected)

    dist.barrier()


# Rank 1 has ghost gradients for atoms that rank 0 sent
# After backward, verify rank 0 gets gradients at the correct positions
def test_backward_values():
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    num_local = 4
    F = 3

    if rank == 0:
        # Forward sent atoms 0 and 2 to rank 1, received 1 ghost from rank 1
        info = _make_info([0, 0, 2], [0, 2], [0, 0, 1], [4])
        # Grad at the 1 ghost slot this rank received
        grad_output = torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32, device="cuda")
    else:
        # Forward sent atom 1 to rank 0, received 2 ghosts from rank 0
        info = _make_info([0, 1, 1], [1], [0, 2, 2], [3, 4])
        # Grad at the 2 ghost slots this rank received
        grad_output = torch.tensor(
            [[1.0, 1.0, 1.0],
             [2.0, 2.0, 2.0]],
            dtype=torch.float32, device="cuda",
        )

    result = exchanger.feature_push_backward(grad_output, (num_local, F), info)

    if rank == 0:
        # Rank 0 sent atoms 0 and 2 — should get back grads from rank 1's ghost slots
        # Atom 0 gets [1,1,1], atom 2 gets [2,2,2], atoms 1 and 3 get zeros
        expected = torch.tensor(
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0],
             [2.0, 2.0, 2.0],
             [0.0, 0.0, 0.0]],
            dtype=torch.float32, device="cuda",
        )
        assert torch.allclose(result, expected)
    else:
        # Rank 1 sent atom 1 — should get back grad from rank 0's ghost slot
        # Atom 1 gets [9,9,9], atoms 0, 2, 3 get zeros
        expected = torch.tensor(
            [[0.0, 0.0, 0.0],
             [9.0, 9.0, 9.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            dtype=torch.float32, device="cuda",
        )
        assert torch.allclose(result, expected)

    dist.barrier()


# Same atom sent to rank 1 twice (appears twice in send_local_indices)
# Backward should sum both gradient contributions at that position
def test_backward_gradient_accumulation():
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    num_local = 3
    F = 3

    if rank == 0:
        # Send atom 1 twice to rank 1 (simulates sending to 2 ranks)
        info = _make_info([0, 0, 2], [1, 1], [0, 0, 1], [3])
        grad_output = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32, device="cuda")
    else:
        # Send 1 atom to rank 0, receive 2 ghosts
        info = _make_info([0, 1, 1], [0], [0, 2, 2], [3, 4])
        grad_output = torch.tensor(
            [[3.0, 3.0, 3.0],
             [7.0, 7.0, 7.0]],
            dtype=torch.float32, device="cuda",
        )

    result = exchanger.feature_push_backward(grad_output, (num_local, F), info)

    if rank == 0:
        # Atom 1 was sent twice — both grads [3,3,3] and [7,7,7] should accumulate
        expected = torch.tensor(
            [[0.0, 0.0, 0.0],
             [10.0, 10.0, 10.0],
             [0.0, 0.0, 0.0]],
            dtype=torch.float32, device="cuda",
        )
        assert torch.allclose(result, expected)

    dist.barrier()


# No atoms sent or received on either rank
# Forward should return [0, F]
@pytest.mark.parametrize("F", range(1, 129))
def test_forward_empty_exchange(F):
    exchanger, _, world_size = _setup()
    assert world_size == 2

    info = _make_info([0, 0, 0], [], [0, 0, 0], [])

    features = torch.randn(5, F, dtype=torch.float32, device="cuda")
    result = exchanger.feature_push_forward(features, info)

    assert result.shape == (0, F)

    dist.barrier()


# No atoms sent or received on either rank
# Backward should return zeros of grad_input_shape
@pytest.mark.parametrize("F", range(1, 129))
def test_backward_empty_exchange(F):
    exchanger, _, world_size = _setup()
    assert world_size == 2

    num_local = 5
    info = _make_info([0, 0, 0], [], [0, 0, 0], [])

    grad_output = torch.empty(0, F, dtype=torch.float32, device="cuda")
    result = exchanger.feature_push_backward(grad_output, (num_local, F), info)

    assert result.shape == (num_local, F)
    assert torch.allclose(result, torch.zeros(num_local, F, dtype=torch.float32, device="cuda"))

    dist.barrier()

# Large forward exchange
# Verifies correctness at scale with random data
@pytest.mark.parametrize("i", range(16))
@pytest.mark.parametrize("seed", range(5))
def test_forward_large(i, seed):
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    torch.manual_seed(seed + rank * 1000)
    F = 2**i
    N = 10000
    num_send = 1000

    features = torch.randn(N, F, dtype=torch.float32, device="cuda")
    send_indices = torch.randperm(N, device="cuda")[:num_send].sort().values

    if rank == 0:
        info = _make_info([0, 0, num_send], send_indices.tolist(), [0, 0, num_send], list(range(N, N + num_send)))
    else:
        info = _make_info([0, num_send, num_send], send_indices.tolist(), [0, num_send, num_send], list(range(N, N + num_send)))

    # Share features and indices so each rank knows what the other sent
    rank0_features = features.clone() if rank == 0 else torch.empty(N, F, dtype=torch.float32, device="cuda")
    rank1_features = features.clone() if rank == 1 else torch.empty(N, F, dtype=torch.float32, device="cuda")
    rank0_indices = send_indices.clone() if rank == 0 else torch.empty(num_send, dtype=torch.int64, device="cuda")
    rank1_indices = send_indices.clone() if rank == 1 else torch.empty(num_send, dtype=torch.int64, device="cuda")
    dist.broadcast(rank0_features, src=0)
    dist.broadcast(rank1_features, src=1)
    dist.broadcast(rank0_indices, src=0)
    dist.broadcast(rank1_indices, src=1)

    result = exchanger.feature_push_forward(features, info)

    assert result.shape == (num_send, F)

    if rank == 0:
        # Should have received rank 1's atoms at rank1_indices
        expected = rank1_features[rank1_indices]
        assert torch.allclose(result, expected)
    else:
        # Should have received rank 0's atoms at rank0_indices
        expected = rank0_features[rank0_indices]
        assert torch.allclose(result, expected)

    dist.barrier()


# Send features forward, then use the received values as grad_output in backward
# Verifies the gradient path is consistent with the forward path
@pytest.mark.parametrize("F", range(1, 129))
@pytest.mark.parametrize("seed", range(5))
def test_forward_backward_roundtrip(F, seed):
    exchanger, rank, world_size = _setup()
    assert world_size == 2

    torch.manual_seed(seed)
    num_local = 4

    if rank == 0:
        features = torch.randn(num_local, F, dtype=torch.float32, device="cuda")
        # Send atoms 1 and 2 to rank 1, receive 1 from rank 1
        info = _make_info([0, 0, 2], [1, 2], [0, 0, 1], [4])
    else:
        features = torch.randn(num_local, F, dtype=torch.float32, device="cuda")
        # Send atom 0 to rank 0, receive 2 from rank 0
        info = _make_info([0, 1, 1], [0], [0, 2, 2], [4, 5])

    # Forward
    ghost_features = exchanger.feature_push_forward(features, info)

    # Use received ghost features as grad_output for backward
    grad_input = exchanger.feature_push_backward(ghost_features, (num_local, F), info)

    if rank == 0:
        # Rank 0 sent atoms 1 and 2 — grads should match those original features
        assert torch.allclose(grad_input[1], features[1])
        assert torch.allclose(grad_input[2], features[2])
        # Unsent atoms should be zero
        assert torch.allclose(grad_input[0], torch.zeros(F, device="cuda"))
        assert torch.allclose(grad_input[3], torch.zeros(F, device="cuda"))
    else:
        # Rank 1 sent atom 0 — grad should match that original feature
        assert torch.allclose(grad_input[0], features[0])
        # Unsent atoms should be zero
        assert torch.allclose(grad_input[1], torch.zeros(F, device="cuda"))
        assert torch.allclose(grad_input[2], torch.zeros(F, device="cuda"))
        assert torch.allclose(grad_input[3], torch.zeros(F, device="cuda"))

    dist.barrier()
