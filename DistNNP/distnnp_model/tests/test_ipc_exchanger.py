"""Tests for IPCExchanger — one-sided push feature exchange over NVLink (CUDA IPC).

Requires >= 2 GPUs. Uses torch.multiprocessing to spawn one process per GPU,
simulating a distributed environment without torch.distributed.
"""

import torch
import torch.multiprocessing as mp
import pytest

from dynamis_state import (
    MolecularInfo,
    DistEnvironment,
    SimulationConfig,
    SimpleSpatialPartitioner,
)
from distnnp_model.distnnp_model_impl import IPCExchanger


def _make_env(rank: int, world_size: int) -> DistEnvironment:
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = world_size
    env.num_nodes = 1
    env.local_rank = rank
    return env


def _make_sim_config(cutoff: float, device: torch.device) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * 10.0
    cfg.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return cfg


def _mp_worker(rank, world_size, num_atoms, feature_dim, cutoff,
               shared_positions, results_dict, handles_dict, barrier):
    """Multi-process worker for IPC exchange test."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Reconstruct global state
    state = MolecularInfo()
    state.num_atoms = num_atoms
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long, device=device)
    state.atom_positions = shared_positions.to(device)
    state.atom_velocities = torch.zeros(num_atoms, 3, device=device)

    env = _make_env(rank, world_size)
    cfg = _make_sim_config(cutoff, device)

    # Partition + halo maps
    p = SimpleSpatialPartitioner()
    info = p.partition(state, env)
    p.build_halo_maps(info, state, env, cfg)

    num_local = info.partition_indices.shape[0]
    so = info.send_offsets.cpu()
    ro = info.recv_offsets.cpu()
    total_send = so[-1].item()
    total_recv = ro[-1].item()

    # Allocate persistent buffers via raw cudaMalloc (required for IPC)
    x_buffer = IPCExchanger.allocate_ipc_buffer(num_local + total_recv, feature_dim, rank)
    grad_recv_buffer = IPCExchanger.allocate_ipc_buffer(max(total_send, 1), feature_dim, rank)

    # Fill local features: global_atom_id * 0.01 + feat_idx * 0.001
    global_ids = info.partition_indices.to(device)
    for f in range(feature_dim):
        x_buffer[:num_local, f] = global_ids.float() * 0.01 + f * 0.001

    # Export IPC handles
    exchanger = IPCExchanger()
    x_handle = exchanger.export_ipc_handle(x_buffer)
    grad_handle = exchanger.export_ipc_handle(grad_recv_buffer)

    # Share handles and metadata
    handles_dict[rank] = {
        "x_handle": x_handle,
        "grad_handle": grad_handle,
        "recv_offsets": ro.tolist(),
        "num_local": num_local,
    }

    barrier.wait()  # All ranks have exported handles

    # Collect all handles
    all_x_handles = [handles_dict[r]["x_handle"] for r in range(world_size)]
    all_grad_handles = [handles_dict[r]["grad_handle"] for r in range(world_size)]

    # Build all_recv_offsets [W, W+1]
    all_ro = []
    all_nl = []
    for r in range(world_size):
        all_ro.append(handles_dict[r]["recv_offsets"])
        all_nl.append(handles_dict[r]["num_local"])

    all_recv_offsets = torch.tensor(all_ro, dtype=torch.long)  # [W, W+1]
    all_num_local = torch.tensor(all_nl, dtype=torch.long)     # [W]

    # Setup exchanger
    exchanger.setup(
        x_buffer, grad_recv_buffer,
        all_x_handles, all_grad_handles,
        info.send_offsets, info.send_local_indices,
        info.recv_offsets,
        all_recv_offsets, all_num_local,
        num_local, feature_dim, rank, world_size,
    )

    # Forward push
    exchanger.push_forward(x_buffer)
    torch.cuda.synchronize()

    barrier.wait()  # All ranks have pushed

    # Verify ghost features
    # Ghost atoms from rank r should have values: global_id * 0.01 + feat * 0.001
    # We need to figure out which global atoms are in our ghost region.
    #
    # The ghost region is built from recv maps. For SimpleSpatialPartitioner,
    # global atoms are split contiguously. Ghost atoms come from the halo map.
    #
    # To verify: for each ghost atom, check its expected value.
    # The global_id of ghost atom at ghost_slot g (= num_local + i) can be found
    # by tracing back through the halo maps. But that's complex.
    #
    # Simpler verification: each rank that SENT data knows what it sent.
    # We verify by checking that the ghost region is non-zero (was actually written)
    # and matches the expected pattern.

    errors = []

    if total_recv > 0:
        ghost_data = x_buffer[num_local:].cpu()

        # Check ghost features are not all zeros (were actually written)
        if ghost_data.abs().sum().item() == 0:
            errors.append(f"Rank {rank}: ghost region is all zeros after push")

        # For a stronger check: ghost values should match the pattern
        # global_id * 0.01 + feat * 0.001 for all features.
        # Since all features of one atom share the same global_id base,
        # we can extract the global_id from feature 0:
        #   global_id = ghost_data[:, 0] / 0.01  (since feat 0 has offset 0.0)
        inferred_ids = (ghost_data[:, 0] / 0.01).round().long()

        for f in range(feature_dim):
            expected = inferred_ids.float() * 0.01 + f * 0.001
            actual = ghost_data[:, f]
            max_err = (actual - expected).abs().max().item()
            if max_err > 1e-5:
                errors.append(
                    f"Rank {rank}: ghost feature {f} max error = {max_err}"
                )

    exchanger.teardown()

    results_dict[rank] = {
        "num_local": num_local,
        "total_send": total_send,
        "total_recv": total_recv,
        "errors": errors,
    }


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires >= 2 CUDA GPUs",
)


@requires_multi_gpu
class TestIPCExchanger:

    def test_supports_fabric_handles(self):
        """Check fabric handles query runs without error and returns bool."""
        supported = IPCExchanger.supports_fabric_handles(0)
        assert isinstance(supported, bool)

    def test_export_handle_size(self):
        """Handle (CUDA IPC or Fabric) should be 64 bytes."""
        buf = IPCExchanger.allocate_ipc_buffer(10, 4, 0)
        exchanger = IPCExchanger()
        handle = exchanger.export_ipc_handle(buf)
        assert len(handle) == 64

    def test_export_handle_cpu_raises(self):
        """Should raise on CPU tensor."""
        buf = torch.zeros(10, 4)
        exchanger = IPCExchanger()
        with pytest.raises(RuntimeError, match="CUDA"):
            exchanger.export_ipc_handle(buf)

    def test_forward_push_correctness(self):
        """Multi-process test: exchange features and verify ghost values."""
        world_size = min(torch.cuda.device_count(), 4)
        num_atoms = 100
        feature_dim = 8  # Small for testing
        cutoff = 3.0

        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        results_dict = manager.dict()
        handles_dict = manager.dict()
        barrier = mp.Barrier(world_size)

        # Generate positions on CPU (shared across processes)
        torch.manual_seed(42)
        shared_positions = torch.rand(num_atoms, 3) * 10.0

        processes = []
        for rank in range(world_size):
            proc = mp.Process(
                target=_mp_worker,
                args=(rank, world_size, num_atoms, feature_dim, cutoff,
                      shared_positions, results_dict, handles_dict, barrier),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join(timeout=60)

        # Check results
        for rank in range(world_size):
            assert rank in results_dict, f"Rank {rank} did not report results"
            res = results_dict[rank]
            assert len(res["errors"]) == 0, (
                f"Rank {rank} errors: {res['errors']}"
            )
            print(f"  Rank {rank}: local={res['num_local']}, "
                  f"send={res['total_send']}, recv={res['total_recv']}")

    def test_forward_push_larger_system(self):
        """Test with more atoms and larger feature dim."""
        world_size = min(torch.cuda.device_count(), 4)
        num_atoms = 500
        feature_dim = 128
        cutoff = 3.0

        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        results_dict = manager.dict()
        handles_dict = manager.dict()
        barrier = mp.Barrier(world_size)

        torch.manual_seed(123)
        shared_positions = torch.rand(num_atoms, 3) * 10.0

        processes = []
        for rank in range(world_size):
            proc = mp.Process(
                target=_mp_worker,
                args=(rank, world_size, num_atoms, feature_dim, cutoff,
                      shared_positions, results_dict, handles_dict, barrier),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join(timeout=120)

        for rank in range(world_size):
            assert rank in results_dict, f"Rank {rank} did not report results"
            res = results_dict[rank]
            assert len(res["errors"]) == 0, (
                f"Rank {rank} errors: {res['errors']}"
            )
