"""Tests for SimpleSpatialPartitioner (C++) and build_halo_maps (C++/CUDA)."""

import torch
import pytest

from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
    SimulationConfig,
    SimpleSpatialPartitioner,
)


def _make_global_state(num_atoms: int, device: torch.device) -> MolecularInfo:
    """Create a MolecularInfo with random positions in a 10x10x10 box."""
    state = MolecularInfo()
    state.num_atoms = num_atoms
    state.atom_indices = torch.arange(num_atoms, dtype=torch.long, device=device)
    state.atom_positions = torch.rand(num_atoms, 3, device=device) * 10.0
    state.atom_velocities = torch.randn(num_atoms, 3, device=device) * 0.1
    return state


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


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for build_halo_maps"
)


class TestCppPartitioner:
    """Test the C++ SimpleSpatialPartitioner (works on CPU and CUDA)."""

    def test_rank0_gets_first_quarter(self, device):
        state = _make_global_state(100, device)
        env = _make_env(rank=0, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        assert info.partition_indices.shape == (25,)
        assert torch.equal(info.partition_indices, torch.arange(25, device=device))
        assert info.local_atom_positions.shape == (25, 3)

    def test_last_rank_gets_remainder(self, device):
        state = _make_global_state(103, device)
        env = _make_env(rank=3, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        expected_start = 3 * (103 // 4)
        expected_count = 103 - expected_start
        assert info.partition_indices.shape == (expected_count,)
        assert info.partition_indices[0].item() == expected_start
        assert info.partition_indices[-1].item() == 102

    def test_partition_id_correct(self, device):
        state = _make_global_state(100, device)
        env = _make_env(rank=0, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        assert info.partition_id.shape == (100,)
        for r in range(4):
            assert (info.partition_id == r).sum().item() == 25

    def test_all_ranks_cover_all_atoms(self, device):
        state = _make_global_state(100, device)
        p = SimpleSpatialPartitioner()

        all_indices = []
        for rank in range(4):
            env = _make_env(rank=rank, world_size=4)
            all_indices.append(p.partition(state, env).partition_indices)

        combined = torch.cat(all_indices)
        assert torch.equal(combined.sort()[0], torch.arange(100, device=device))

    def test_positions_match_global(self, device):
        state = _make_global_state(100, device)
        env = _make_env(rank=2, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        assert torch.equal(info.local_atom_positions, state.atom_positions[50:75])

    def test_diagonal_send_recv_matrix(self, device):
        state = _make_global_state(100, device)
        env = _make_env(rank=1, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        assert info.global_send_recv_matrix[1, 1].item() == 25

    def test_output_device(self, device):
        state = _make_global_state(100, device)
        env = _make_env(rank=0, world_size=4)
        info = SimpleSpatialPartitioner().partition(state, env)

        assert info.partition_id.device.type == device.type
        assert info.partition_indices.device.type == device.type
        assert info.local_atom_positions.device.type == device.type


@requires_cuda
class TestBuildHaloMaps:
    """Test C++ build_halo_maps (requires CUDA for radius_graph_pbc)."""

    def test_halo_fields_populated(self):
        device = torch.device("cuda")
        state = _make_global_state(40, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        assert info.send_offsets.shape == (5,)  # W+1
        assert info.recv_offsets.shape == (5,)
        assert info.send_offsets[0].item() == 0
        assert info.recv_offsets[0].item() == 0
        assert info.interior_mask.shape == (10,)  # 40/4
        assert info.interior_mask.dtype == torch.bool

    def test_self_rank_no_send(self):
        device = torch.device("cuda")
        state = _make_global_state(40, device)
        cfg = _make_sim_config(3.0, device)
        rank = 1
        env = _make_env(rank=rank, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        self_count = info.send_offsets[rank + 1] - info.send_offsets[rank]
        assert self_count.item() == 0

    def test_send_recv_consistency(self):
        """send[i,j] == recv[j,i] for all rank pairs."""
        device = torch.device("cuda")
        num_atoms = 60
        world_size = 3
        state = _make_global_state(num_atoms, device)
        cfg = _make_sim_config(3.0, device)

        all_send = torch.zeros(world_size, world_size, dtype=torch.long)
        all_recv = torch.zeros(world_size, world_size, dtype=torch.long)

        for rank in range(world_size):
            env = _make_env(rank=rank, world_size=world_size)
            p = SimpleSpatialPartitioner()
            info = p.partition(state, env)
            p.build_halo_maps(info, state, env, cfg)

            for r in range(world_size):
                all_send[rank, r] = info.send_offsets[r + 1] - info.send_offsets[r]
                all_recv[rank, r] = info.recv_offsets[r + 1] - info.recv_offsets[r]

        for i in range(world_size):
            for j in range(world_size):
                assert all_send[i, j].item() == all_recv[j, i].item(), (
                    f"send[{i},{j}]={all_send[i,j]} != recv[{j},{i}]={all_recv[j,i]}"
                )

    def test_interior_mask_shape(self):
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(4.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        assert info.interior_mask.dtype == torch.bool
        assert info.interior_mask.shape[0] == 25

    def test_output_device_cuda(self):
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        assert info.send_offsets.is_cuda
        assert info.send_local_indices.is_cuda
        assert info.recv_offsets.is_cuda
        assert info.recv_ghost_indices.is_cuda
        assert info.interior_mask.is_cuda

    def test_send_recv_matrix_updated(self):
        """build_halo_maps should update send_recv_matrix with halo counts."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        # Before: diagonal only
        assert info.global_send_recv_matrix[0, 0].item() == 25

        p.build_halo_maps(info, state, env, cfg)
        # After: diagonal still has local count, off-diagonal may have halo counts
        assert info.global_send_recv_matrix[0, 0].item() == 25
        # Total sent should be >= 0 (might be 0 if no atoms near boundaries)
        total_sent = info.global_send_recv_matrix[0].sum().item() - 25
        assert total_sent >= 0

    def test_send_local_indices_valid(self):
        """send_local_indices must be valid indices into local partition."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        num_local = info.local_atom_positions.shape[0]
        if info.send_local_indices.numel() > 0:
            assert info.send_local_indices.max().item() < num_local

    def test_recv_ghost_indices_offset(self):
        """recv_ghost_indices should start at num_local (ghosts appended after local)."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SimpleSpatialPartitioner()
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        num_local = info.local_atom_positions.shape[0]
        if info.recv_ghost_indices.numel() > 0:
            assert info.recv_ghost_indices.min().item() >= num_local
