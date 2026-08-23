"""Tests for SFCPartitioner (C++) and build_halo_maps integration."""

import torch
import pytest

from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
    SimulationConfig,
    SFCPartitioner,
    SFCCurveType,
)


def _make_global_state(num_atoms: int, device: torch.device) -> MolecularInfo:
    """Create a MolecularInfo with random positions in a 10x10x10 box."""
    state = MolecularInfo()
    state.num_atoms = num_atoms
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long, device=device)
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


@pytest.fixture(params=[SFCCurveType.Morton, SFCCurveType.Hilbert])
def curve_type(request):
    return request.param


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


class TestSFCPartitioner:
    """Test the C++ SFCPartitioner (works on CPU and CUDA)."""

    def test_partition_id_range(self, device, curve_type):
        """Partition IDs must be in [0, W)."""
        state = _make_global_state(100, device)
        env = _make_env(rank=0, world_size=4)
        info = SFCPartitioner(curve_type).partition(state, env)

        assert info.partition_id.shape == (100,)
        assert info.partition_id.min().item() >= 0
        assert info.partition_id.max().item() < 4

    def test_all_ranks_cover_all_atoms(self, device, curve_type):
        """Union of partition_indices across all ranks = arange(N)."""
        state = _make_global_state(100, device)
        p = SFCPartitioner(curve_type)

        all_indices = []
        for rank in range(4):
            env = _make_env(rank=rank, world_size=4)
            all_indices.append(p.partition(state, env).partition_indices)

        combined = torch.cat(all_indices)
        assert combined.shape == (100,)
        assert torch.equal(combined.sort()[0], torch.arange(100, device=device))

    def test_balance_unweighted(self, device, curve_type):
        """Unweighted partitions should be roughly balanced (within 20%)."""
        N = 200
        W = 4
        state = _make_global_state(N, device)
        env = _make_env(rank=0, world_size=W)
        info = SFCPartitioner(curve_type).partition(state, env)

        expected = N / W
        for r in range(W):
            count = (info.partition_id == r).sum().item()
            assert abs(count - expected) / expected < 0.20, (
                f"Rank {r}: {count} atoms, expected ~{expected}"
            )

    def test_weighted_balances_edge_cost(self, device, curve_type):
        """With skewed edge_counts, weighted mode should balance total cost per partition."""
        N = 200
        W = 2
        state = _make_global_state(N, device)
        weights = torch.ones(N, dtype=torch.long, device=device)
        weights[:N // 2] = 10
        state.edge_counts = weights

        env = _make_env(rank=0, world_size=W)
        info = SFCPartitioner(curve_type).partition(state, env)

        total = weights.float().sum().item()
        for r in range(W):
            mask = info.partition_id == r
            cost = weights[mask].float().sum().item()
            ratio = cost / total
            assert 0.20 < ratio < 0.80, (
                f"Rank {r}: cost ratio {ratio:.2f}, expected ~0.50"
            )

    def test_unweighted_default(self, device, curve_type):
        """No edge_counts set → runs without error (unweighted mode)."""
        state = _make_global_state(50, device)
        env = _make_env(rank=0, world_size=2)
        info = SFCPartitioner(curve_type).partition(state, env)
        assert info.partition_id.shape == (50,)

    def test_spatial_locality(self, device, curve_type):
        """Two separated clusters should end up in different partitions."""
        N = 100
        pos = torch.zeros(N, 3, device=device)
        pos[:50] = torch.rand(50, 3, device=device) * 1.0
        pos[50:] = torch.rand(50, 3, device=device) * 1.0 + 8.0

        state = MolecularInfo()
        state.num_atoms = N
        state.atom_vertex_numbers = torch.arange(N, dtype=torch.long, device=device)
        state.atom_positions = pos
        state.atom_velocities = torch.zeros(N, 3, device=device)

        env = _make_env(rank=0, world_size=2)
        info = SFCPartitioner(curve_type).partition(state, env)

        labels_a = info.partition_id[:50]
        labels_b = info.partition_id[50:]
        majority_a = labels_a.mode().values.item()
        majority_b = labels_b.mode().values.item()
        assert majority_a != majority_b, "Two separated clusters should get different partitions"
        assert (labels_a == majority_a).sum().item() >= 40
        assert (labels_b == majority_b).sum().item() >= 40

    def test_diagonal_send_recv_matrix(self, device, curve_type):
        """matrix[R,R] should equal num_local atoms for rank R."""
        state = _make_global_state(100, device)
        env = _make_env(rank=1, world_size=4)
        info = SFCPartitioner(curve_type).partition(state, env)

        num_local = info.partition_indices.shape[0]
        assert info.global_send_recv_matrix[1, 1].item() == num_local

    def test_non_power_of_two(self, device, curve_type):
        """W=3 (non-power-of-two) should work correctly."""
        N = 99
        W = 3
        state = _make_global_state(N, device)
        p = SFCPartitioner(curve_type)

        all_indices = []
        for rank in range(W):
            env = _make_env(rank=rank, world_size=W)
            all_indices.append(p.partition(state, env).partition_indices)

        combined = torch.cat(all_indices)
        assert torch.equal(combined.sort()[0], torch.arange(N, device=device))

    def test_output_device(self, device, curve_type):
        """Output tensors should be on the same device as input."""
        state = _make_global_state(100, device)
        env = _make_env(rank=0, world_size=4)
        info = SFCPartitioner(curve_type).partition(state, env)

        assert info.partition_id.device.type == device.type
        assert info.partition_indices.device.type == device.type

    def test_single_atom(self, device, curve_type):
        """N=1: single atom assigned to exactly one valid rank."""
        N, W = 1, 2
        state = _make_global_state(N, device)
        p = SFCPartitioner(curve_type)

        all_indices = []
        for rank in range(W):
            env = _make_env(rank=rank, world_size=W)
            info = p.partition(state, env)
            all_indices.append(info.partition_indices)

        assert info.partition_id.shape == (N,)
        assert 0 <= info.partition_id[0].item() < W
        combined = torch.cat(all_indices)
        assert torch.equal(combined.sort()[0], torch.arange(N, device=device))

    def test_single_worker(self, device, curve_type):
        """W=1: all atoms go to rank 0."""
        N = 50
        state = _make_global_state(N, device)
        env = _make_env(rank=0, world_size=1)
        info = SFCPartitioner(curve_type).partition(state, env)

        assert (info.partition_id == 0).all()
        assert info.partition_indices.shape == (N,)
        assert torch.equal(info.partition_indices.sort()[0], torch.arange(N, device=device))

    def test_more_workers_than_atoms(self, device, curve_type):
        """W > N: some partitions must be empty, no crash."""
        N = 3
        W = 8
        state = _make_global_state(N, device)
        p = SFCPartitioner(curve_type)

        all_indices = []
        for rank in range(W):
            env = _make_env(rank=rank, world_size=W)
            info = p.partition(state, env)
            all_indices.append(info.partition_indices)

        combined = torch.cat(all_indices)
        assert torch.equal(combined.sort()[0], torch.arange(N, device=device))

        empty_ranks = sum(1 for t in all_indices if t.numel() == 0)
        assert empty_ranks >= W - N


@requires_cuda
class TestSFCCUDAProperties:
    """Test CUDA-specific properties: determinism, CPU/CUDA consistency."""

    @pytest.fixture(params=[SFCCurveType.Morton, SFCCurveType.Hilbert])
    def curve_type(self, request):
        return request.value if hasattr(request, 'value') else request.param

    def test_determinism(self, curve_type):
        """Same input must produce identical partition_id across 10 runs."""
        device = torch.device("cuda")
        N, W = 500, 8
        state = _make_global_state(N, device)
        state.edge_counts = torch.randint(1, 50, (N,), dtype=torch.long, device=device)
        env = _make_env(rank=0, world_size=W)
        p = SFCPartitioner(curve_type)

        ref = p.partition(state, env).partition_id.clone()
        for _ in range(9):
            result = p.partition(state, env).partition_id
            assert torch.equal(result, ref), "CUDA partition must be deterministic"

    def test_cpu_cuda_balance_consistency(self, curve_type):
        """CPU and CUDA should produce partitions of similar quality."""
        N, W = 200, 4
        cpu_state = _make_global_state(N, torch.device("cpu"))

        cuda_state = MolecularInfo()
        cuda_state.num_atoms = N
        cuda_state.atom_vertex_numbers = cpu_state.atom_vertex_numbers.cuda()
        cuda_state.atom_positions = cpu_state.atom_positions.cuda()
        cuda_state.atom_velocities = cpu_state.atom_velocities.cuda()

        p_cpu = SFCPartitioner(curve_type)
        p_cuda = SFCPartitioner(curve_type)
        env = _make_env(rank=0, world_size=W)

        cpu_labels = p_cpu.partition(cpu_state, env).partition_id
        cuda_labels = p_cuda.partition(cuda_state, env).partition_id.cpu()

        assert cpu_labels.unique().numel() == W
        assert cuda_labels.unique().numel() == W

        expected = N / W
        for r in range(W):
            cpu_count = (cpu_labels == r).sum().item()
            cuda_count = (cuda_labels == r).sum().item()
            assert abs(cpu_count - expected) / expected < 0.25, (
                f"CPU rank {r}: {cpu_count} atoms, expected ~{expected}"
            )
            assert abs(cuda_count - expected) / expected < 0.25, (
                f"CUDA rank {r}: {cuda_count} atoms, expected ~{expected}"
            )

    def test_weighted_cpu_cuda_balance_consistency(self, curve_type):
        """Weighted CPU and CUDA paths produce similarly balanced partitions."""
        N, W = 200, 4
        cpu_state = _make_global_state(N, torch.device("cpu"))
        weights = torch.randint(1, 50, (N,), dtype=torch.long)
        cpu_state.edge_counts = weights

        cuda_state = MolecularInfo()
        cuda_state.num_atoms = N
        cuda_state.atom_vertex_numbers = cpu_state.atom_vertex_numbers.cuda()
        cuda_state.atom_positions = cpu_state.atom_positions.cuda()
        cuda_state.atom_velocities = cpu_state.atom_velocities.cuda()
        cuda_state.edge_counts = weights.cuda()

        p = SFCPartitioner(curve_type)
        env = _make_env(rank=0, world_size=W)

        cpu_labels = p.partition(cpu_state, env).partition_id
        cuda_labels = p.partition(cuda_state, env).partition_id.cpu()

        assert cpu_labels.min().item() >= 0 and cpu_labels.max().item() < W
        assert cuda_labels.min().item() >= 0 and cuda_labels.max().item() < W

        total_cost = weights.float().sum().item()
        for path_name, labels in [("CPU", cpu_labels), ("CUDA", cuda_labels)]:
            for r in range(W):
                cost = weights[labels == r].float().sum().item()
                ratio = cost / total_cost
                assert 0.10 < ratio < 0.45, (
                    f"{path_name} rank {r}: cost ratio {ratio:.2f}, expected ~0.25"
                )


@requires_cuda
class TestSFCAutoWeighted:
    """Test mode 3: auto-compute edge weights via radius_graph_pbc (CUDA)."""

    @pytest.fixture(params=[SFCCurveType.Morton, SFCCurveType.Hilbert])
    def curve_type(self, request):
        return request.param

    def test_auto_weighted_partition(self, curve_type):
        """partition(state, env, config) should auto-compute weights and partition."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        info = SFCPartitioner(curve_type).partition(state, env, cfg)

        assert info.partition_id.shape == (100,)
        assert info.partition_id.min().item() >= 0
        assert info.partition_id.max().item() < 4

    def test_auto_weighted_edge_index_populated(self, curve_type):
        """Mode 3 should populate edge_index [2, E] at cutoff radius."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        info = SFCPartitioner(curve_type).partition(state, env, cfg)

        assert info.edge_index is not None
        assert info.edge_index.dim() == 2
        assert info.edge_index.shape[0] == 2
        assert info.edge_index.shape[1] > 0

    def test_auto_weighted_covers_all_atoms(self, curve_type):
        """Union of partition_indices across all ranks = arange(N)."""
        device = torch.device("cuda")
        N = 100
        W = 4
        state = _make_global_state(N, device)
        cfg = _make_sim_config(3.0, device)
        p = SFCPartitioner(curve_type)

        all_indices = []
        for rank in range(W):
            env = _make_env(rank=rank, world_size=W)
            all_indices.append(p.partition(state, env, cfg).partition_indices)

        combined = torch.cat(all_indices)
        assert torch.equal(combined.sort()[0], torch.arange(N, device=device))

    def test_unweighted_no_edge_index(self, curve_type):
        """Mode 1 (unweighted, no config) should NOT populate edge_index."""
        device = torch.device("cuda")
        state = _make_global_state(50, device)
        env = _make_env(rank=0, world_size=2)

        info = SFCPartitioner(curve_type).partition(state, env)
        assert info.edge_index is None


@requires_cuda
class TestSFCHaloMaps:
    """Test build_halo_maps with SFCPartitioner (requires CUDA)."""

    @pytest.fixture(params=[SFCCurveType.Morton, SFCCurveType.Hilbert])
    def curve_type(self, request):
        return request.param

    def test_halo_fields_populated(self, curve_type):
        device = torch.device("cuda")
        state = _make_global_state(40, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SFCPartitioner(curve_type)
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        W = 4
        assert info.send_offsets.shape == (W + 1,)
        assert info.recv_offsets.shape == (W + 1,)
        assert info.send_offsets[0].item() == 0
        assert info.recv_offsets[0].item() == 0
        num_local = info.partition_indices.shape[0]
        assert info.interior_mask.shape == (num_local,)
        assert info.interior_mask.dtype == torch.bool

    def test_send_recv_consistency(self, curve_type):
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
            p = SFCPartitioner(curve_type)
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

    def test_send_local_indices_valid(self, curve_type):
        """send_local_indices must be valid indices into local partition."""
        device = torch.device("cuda")
        state = _make_global_state(100, device)
        cfg = _make_sim_config(3.0, device)
        env = _make_env(rank=0, world_size=4)

        p = SFCPartitioner(curve_type)
        info = p.partition(state, env)
        p.build_halo_maps(info, state, env, cfg)

        num_local = info.partition_indices.shape[0]
        if info.send_local_indices.numel() > 0:
            assert info.send_local_indices.max().item() < num_local
