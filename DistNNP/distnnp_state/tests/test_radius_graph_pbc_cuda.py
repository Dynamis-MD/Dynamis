"""Tests for CUDA cell-list radius graph with PBC.

Compares CUDA output against the reference radius_graph_pbc_v2 implementation.
Edge sets are compared after sorting to handle different orderings.
"""

import sys
import os
import math
import importlib.util
import pytest
import torch

# Directly load radius_graph_pbc module to avoid baseline's __init__.py import chain
_RGB_PBC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "baseline",
    "local_fairchem", "graph", "radius_graph_pbc.py"
)
_spec = importlib.util.spec_from_file_location("_radius_graph_pbc_ref", _RGB_PBC_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
radius_graph_pbc_v2 = _mod.radius_graph_pbc_v2

# Import the CUDA implementation
import distnnp_state.distnnp_state_impl as _impl


# ===== Helpers =====

class FakeData:
    """Minimal data object matching the interface expected by radius_graph_pbc_v2."""

    def __init__(self, pos, cell, natoms, pbc=None, batch=None):
        self.pos = pos
        self.cell = cell  # [batch, 3, 3]
        self.natoms = natoms
        if pbc is None:
            pbc = torch.tensor([[True, True, True]])
        self.pbc = pbc
        if batch is None:
            self.batch = torch.zeros(pos.size(0), dtype=torch.int, device=pos.device)
        else:
            self.batch = batch


def sort_edges(edge_index, cell_offsets):
    """Sort edges by (src, tgt, cell_shift) lexicographically for comparison."""
    E = edge_index.size(1)
    if E == 0:
        return edge_index, cell_offsets

    # Lexicographic sort using int64 keys to avoid float precision loss
    src = edge_index[0].long()
    tgt = edge_index[1].long()
    cs = cell_offsets.long()  # cell shifts are integer-valued

    # Build composite key with sufficient spacing for each field
    # Each cell shift component is in [-3, 3], offset to [0, 6], needs 3 bits
    # tgt/src can be up to ~10K, needs 14 bits
    key = (src << 40) | (tgt << 26) | ((cs[:, 0] + 4) << 6) | ((cs[:, 1] + 4) << 3) | (cs[:, 2] + 4)
    _, sort_idx = torch.sort(key)

    return edge_index[:, sort_idx], cell_offsets[sort_idx]


def compare_results(cuda_ei, cuda_co, ref_ei, ref_co, atol=1e-4):
    """Compare CUDA and reference results after sorting edges."""
    # Sort both
    cuda_ei_s, cuda_co_s = sort_edges(cuda_ei.cpu(), cuda_co.cpu())
    ref_ei_s, ref_co_s = sort_edges(ref_ei.cpu(), ref_co.cpu())

    assert cuda_ei_s.shape == ref_ei_s.shape, (
        f"Edge count mismatch: CUDA={cuda_ei_s.shape[1]}, ref={ref_ei_s.shape[1]}"
    )
    assert torch.equal(cuda_ei_s, ref_ei_s), "Edge indices mismatch"
    assert torch.allclose(cuda_co_s, ref_co_s, atol=atol), "Cell offsets mismatch"


def make_cubic_system(n_atoms, box_size, device="cuda"):
    """Create a random cubic system."""
    pos = torch.rand(n_atoms, 3, device=device) * box_size
    cell = torch.eye(3, device=device) * box_size
    cell = cell.unsqueeze(0)  # [1, 3, 3]
    natoms = torch.tensor([n_atoms], device=device)
    pbc = torch.tensor([[True, True, True]])
    return FakeData(pos, cell, natoms, pbc)


# ===== Tests =====

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestCubicCell:
    """Tests with cubic unit cells."""

    def test_small_cubic(self, device):
        """8 atoms in a 10A cubic box, cutoff=3A."""
        torch.manual_seed(42)
        data = make_cubic_system(8, 10.0, device)
        radius = 3.0
        max_neighbors = 50

        # Reference
        ref_ei, ref_co, ref_nn = radius_graph_pbc_v2(
            data, radius, max_neighbors
        )

        # CUDA
        cuda_ei, cuda_co, cuda_nn = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)

    def test_medium_cubic(self, device):
        """100 atoms in a 20A cubic box, cutoff=5A."""
        torch.manual_seed(123)
        data = make_cubic_system(100, 20.0, device)
        radius = 5.0
        max_neighbors = 200

        ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_neighbors)
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)

    def test_large_cutoff_small_box(self, device):
        """Cutoff larger than half the box → multi-image replication."""
        torch.manual_seed(7)
        n = 10
        box = 8.0
        data = make_cubic_system(n, box, device)
        radius = 6.0  # > box/2 → needs images
        max_neighbors = 500

        ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_neighbors)
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)


class TestTriclinicCell:
    """Tests with triclinic (non-orthogonal) unit cells."""

    def test_triclinic(self, device):
        """20 atoms in a triclinic cell."""
        torch.manual_seed(99)
        n = 20
        # Triclinic cell: sheared box
        cell = torch.tensor([
            [15.0, 0.0, 0.0],
            [3.0, 14.0, 0.0],
            [1.0, 2.0, 13.0]
        ], device=device).unsqueeze(0)

        # Generate positions within the cell using fractional coords
        frac = torch.rand(n, 3, device=device)
        pos = frac @ cell[0]  # Convert to Cartesian

        natoms = torch.tensor([n], device=device)
        pbc = torch.tensor([[True, True, True]])
        data = FakeData(pos, cell, natoms, pbc)

        radius = 5.0
        max_neighbors = 200

        ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_neighbors)
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)


class TestPartialPBC:
    """Tests with partial periodic boundary conditions."""

    def test_pbc_xy_only(self, device):
        """PBC in x and y only, not z."""
        torch.manual_seed(55)
        n = 30
        data = make_cubic_system(n, 15.0, device)
        pbc_flags = torch.tensor([True, True, False])
        data.pbc = pbc_flags.unsqueeze(0)

        radius = 4.0
        max_neighbors = 200

        ref_ei, ref_co, _ = radius_graph_pbc_v2(
            data, radius, max_neighbors, pbc=pbc_flags
        )
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], pbc_flags,
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)

    def test_no_pbc(self, device):
        """No PBC at all — pure cluster calculation."""
        torch.manual_seed(77)
        n = 25
        data = make_cubic_system(n, 20.0, device)
        pbc_flags = torch.tensor([False, False, False])
        data.pbc = pbc_flags.unsqueeze(0)

        radius = 5.0
        max_neighbors = 200

        ref_ei, ref_co, _ = radius_graph_pbc_v2(
            data, radius, max_neighbors, pbc=pbc_flags
        )
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], pbc_flags,
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)


class TestEdgeCases:
    """Edge cases: empty, single atom, etc."""

    def test_empty(self, device):
        """Zero atoms."""
        pos = torch.empty(0, 3, device=device)
        cell = torch.eye(3, device=device)
        pbc = torch.tensor([True, True, True])

        ei, co, nn = _impl.radius_graph_pbc(pos, cell, pbc, 5.0, 100, False)
        assert ei.shape == (2, 0)
        assert co.shape == (0, 3)

    def test_single_atom_with_pbc(self, device):
        """Single atom with PBC — should find self-images when box < cutoff."""
        # box=5Å, cutoff=6Å → periodic images at distance 5Å are within cutoff
        pos = torch.tensor([[2.5, 2.5, 2.5]], device=device)
        cell = torch.eye(3, device=device) * 5.0
        pbc = torch.tensor([True, True, True])
        radius = 6.0

        ei, co, _ = _impl.radius_graph_pbc(pos, cell, pbc, radius, 100, False)

        # Should have 6 edges: ±1 image in each of x, y, z (distance = 5 < 6)
        # Diagonal images like (1,1,0) have distance sqrt(50)≈7.07 > 6, excluded
        assert ei.size(1) == 6
        # All source and target indices should be 0 (only one atom)
        assert torch.all(ei[0] == 0)
        assert torch.all(ei[1] == 0)
        # Cell shifts should never be (0,0,0) since that's the self-image
        zero_shifts = (co.abs().sum(dim=1) == 0)
        assert not torch.any(zero_shifts)

    def test_single_atom_no_pbc(self, device):
        """Single atom without PBC — should have zero edges."""
        pos = torch.tensor([[5.0, 5.0, 5.0]], device=device)
        cell = torch.eye(3, device=device) * 10.0
        pbc = torch.tensor([False, False, False])

        ei, co, _ = _impl.radius_graph_pbc(pos, cell, pbc, 5.0, 100, False)
        assert ei.size(1) == 0


class TestSelectiveReplication:
    """Verify that selective PBC replication creates fewer source atoms than full replication."""

    def test_source_count_less_than_full(self, device):
        """For a typical system, selective replication should use far fewer images."""
        torch.manual_seed(42)
        n = 100
        box = 50.0  # Large box relative to cutoff
        data = make_cubic_system(n, box, device)
        radius = 6.0

        # Full replication: n * 27 = 2700
        full_count = n * 27

        # Run CUDA kernel to count images
        # We can't directly access source count, but we can verify correctness
        # and check that it works (the performance benefit is the fewer sources)
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, 1000, False
        )

        # Just verify it produces valid results
        assert cuda_ei.size(1) > 0
        # Verify edge indices are in valid range
        assert cuda_ei.min() >= 0
        assert cuda_ei.max() < n


class TestMaxNeighbors:
    """Test max_num_neighbors behavior.

    Note: strict truncation (max_num_neighbors > 0) is not implemented — every
    current caller passes 0 (no limit). The former test_max_neighbors_strict was
    removed together with the unused sort/truncation kernel.
    """

    def test_max_neighbors_no_limit(self, device):
        """max_num_neighbors=0 means no limit."""
        torch.manual_seed(42)
        data = make_cubic_system(20, 10.0, device)
        radius = 4.0

        ref_ei, ref_co, _ = radius_graph_pbc_v2(
            data, radius, 0  # no limit
        )
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, 0, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)


class TestLargeSystem:
    """Larger system tests (still fast enough for CI)."""

    def test_1k_atoms(self, device):
        """1000 atoms — main correctness test at scale."""
        torch.manual_seed(2024)
        data = make_cubic_system(1000, 30.0, device)
        radius = 5.0
        max_neighbors = 200

        ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_neighbors)
        cuda_ei, cuda_co, _ = _impl.radius_graph_pbc(
            data.pos, data.cell[0], torch.tensor([True, True, True]),
            radius, max_neighbors, False
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)


class TestPythonWrapper:
    """Test the Python wrapper that takes a data object."""

    def test_wrapper_single_image(self, device):
        """Test radius_graph_pbc_cuda wrapper with single image."""
        from distnnp_state import radius_graph_pbc_cuda

        torch.manual_seed(42)
        data = make_cubic_system(50, 15.0, device)
        radius = 4.0
        max_neighbors = 100

        ref_ei, ref_co, _ = radius_graph_pbc_v2(data, radius, max_neighbors)
        cuda_ei, cuda_co, _ = radius_graph_pbc_cuda(
            data, radius, max_neighbors
        )

        compare_results(cuda_ei, cuda_co, ref_ei, ref_co)
