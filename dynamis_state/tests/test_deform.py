"""deform_state / set_state_cell: positions and cell change together, in place."""

import pytest
import torch

from dynamis_state import (
    DynamisState,
    MolecularInfo,
    SimulationConfig,
    deform_state,
    set_state_cell,
)

CELL = [[10.0, 0.0, 0.0], [1.0, 11.0, 0.0], [0.5, -0.7, 12.0]]  # triclinic


def make_state(n=16, dtype=torch.float32):
    torch.manual_seed(0)
    mol = MolecularInfo()
    mol.num_atoms = n
    mol.atom_positions = torch.rand(n, 3, dtype=dtype) * 10.0
    mol.atom_velocities = torch.zeros(n, 3, dtype=dtype)
    mol.atom_types = torch.full((n,), 18, dtype=torch.long)
    sim = SimulationConfig()
    sim.cutoff = 3.5
    sim.ghost_cutoff_buffer = 0.5
    sim.cell = torch.tensor(CELL, dtype=dtype)
    sim.pbc = torch.tensor([True, True, True])
    state = DynamisState()
    state.local_molecular_info = mol
    state.sim_config = sim
    return state


def fractional(state):
    positions = state.local_molecular_info.atom_positions.double()
    return positions @ torch.linalg.inv(state.sim_config.cell.double())


def test_deform_maps_positions_and_cell_and_preserves_fractional_coordinates():
    state = make_state()
    x0 = state.local_molecular_info.atom_positions.clone()
    h0 = state.sim_config.cell.clone()
    frac0 = fractional(state)
    transform = torch.tensor(
        [[1.002, 0.001, 0.0], [0.0, 0.997, -0.002], [0.0005, 0.0, 1.001]]
    )
    deform_state(state, transform)
    torch.testing.assert_close(
        state.local_molecular_info.atom_positions, x0 @ transform, rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        state.sim_config.cell, h0 @ transform, rtol=0, atol=1e-5
    )
    torch.testing.assert_close(fractional(state), frac0, rtol=0, atol=1e-5)


def test_identity_is_a_bitwise_noop():
    state = make_state()
    x0 = state.local_molecular_info.atom_positions.clone()
    h0 = state.sim_config.cell.clone()
    deform_state(state, torch.eye(3))
    assert torch.equal(state.local_molecular_info.atom_positions, x0)
    assert torch.equal(state.sim_config.cell, h0)


def test_deform_is_in_place_so_existing_handles_see_it():
    state = make_state()
    cell_handle = state.sim_config.cell
    pos_handle = state.local_molecular_info.atom_positions
    cell_ptr, pos_ptr = cell_handle.data_ptr(), pos_handle.data_ptr()
    deform_state(state, torch.eye(3) * 1.01)
    assert state.sim_config.cell.data_ptr() == cell_ptr
    assert state.local_molecular_info.atom_positions.data_ptr() == pos_ptr
    assert torch.equal(cell_handle, state.sim_config.cell)
    assert torch.equal(pos_handle, state.local_molecular_info.atom_positions)
    assert cell_handle[0, 0].item() == pytest.approx(10.1, rel=1e-6)


def test_deform_accepts_a_float64_transform_and_leaves_no_graph():
    state = make_state()
    deform_state(state, (torch.eye(3, dtype=torch.float64) * 0.99).requires_grad_(True))
    assert not state.sim_config.cell.requires_grad
    assert not state.local_molecular_info.atom_positions.requires_grad
    assert state.sim_config.cell.dtype == torch.float32


def test_set_state_cell_replaces_the_cell_in_place_and_moves_no_atoms():
    state = make_state()
    x0 = state.local_molecular_info.atom_positions.clone()
    handle = state.sim_config.cell
    ptr = handle.data_ptr()
    new = torch.tensor([[9.0, 0, 0], [0, 9.5, 0], [0, 0, 10.0]])
    set_state_cell(state, new)
    assert state.sim_config.cell.data_ptr() == ptr
    assert torch.equal(handle, new)
    assert torch.equal(state.local_molecular_info.atom_positions, x0)


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((3, 3), float("nan")),
        torch.tensor([[10.0, 0, 0], [0, 11.0, 0], [0, 0, -12.0]]),  # left-handed
        torch.zeros(3, 3),
        torch.eye(2),
    ],
)
def test_set_state_cell_rejects_malformed_cells_and_keeps_the_old_one(bad):
    state = make_state()
    h0 = state.sim_config.cell.clone()
    with pytest.raises(ValueError):
        set_state_cell(state, bad)
    assert torch.equal(state.sim_config.cell, h0)
