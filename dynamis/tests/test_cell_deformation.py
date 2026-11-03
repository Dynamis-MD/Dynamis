"""
test_cell_deformation.py — the drift watchdog under a changing cell.

    torchrun --nproc_per_node=2 -m pytest dynamis/tests/test_cell_deformation.py -v

The watchdog's job is to keep the halo maps complete between rebuilds. These tests
check that property directly, against a brute-force reference: every edge within the
cutoff at the *current* positions and cell must have a source the destination's rank
either owns or received in the build-time halo. They cover the three ways a
position-only test (|x - x_ref| <= skin / 2) is wrong once the cell can change:

  * it cannot see a cell change that leaves the atoms in place (periodic neighbours
    move closer by the change of the lattice vector);
  * it trips needlessly on a pure affine deformation, whose atoms move a long way in
    absolute terms but keep their neighbours;
  * it knows nothing about the box update an integrator may still apply.
"""

import pytest
import torch
import torch.distributed as dist

from dynamis.simulatorState import DRIFT_BREACH, DRIFT_TRIP, SimState
from dynamis_state import (
    MolecularInfo,
    SFCCurveType,
    SFCPartitioner,
    SimulationConfig,
    radius_graph_pbc_cuda,
)

from .conftest import _make_env

NUM_ATOMS = 216
BOX = 15.0
CUTOFF = 3.5
SKIN = 0.5
THRESHOLD = SKIN / 2


@pytest.fixture
def state():
    import os

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    gen = torch.Generator().manual_seed(7)  # the same universe on every rank
    mol = MolecularInfo()
    mol.num_atoms = NUM_ATOMS
    mol.atom_vertex_numbers = torch.arange(NUM_ATOMS).to(device)
    mol.atom_positions = (torch.rand(NUM_ATOMS, 3, generator=gen) * BOX).to(device)
    mol.atom_velocities = torch.zeros(NUM_ATOMS, 3, device=device)
    mol.atom_types = torch.full((NUM_ATOMS,), 18, dtype=torch.long, device=device)
    sim = SimulationConfig()
    sim.cutoff = CUTOFF
    sim.ghost_cutoff_buffer = SKIN
    sim.cell = torch.eye(3) * BOX
    sim.pbc = torch.tensor([True, True, True])
    env = _make_env(dist.get_rank(), dist.get_world_size(), local_rank)
    return SimState(mol, SFCPartitioner(SFCCurveType.Morton), sim, env, device)


# ---------------------------------------------------------------------------
# helpers: the brute-force halo completeness reference
# ---------------------------------------------------------------------------


def _ghost_sets(st):
    """Global ids each rank receives, from the current halo maps."""
    info = st.local_partition_info
    ids = info.partition_indices
    offsets = info.send_offsets.tolist()
    local = info.send_local_indices
    sent = [
        ids[local[offsets[r] : offsets[r + 1]]].cpu()
        for r in range(dist.get_world_size())
    ]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, sent)
    ghosts = [set() for _ in range(dist.get_world_size())]
    for per_rank in gathered:
        for receiver, ids_sent in enumerate(per_rank):
            ghosts[receiver].update(ids_sent.tolist())
    return ghosts


def _global_positions(st):
    info = st.local_partition_info
    mine = (
        info.partition_indices.cpu(),
        st._state.local_molecular_info.atom_positions.detach().cpu(),
    )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    full = torch.zeros(NUM_ATOMS, 3)
    for ids, pos in gathered:
        full[ids] = pos
    return full


def _missing_edges(st, ghosts):
    """Edges within the cutoff now whose source the destination's rank never gets."""
    cell = st.sim_config.cell.reshape(1, 3, 3)
    data = type(
        "Data",
        (object,),
        {
            "pos": _global_positions(st).to(cell.device),
            "cell": cell,
            "natoms": [NUM_ATOMS],
            "pbc": st.sim_config.pbc.reshape(1, 3),
        },
    )()
    edge_index, _, _ = radius_graph_pbc_cuda(
        data=data,
        radius=CUTOFF,
        max_num_neighbors_threshold=0,
        enforce_max_neighbors_strictly=False,
    )
    owner = st.local_partition_info.partition_id.cpu()
    src, dst = edge_index.cpu()
    receiver = owner[dst]
    covered = (owner[src] == receiver) | torch.tensor(
        [int(s) in ghosts[int(r)] for s, r in zip(src.tolist(), receiver.tolist())],
        dtype=torch.bool,
    )
    return int((~covered).sum())


def _word(st, threshold=THRESHOLD, dt=0.0):
    """Bitwise OR over ranks of the drift word, as the watchdog consumes it."""
    flag = torch.zeros(1, dtype=torch.int32, device=st.device)
    st.write_local_drift_flag(flag, threshold, dt)
    words = [torch.zeros_like(flag) for _ in range(dist.get_world_size())]
    dist.all_gather(words, flag)
    out = 0
    for w in words:
        out |= int(w.item())
    return out


class _Restore:
    """Put positions and cell back between trials (in place)."""

    def __init__(self, st):
        self.st = st
        self.x = st._state.local_molecular_info.atom_positions.clone()
        self.h = st.sim_config.cell.clone()

    def __call__(self):
        self.st._state.local_molecular_info.atom_positions.copy_(self.x)
        self.st.sim_config.cell.copy_(self.h)


def _plain_displacement(st):
    cur = st._state.local_molecular_info.atom_positions
    d = (cur - st._ref_positions).norm(dim=1).max().reshape(1).clone()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return float(d.item())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_fixed_cell_behaviour_is_unchanged(state):
    restore = _Restore(state)
    assert _word(state) == 0
    pos = state._state.local_molecular_info.atom_positions
    pos[:, 0] += 0.1  # inside the budget
    assert _word(state) == 0
    restore()
    pos[:, 0] += 0.3  # past threshold = 0.25
    assert _word(state) == DRIFT_TRIP | DRIFT_BREACH


def test_no_breach_implies_a_complete_halo_under_random_deformation(state):
    ghosts = _ghost_sets(state)
    assert _missing_edges(state, ghosts) == 0, "the build-time halo is incomplete"
    restore = _Restore(state)
    gen = torch.Generator().manual_seed(11)  # same stream on every rank
    noise_gen = torch.Generator().manual_seed(100 + dist.get_rank())
    inside = outside = 0
    for trial in range(60):
        restore()
        scale = 10 ** (-4.0 + 2.5 * torch.rand(1, generator=gen).item())  # 1e-4..3e-2
        amp = 0.35 * torch.rand(1, generator=gen).item()
        transform = torch.eye(3) + scale * torch.randn(3, 3, generator=gen)
        state.deform(transform.to(state.device))
        pos = state._state.local_molecular_info.atom_positions
        direction = torch.randn(pos.shape, generator=noise_gen)
        direction /= direction.norm(dim=1, keepdim=True)
        radius = amp * torch.rand(pos.shape[0], 1, generator=noise_gen)
        pos += (direction * radius).to(pos.device)

        if _word(state) & DRIFT_BREACH:
            outside += 1
            continue
        inside += 1
        assert (
            _missing_edges(state, ghosts) == 0
        ), f"trial {trial}: no breach, but an edge lost its source (scale={scale:.2e}, amp={amp:.2f})"
    assert inside >= 10 and outside >= 10, (inside, outside)


def test_a_cell_shrink_that_leaves_atoms_in_place_is_caught(state):
    """The case a position-only watchdog is blind to."""
    ghosts = _ghost_sets(state)
    h = state.sim_config.cell.clone()
    state.set_cell(h * 0.96)  # 0.6 A at BOX = 15, atoms untouched

    assert _plain_displacement(state) == 0.0  # |x - x_ref| sees nothing
    assert _missing_edges(state, ghosts) > 0  # ...but edges have lost their source
    assert _word(state) & DRIFT_BREACH  # the affine-aware watchdog does see it


def test_a_large_affine_contraction_is_caught_by_the_stretch_allowance(state):
    """Every atom moves exactly with the lattice (no beyond-affine displacement), yet
    pairs 3.5 / 0.85 = 4.1 A apart end up inside the cutoff. Only the delta * halo / 2
    allowance in the budget sees this."""
    ghosts = _ghost_sets(state)
    state.deform((torch.eye(3) * 0.85).to(state.device))

    assert _missing_edges(state, ghosts) > 0
    assert _word(state) & DRIFT_BREACH


def test_a_pure_affine_deformation_does_not_trip_needlessly(state):
    """Atoms move far in absolute terms but keep their neighbours."""
    ghosts = _ghost_sets(state)
    state.deform((torch.eye(3) * 1.015).to(state.device))  # 15 A box: up to ~0.39 A

    assert _plain_displacement(state) > THRESHOLD  # a position-only test would trip
    assert _word(state) == 0
    assert _missing_edges(state, ghosts) == 0


def test_rebuild_after_a_trip_restores_a_complete_halo_and_a_quiet_watchdog(state):
    state.set_cell(state.sim_config.cell.clone() * 0.96)
    assert _word(state) & DRIFT_BREACH

    state.sync_positions_and_rebuild_halo()  # collective: all ranks trip together
    assert _missing_edges(state, _ghost_sets(state)) == 0
    assert _word(state) == 0  # the reference cell was re-latched


def test_cell_change_look_ahead_trips_without_a_breach(state):
    assert _word(state) == 0
    state.cell_change_bound_per_step = 0.2  # 0.5 * (3.5 + 0.5) * 0.2 = 0.4 A of reach
    assert _word(state) == DRIFT_TRIP
    state.cell_change_bound_per_step = 0.0
    assert _word(state) == 0
