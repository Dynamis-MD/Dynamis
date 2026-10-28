"""
test_npt_simulation.py — NPT through the real Simulator on several ranks.

    torchrun --nproc_per_node=2 -m pytest dynamis/tests/test_npt_simulation.py -v

An ideal gas (no forces, so no halo traffic is needed) keeps this about the barostat,
the box bookkeeping and the drift watchdog rather than about a potential. What must
hold:
  * every rank ends with a bit-identical box (the stochastic term is rank-shared and
    the pressure is all-reduced), however many halo rebuilds happened;
  * the box actually moves, and stays finite;
  * the drift watchdog, which now has to look ahead over the barostat's own box update,
    never lets a step's graph be built past the displacement guarantee.
"""

import os

import pytest
import torch
import torch.distributed as dist

from dynamis.simulator import Simulator, SimulatorConfig
from dynamis_integrator import NPT
from dynamis_integrator.units import KB
from dynamis_model import DynamisModel

from .conftest import _make_env, _make_global_state

NUM_ATOMS = 64
VOLUME = 4000.0
KT = 300.0 * KB
P0 = (NUM_ATOMS + 1) * KT / VOLUME  # the ideal gas's mean volume is VOLUME


class IdealGasModel(DynamisModel):
    """No interactions. Needs no halo, but is a proper DynamisModel."""

    def __call__(self, distributed_state, *args, compute_stress=False, **kwargs):
        pos = distributed_state.local_molecular_info.atom_positions
        out = (torch.zeros((), device=pos.device), torch.zeros_like(pos))
        if compute_stress:
            return out + (torch.zeros(3, 3, device=pos.device),)
        return out

    def initialize_buffers(self, partition_info, device=None):
        pass


def _all_equal_across_ranks(tensor: torch.Tensor) -> bool:
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return all(torch.equal(g, gathered[0]) for g in gathered)


@pytest.fixture
def simulator(tmp_path):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    side = VOLUME ** (1 / 3)
    molecular_info = _make_global_state(NUM_ATOMS, 100, device)
    molecular_info.atom_positions *= side / 8.0  # conftest draws in an 8 A box
    molecular_info.atom_velocities *= (KT / 39.948) ** 0.5 / 0.1  # thermal at 300 K
    env = _make_env(dist.get_rank(), dist.get_world_size(), local_rank)
    config = SimulatorConfig(
        cutoff_angstrom=3.5,
        ghost_buffer_angstrom=0.5,
        side_length_angstrom=side,
        partition_algorithm="sfc_morton",
        rebalance_interval=0,
        enable_drift_watchdog=True,
        output_dir=str(tmp_path),
        traj_interval=0,
    )
    integrator = NPT(
        dt=1.0, gamma=0.02, kT=KT, pressure=P0, tau_p=1000.0, compressibility=1.0 / P0,
        max_strain_per_step=0.01, seed=7,
    )
    sim = Simulator(IdealGasModel(), integrator, env, config, device)
    sim.initialize_simulator_state(molecular_info, device=device)
    return sim


def _drive(sim, steps):
    """What Simulator.run does between steps, minus the trajectory writer."""
    rebuilds = 0
    for i in range(steps):
        if i > 0 and sim.drift_watchdog_tripped():
            sim.rebuild_halo()
            rebuilds += 1
        sim.step(sim.integrator.dt)
    return rebuilds


def test_the_box_is_identical_on_every_rank_and_actually_moves(simulator):
    sim = simulator
    cell0 = sim.state.sim_config.cell.clone()
    _drive(sim, 400)
    cell = sim.state.sim_config.cell
    assert torch.isfinite(cell).all()
    assert not torch.equal(cell, cell0), "the barostat never changed the box"
    assert _all_equal_across_ranks(cell), "ranks disagree about the box"


def test_the_watchdog_looks_ahead_over_the_barostat_and_never_breaches(simulator):
    sim = simulator
    assert sim.integrator.cell_change_bound_per_step > 0
    rebuilds = _drive(sim, 400)
    # the watchdog saw the integrator's declared bound ...
    assert sim.state.cell_change_bound_per_step == sim.integrator.cell_change_bound_per_step
    # ... halo rebuilds happened with a changing box, and no step ever outran the guarantee
    assert rebuilds > 0
    assert sim.drift_breach_steps == []


def test_a_repartition_keeps_the_box_and_the_barostat_running(simulator):
    sim = simulator
    _drive(sim, 100)
    cell_before = sim.state.sim_config.cell.clone()
    sim.state.sync_and_repartition()  # SimState-level: no trajectory writer needed
    sim.model.initialize_buffers(sim.state.local_partition_info, device=sim.device)
    sim.integrator.reset()
    assert torch.equal(sim.state.sim_config.cell, cell_before)  # not a box event
    _drive(sim, 100)
    assert _all_equal_across_ranks(sim.state.sim_config.cell)
