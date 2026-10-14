"""
test_drift_watchdog.py — the drift watchdog and the flag word it piggybacks on the IPC
push-sync handshake.

    torchrun --nproc_per_node=2 -m pytest dynamis/tests/test_drift_watchdog.py -v

What must hold, on every rank:
  * after a step, global_ == bitwise OR over ranks of the words carried by that step;
  * the guarantee: on every step's graph, no atom is farther than the threshold from its
    halo-build position, with no breach recorded; decisions agree across ranks;
  * a breach on any rank is reported on every rank;
  * carrying the flag changes no numbers.
"""

import os

import pytest
import torch
import torch.distributed as dist

from dynamis.simulator import Simulator, SimulatorConfig
from dynamis.simulatorState import DRIFT_BREACH, DRIFT_TRIP
from dynamis_integrator.verlet import Verlet
from dynamis_model.comm import drift_flags
from dynamis_model.empirical.lj_potential_model import LennardJonesPotentialModel

from .conftest import _make_env, _make_global_state

NUM_ATOMS = 64
SMALL = 0.003  # A; the LJ test system moves ~0.0003 A per step


@pytest.fixture
def make_simulator(tmp_path):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    created = []

    def make(enable: bool = True, threshold=None) -> Simulator:
        molecular_info = _make_global_state(NUM_ATOMS, 100, device)
        env = _make_env(dist.get_rank(), dist.get_world_size(), local_rank)
        epsilon = torch.ones(100, dtype=torch.float32, device=device)
        sigma = torch.ones(100, dtype=torch.float32, device=device)
        model = LennardJonesPotentialModel(epsilon, sigma)
        config = SimulatorConfig(
            cutoff_angstrom=3.5,
            ghost_buffer_angstrom=0.5,
            side_length_angstrom=8.0,
            partition_algorithm="simple",
            rebalance_interval=0,  # only the watchdog repartitions
            enable_drift_watchdog=enable,
            drift_threshold_angstrom=threshold,
            output_dir=str(tmp_path),
            traj_interval=0,
        )
        sim = Simulator(model, Verlet(dt=1e-3), env, config, device)
        sim.initialize_simulator_state(molecular_info, device=device)
        created.append(model)
        return sim

    yield make

    for model in created:
        model.position_exchanger.exchanger.teardown()
        model.atom_type_exchanger.exchanger.teardown()
    drift_flags.disable()


def _all_ranks(value: int, device) -> list:
    t = torch.tensor([value], dtype=torch.int64, device=device)
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return [int(o.item()) for o in out]


def _global_max_displacement(pos: torch.Tensor, ref: torch.Tensor) -> float:
    d = (pos - ref).norm(dim=1).max() if pos.shape[0] else pos.new_zeros(())
    d = d.reshape(1).clone()
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    return float(d.item())


def test_default_threshold_is_half_skin(make_simulator):
    assert SimulatorConfig().enable_drift_watchdog is True
    assert SimulatorConfig().drift_threshold_angstrom is None
    sim = make_simulator()
    assert sim._drift_threshold == pytest.approx(0.5 / 2)


def test_word_or_over_many_pushes(make_simulator):
    """Random 2-bit words per rank over many generations (both generation slots, both
    parities), with per-rank GPU jitter: global_ is exactly the bitwise OR."""
    sim = make_simulator()
    flags = drift_flags.get(sim.device)
    rank, world = dist.get_rank(), dist.get_world_size()
    gen = torch.Generator().manual_seed(1234)  # same stream on every rank

    for it in range(200):
        words = torch.randint(0, 4, (world,), generator=gen)
        delay = torch.randint(0, 200_000, (world,), generator=gen)
        if it < 4:  # all zero, only rank 0, only the last rank, all set
            words = torch.tensor(
                [
                    [0] * world,
                    [1] + [0] * (world - 1),
                    [0] * (world - 1) + [2],
                    [3] * world,
                ][it]
            )
        flags.local.fill_(int(words[rank]))
        torch.cuda._sleep(int(delay[rank]))
        # the bare model, so our word is not overwritten before the pushes carry it
        sim.integrator.step(sim.state._state, sim.model, None)
        expected = 0
        for w in words.tolist():
            expected |= w
        got = int(flags.global_.item())
        assert got == expected, f"iteration {it}: words={words.tolist()} global={got}"
        assert len(set(_all_ranks(got, sim.device))) == 1

    assert flags.n_wired > 0, "no exchanger picked the flags up"


def _run_and_check_guarantee(sim, steps):
    """Drive the watchdog as run() does and check, on every step, the property the
    watchdog exists for: the graph is built with every atom within the threshold of its
    halo-build position. Also check the decision is the OR of the words the last step
    carried, and that ranks agree."""
    thr = sim._drift_threshold
    flags = sim._drift_flags
    ref = sim.state.pos.detach().clone()
    carried = None  # OR over ranks of the words carried by the last step
    trips, worst = 0, 0.0
    for t in range(steps):
        if t > 0:
            tripped = sim.drift_watchdog_tripped()
            assert len(set(_all_ranks(int(tripped), sim.device))) == 1
            assert tripped == bool(carried & DRIFT_TRIP)
            if tripped:
                trips += 1
                sim.rebuild_halo()  # what run() does on a drift trip
                ref = sim.state.pos.detach().clone()
        sim.step()
        # Verlet builds the graph after halfstep1; positions do not move afterwards
        d = _global_max_displacement(sim.state.pos.detach(), ref)
        worst = max(worst, d)
        assert d <= thr, f"step {t}: graph built at {d:.5f} A > {thr} A"
        carried = 0
        for w in _all_ranks(int(flags.local.item()), sim.device):
            carried |= w
    assert sim.drift_breach_steps == []
    return trips, worst


def test_guarantee_holds_piggyback(make_simulator):
    sim = make_simulator(threshold=SMALL)
    trips, worst = _run_and_check_guarantee(sim, 120)
    assert sim._drift_mode == "piggyback"
    assert trips >= 3, f"only {trips} trips; worst displacement {worst}"
    assert worst > 0.5 * SMALL  # the gate is not tripping trivially early


def test_guarantee_holds_all_reduce_fallback(make_simulator):
    sim = make_simulator(threshold=SMALL)
    sim._drift_mode = "all_reduce"
    trips, _ = _run_and_check_guarantee(sim, 120)
    assert trips >= 3


def test_breach_is_reported_on_every_rank(make_simulator):
    """An atom teleported past the threshold on ONE rank: every rank sees the breach
    for that step, warns, and trips."""
    sim = make_simulator(threshold=0.25)
    sim.sync_and_repartition()
    sim.step()
    assert not sim.drift_watchdog_tripped()
    if dist.get_rank() == 0:
        sim.state.pos[0, 0] += 1.0
    sim.step()  # step index 1: its graph is built with the teleported atom
    with pytest.warns(RuntimeWarning, match="drift watchdog"):
        tripped = sim.drift_watchdog_tripped()
    assert tripped
    assert sim.drift_breach_steps == [1]
    assert int(sim._drift_flags.global_.item()) & DRIFT_BREACH


def test_rebuild_halo_keeps_partition_and_state(make_simulator):
    """A halo-only rebuild refreshes the ghost lists on the current positions but
    keeps ownership, local order, velocities and the integrator's cached forces; the
    reference is re-latched and the word cleared."""
    sim = make_simulator(threshold=SMALL)
    sim.sync_and_repartition()
    for _ in range(15):
        sim.step()
    st = sim.state
    pidx = st.local_partition_info.partition_indices.clone()
    owner = st.local_partition_info.partition_id.clone()
    pos, vel = st.pos.detach().clone(), st.vel.detach().clone()
    forces = sim.integrator.prev_forces
    assert forces is not None

    sim.rebuild_halo()

    assert torch.equal(st.local_partition_info.partition_indices, pidx)
    assert torch.equal(st.local_partition_info.partition_id, owner)
    assert torch.equal(st.pos, pos) and torch.equal(st.vel, vel)
    assert sim.integrator.prev_forces is forces
    assert torch.equal(st._ref_positions, pos)
    assert int(sim._drift_flags.local.item()) == 0
    # the gathered global positions are exactly every rank's current local positions
    full = st._state.global_molecular_info.atom_positions
    assert torch.equal(full[pidx], pos)
    sim.step()  # the rebuilt exchangers work


def test_run_rebuilds_halo_on_trip_and_repartitions_on_schedule(make_simulator):
    sim = make_simulator(threshold=SMALL)
    sim.config.rebalance_interval = 40
    sim.config.total_steps = 100
    sim.config.timestep_fs = 1e-3  # run() steps with this; match the Verlet dt above
    calls = {"full": 0, "halo": 0}
    full, halo = sim.sync_and_repartition, sim.rebuild_halo

    def count_full(*a, **k):
        calls["full"] += 1
        full(*a, **k)

    def count_halo(*a, **k):
        calls["halo"] += 1
        halo(*a, **k)

    sim.sync_and_repartition, sim.rebuild_halo = count_full, count_halo
    sim.run()
    assert calls["full"] == 3  # steps 0, 40, 80
    assert calls["halo"] == len(sim.drift_rebuild_steps) >= 3
    assert not set(sim.drift_rebuild_steps) & {0, 40, 80}
    assert sim.drift_breach_steps == []
    assert len(set(_all_ranks(len(sim.drift_rebuild_steps), sim.device))) == 1


def test_carrying_flags_does_not_change_trajectory(make_simulator):
    """A watchdog that never trips must leave the trajectory bitwise unchanged."""

    def run(enable: bool) -> torch.Tensor:
        sim = make_simulator(enable=enable, threshold=1e9)
        for t in range(20):
            if t > 0 and sim.drift_watchdog_tripped():
                sim.sync_and_repartition()
            sim.step()
        idx = sim.state.local_partition_info.partition_indices
        return sim.state.pos.detach()[torch.argsort(idx)].clone()

    with_flags = run(enable=True)
    drift_flags.disable()
    without = run(enable=False)
    assert torch.equal(with_flags, without)


def test_disabled_watchdog_is_inert(make_simulator):
    sim = make_simulator(enable=False)
    assert drift_flags.get(sim.device) is None
    for _ in range(3):
        sim.step()
        assert sim.drift_watchdog_tripped() is False
    assert sim.model.position_exchanger._wired_flags is None


def test_new_simulator_starts_clean(make_simulator):
    """A second Simulator in the same process must not inherit the first one's flag
    values (spurious trip) or its wiring count (piggyback mode with nothing wired)."""
    old = make_simulator(threshold=1e9)
    old.step()
    old._drift_flags.local.fill_(DRIFT_TRIP | DRIFT_BREACH)
    old._drift_flags.global_.fill_(DRIFT_TRIP | DRIFT_BREACH)

    sim = make_simulator(threshold=1e9)
    flags = sim._drift_flags
    assert flags is not old._drift_flags
    assert flags.n_wired == 0
    assert int(flags.local.item()) == 0 and int(flags.global_.item()) == 0

    sim.step()
    assert flags.n_wired > 0
    assert sim.drift_watchdog_tripped() is False
    assert sim._drift_mode == "piggyback"
