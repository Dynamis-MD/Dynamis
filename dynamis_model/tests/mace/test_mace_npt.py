"""NPT with a real potential: the sign and the bookkeeping of the pressure.

An ideal gas cannot test the stress sign (its stress is zero), so these use MACE on
silicon. MACE-MP's equilibrium lattice constant for Si is ~5.47 A, so a crystal at
0.98 x 5.43 A is under several GPa of pressure and one at 1.04 x 5.43 A under several
GPa of tension. NPT at zero target pressure must expand the first and contract the
second; a flipped stress sign would reverse both.

    pytest dynamis_model/tests/mace/test_mace_npt.py                          # 1 process
    torchrun --nproc-per-node=2 -m pytest dynamis_model/tests/mace/test_mace_npt.py   # 2 ranks
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from stress_common import raw_from, scaled_silicon, build_state

from dynamis_integrator import NPT
from dynamis_integrator.masses import TORCH_MASSES
from dynamis_integrator.npt import cell_volume
from dynamis_integrator.units import BAR, GPA, KB
from dynamis_model.nnp.mace import MACEModel, load_raw_mace_model
from dynamis_state import TorchDistEnv

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
KT = 300.0 * KB
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def raw_model():
    try:
        return load_raw_mace_model("mp", "medium-mpa-0", DEVICE).to(DEVICE).eval()
    except Exception as exc:  # no network / weights
        pytest.skip(f"MACE-MP weights unavailable: {exc}")


def make_npt(**kwargs):
    defaults = dict(
        dt=2.0, gamma=0.01, kT=KT, pressure=0.0, tau_p=300.0,
        compressibility=1.0e-6 / BAR, max_strain_per_step=1e-3, seed=1,
    )
    defaults.update(kwargs)
    return NPT(**defaults)


def thermalise(state):
    local = state.local_molecular_info
    mass = TORCH_MASSES.to(local.atom_velocities.device)[local.atom_types.long()]
    torch.manual_seed(100 + (dist.get_rank() if dist.is_initialized() else 0))
    local.atom_velocities.copy_(
        torch.randn_like(local.atom_velocities) * (KT / mass[:, None]) ** 0.5
    )


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
@pytest.mark.parametrize(
    "scale, expands", [(0.98, True), (1.04, False)], ids=["compressed", "stretched"]
)
def test_the_box_relaxes_toward_zero_pressure_in_the_right_direction(
    raw_model, scale, expands
):
    env = TorchDistEnv(is_distributed=False)
    state = build_state(raw_from(scaled_silicon(scale)), float(raw_model.r_max), env)
    model = MACEModel(raw_model)
    model.initialize_buffers(state.local_partition_info, env.device, env.world_size)
    thermalise(state)
    npt = make_npt()

    v0 = cell_volume(state.sim_config.cell).item()
    npt.step(state, model)
    p_start = npt.instantaneous_pressure.item() / GPA
    for _ in range(299):
        npt.step(state, model)
    v_end = cell_volume(state.sim_config.cell).item()
    p_end = npt.instantaneous_pressure.item() / GPA

    if expands:
        assert p_start > 3.0 and v_end > 1.03 * v0
    else:
        assert p_start < -3.0 and v_end < 0.97 * v0
    assert abs(p_end) < 0.7 * abs(p_start), "the pressure did not relax"


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs torchrun --nproc-per-node>=2")
def test_two_ranks_with_real_halos_keep_one_box_and_the_watchdog_quiet(raw_model, tmp_path):
    from dynamis.simulator import Simulator, SimulatorConfig
    from dynamis_state import MolecularInfo

    env = TorchDistEnv(is_distributed=True)
    atoms = scaled_silicon(0.98)
    mol = MolecularInfo()
    mol.num_atoms = len(atoms)
    mol.atom_vertex_numbers = torch.arange(len(atoms)).to(env.device)
    mol.atom_positions = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(env.device)
    mol.atom_types = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(env.device)
    mass = TORCH_MASSES.to(env.device)[mol.atom_types]
    torch.manual_seed(100)  # the same universe on every rank
    mol.atom_velocities = torch.randn(len(atoms), 3, device=env.device) * (KT / mass[:, None]) ** 0.5

    config = SimulatorConfig(
        cutoff_angstrom=float(raw_model.r_max), ghost_buffer_angstrom=0.5,
        side_length_angstrom=float(atoms.cell[0, 0]), partition_algorithm="sfc_morton",
        rebalance_interval=0, enable_drift_watchdog=True, output_dir=str(tmp_path), traj_interval=0,
    )
    sim = Simulator(MACEModel(raw_model), make_npt(), env.dist_env, config, env.device)
    sim.initialize_simulator_state(mol, device=env.device)
    # What setup()'s sync_and_repartition does (it needs a trajectory writer, which this
    # test has no use for): MACEModel tracks the identity of the partition info it was
    # given, and initialize_simulator_state hands it SimState.partition_info, a copy.
    sim.model.initialize_buffers(sim.state.local_partition_info, device=env.device)

    v0 = cell_volume(sim.state.sim_config.cell).item()
    for i in range(200):
        if i > 0 and sim.drift_watchdog_tripped():
            sim.rebuild_halo()
        sim.step(sim.integrator.dt)

    cell = sim.state.sim_config.cell
    gathered = [torch.zeros_like(cell) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, cell.contiguous())
    assert all(torch.equal(g, gathered[0]) for g in gathered), "ranks disagree about the box"
    assert cell_volume(cell).item() > 1.02 * v0  # compressed crystal expands
    assert sim.drift_breach_steps == []
