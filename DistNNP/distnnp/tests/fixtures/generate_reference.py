"""
generate_reference.py — generates reference fixture data for regression tests.

Reference system: 16-atom LJ argon in an 8x8x8 box, fixed seed=100.
Uses the same setup as conftest.py simulation_factory (dt=1e-3).

Run via:
    torchrun --nproc_per_node=1 distnnp/tests/fixtures/generate_reference.py

Output: distnnp/tests/fixtures/lj_reference_{integrator_name}.pt

For deterministic integrators (euler, verlet) each snapshot is:
    {"positions": Tensor(n_atoms, 3),
     "forces":    Tensor(n_atoms, 3),
     "energy":    Tensor()}

For langevin (stochastic) the fixture stores per-run time-averaged total
KE and PE from N_RUNS independent trajectories, used by the KS-based
regression test:
    {"ke_per_run": Tensor(N_RUNS,),
     "pe_per_run": Tensor(N_RUNS,),
     "n_runs":     int,
     "n_steps":    int,
     "n_equil":    int}

One file is generated per integrator (euler, verlet, langevin).
"""

import torch
import torch.distributed as dist
from pathlib import Path

from distnnp_model.empirical.lj_potential_model import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator.euler import Euler
from distnnp_integrator.verlet import Verlet
from distnnp_integrator.langevin import Langevin
from distnnp_integrator.masses import TORCH_MASSES
from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
)

# The steps at which to save snapshots
SNAPSHOT_STEPS = [1, 5, 10, 15, 20]
OUTPUT_DIR = Path(__file__).parent


def _make_global_state(num_atoms: int, seed: int, device: torch.device) -> MolecularInfo:
    """
    Build a MolecularInfo with random positions/velocities in an 8x8x8 box,
    using a fixed seed so every rank gets the same initial universe.
    """
    torch.manual_seed(seed)
    state = MolecularInfo()
    state.num_atoms = num_atoms

    # Generate on CPU first so all ranks see identical initial state
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long)
    state.atom_positions = torch.rand(num_atoms, 3) * 8.0
    state.atom_velocities = torch.randn(num_atoms, 3) * 0.1
    # Argon — atomic number 18; TORCH_MASSES is indexed by atomic number,
    # and index 0 is a 0.0 placeholder that would give infinite acceleration.
    state.atom_types = torch.full((num_atoms,), 18, dtype=torch.long)

    state.atom_vertex_numbers = state.atom_vertex_numbers.to(device)
    state.atom_positions = state.atom_positions.to(device)
    state.atom_velocities = state.atom_velocities.to(device)
    state.atom_types = state.atom_types.to(device)
    return state


def _make_env(rank: int, world_size: int, local_rank: int) -> DistEnvironment:
    gpus_per_node = 1
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = gpus_per_node
    env.num_nodes = world_size // gpus_per_node
    env.local_rank = local_rank
    return env

def _build_simulator(molecular_info, env, integrator, device):
    epsilon = torch.tensor([1.0], dtype=torch.float32, device=device)
    sigma = torch.tensor([1.0], dtype=torch.float32, device=device)
    lj_model = LennardJonesPotentialModel(epsilon, sigma)

    # Constants are fairly arbitrary
    # Not sure if the partition algo should be adjusted
    config = SimulatorConfig(
        cutoff_angstrom=3.5,
        ghost_buffer_angstrom=0.5,
        side_length_angstrom=8.0,
        partition_algorithm="simple",
    )

    simulator = Simulator(
        model=lj_model,
        integrator=integrator,
        dist_environment=env,
        config=config,
        device=device,
    )
    simulator.initialize_simulator_state(molecular_info, device=device)
    return simulator, lj_model


def _teardown_model(lj_model):
    for wrapper in (
        lj_model.position_exchanger,
        lj_model.atom_type_exchanger,
    ):
        if wrapper.is_ipc_setup:
            wrapper.exchanger.teardown()


def generate_reference(integrator_name: str, device: torch.device) -> None:
    """Generate and save the reference fixture for a given integrator.

    Args:
        integrator_name: One of "euler", "verlet", or "langevin".
        device: CUDA device to run on.
    """

    # Single-rank reference run — world_size=1, local_rank=0
    env = _make_env(0, 1, 0)

    if integrator_name == "langevin":
        # Reference for the langevin canonical-ensemble check in
        # test_simulation_regression.py: N_RUNS long trajectories,
        # time-average total KE/PE over the post-equilibration window.
        # tau = 1/gamma = ~1000 steps; N_EQUIL=1*tau, averaging over 1.5*tau.
        # Knobs must match the test side.
        N_RUNS = 4
        N_STEPS = 2500
        N_EQUIL = 1000

        ke_per_run, pe_per_run = [], []
        for run in range(N_RUNS):
            torch.manual_seed(1000 + run)
            molecular_info = _make_global_state(16, 100, device)
            integrator = Langevin(dt=1e-3, gamma=1.0, kT=0.6)
            simulator, lj_model = _build_simulator(molecular_info, env, integrator, device)

            ke_acc, pe_acc = [], []
            for step in range(1, N_STEPS + 1):
                energy, _ = simulator.step()
                if step > N_EQUIL:
                    local = simulator.state.local_state
                    local_masses = TORCH_MASSES.to(local.atom_positions.device)[local.atom_types.reshape(-1)]
                    ke = 0.5 * (local_masses.unsqueeze(1) * local.atom_velocities ** 2).sum()
                    ke_acc.append(ke.item())
                    pe_acc.append(energy.item())
            ke_per_run.append(sum(ke_acc) / len(ke_acc))
            pe_per_run.append(sum(pe_acc) / len(pe_acc))

            _teardown_model(lj_model)

        torch.save(
            {
                "ke_per_run": torch.tensor(ke_per_run),
                "pe_per_run": torch.tensor(pe_per_run),
                "n_runs": N_RUNS,
                "n_steps": N_STEPS,
                "n_equil": N_EQUIL,
            },
            OUTPUT_DIR / f"lj_reference_{integrator_name}.pt",
        )

    elif integrator_name in ("euler", "verlet"):
        molecular_info = _make_global_state(16, 100, device)
        integrator = Euler(dt=1e-3) if integrator_name == "euler" else Verlet(dt=1e-3)
        simulator, lj_model = _build_simulator(molecular_info, env, integrator, device)

        snapshots = {}
        for step in range(1, max(SNAPSHOT_STEPS) + 1):
            energy, forces = simulator.step()
            if step in SNAPSHOT_STEPS:
                snapshots[step] = {
                    "positions": simulator.state.pos.clone(),
                    "forces":    forces.clone(),
                    "energy":    energy.clone(),
                }

        torch.save(snapshots, OUTPUT_DIR / f"lj_reference_{integrator_name}.pt")
        _teardown_model(lj_model)

    else:
        raise ValueError("Invalid integrator_name: must be euler, verlet or langevin")


def main():
    """
    Generate and save reference fixtures for all three integrators. Each
    integrator produces a different trajectory from the same initial
    conditions, so each gets its own file.
    """

    device = torch.device("cuda:0")
    dist.init_process_group("nccl")

    generate_reference("euler", device)
    generate_reference("verlet", device)
    generate_reference("langevin", device)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()