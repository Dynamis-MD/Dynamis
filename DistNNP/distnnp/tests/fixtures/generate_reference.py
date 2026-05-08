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
from distnnp.simulation import SimState, Simulation
from distnnp_integrator.euler import Euler
from distnnp_integrator.verlet import Verlet
from distnnp_integrator.langevin import Langevin
from distnnp_state import (
    MolecularInfo,
    SimulationConfig,
    DistEnvironment,
    SimpleSpatialPartitioner,
    DistNNPState,
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

    state.atom_vertex_numbers = state.atom_vertex_numbers.to(device)
    state.atom_positions = state.atom_positions.to(device)
    state.atom_velocities = state.atom_velocities.to(device)
    return state


def _make_sim_config(cutoff: float, device: torch.device) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * 8.0
    cfg.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return cfg


def _make_env(rank: int, world_size: int, local_rank: int) -> DistEnvironment:
    gpus_per_node = 1
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = gpus_per_node
    env.num_nodes = world_size // gpus_per_node
    env.local_rank = local_rank
    return env

def generate_reference(integrator_name: str, device: torch.device) -> None:
    """Generate and save the reference fixture for a given integrator.

    Args:
        integrator_name: One of "euler", "verlet", or "langevin".
        device: CUDA device to run on.
    """

    molecular_info = _make_global_state(16, 100, device)
    sim_config = _make_sim_config(3.5, device)

    # Single-rank reference run — world_size=1, local_rank=0
    env = _make_env(0, 1, 0)

    # Partitioner is a no-op at world_size=1 but still populates
    # local_atom_positions, which the langevin loop reads as initial state.
    partitioner = SimpleSpatialPartitioner()
    partition_info = partitioner.partition(molecular_info, env)
    partitioner.build_halo_maps(partition_info, molecular_info, env, sim_config)

    distnnp_state = DistNNPState()
    distnnp_state.dist_env = env
    distnnp_state.global_molecular_info = molecular_info
    distnnp_state.local_partition_info = partition_info

    epsilon = torch.tensor([1.0], dtype=torch.float32, device=device)
    sigma = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Use the same model class the regression tests run against so the fixture
    # exercises the real FeaturePushToExchange / exchanger code path.
    lj_model = LennardJonesPotentialModel(epsilon, sigma)

    n_local = partition_info.local_atom_positions.shape[0]

    if integrator_name == "langevin":
        # Reference for the langevin canonical-ensemble check in
        # test_simulation_regression.py: N_RUNS long trajectories,
        # time-average total KE/PE over the post-equilibration window.
        # tau = 1/gamma = ~1000 steps; N_EQUIL=1*tau, averaging over 1.5*tau.
        # Knobs must match the test side.
        N_RUNS = 4
        N_STEPS = 2500
        N_EQUIL = 1000

        initial_pos = partition_info.local_atom_positions.clone()
        initial_vel = molecular_info.atom_velocities.clone()

        ke_per_run, pe_per_run = [], []
        for run in range(N_RUNS):
            torch.manual_seed(1000 + run)

            forces = torch.zeros(n_local, 3, dtype=torch.float32, device=device)
            mass = torch.full((n_local,), 1.0, dtype=torch.float32, device=device)
            sim_state = SimState(distnnp_state, sim_config, forces, mass)
            sim_state.pos.copy_(initial_pos)
            sim_state.vel.copy_(initial_vel)
            integrator = Langevin(dt=1e-3, gamma=1.0, kT=0.6)
            simulation = Simulation(sim_state, lj_model, integrator)

            ke_acc, pe_acc = [], []
            for step in range(1, N_STEPS + 1):
                energy, _ = simulation.step()
                if step > N_EQUIL:
                    ke = 0.5 * (simulation.state.mass.unsqueeze(1) * simulation.state.vel ** 2).sum()
                    ke_acc.append(ke.item())
                    pe_acc.append(energy.item())
            ke_per_run.append(sum(ke_acc) / len(ke_acc))
            pe_per_run.append(sum(pe_acc) / len(pe_acc))

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
        # Deterministic integrators: a single trajectory is reproducible.
        integrator = Euler(dt=1e-3) if integrator_name == "euler" else Verlet(dt=1e-3)

        forces = torch.zeros(n_local, 3, dtype=torch.float32, device=device)
        mass = torch.full((n_local,), 1.0, dtype=torch.float32, device=device)
        sim_state = SimState(distnnp_state, sim_config, forces, mass)

        simulation = Simulation(sim_state, lj_model, integrator)

        snapshots = {}
        for step in range(1, max(SNAPSHOT_STEPS) + 1):
            energy, forces = simulation.step()
            if step in SNAPSHOT_STEPS:
                snapshots[step] = {
                    "positions": simulation.state.pos.clone(),
                    "forces":    forces.clone(),
                    "energy":    energy.clone(),
                }

        torch.save(snapshots, OUTPUT_DIR / f"lj_reference_{integrator_name}.pt")

    else:
        raise ValueError("Invalid integrator_name: must be euler, verlet or langevin")

    # Tear down exchangers before the next integrator builds its own model,
    # mirroring the conftest fixture teardown. At world_size=1 IPC is never
    # set up, so this is a no-op here but keeps the contract consistent.
    for wrapper in (
        lj_model.position_exchanger,
        lj_model.atom_type_exchanger,
        lj_model.global_index_exchanger,
    ):
        if wrapper.is_ipc_setup:
            wrapper.exchanger.teardown()


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