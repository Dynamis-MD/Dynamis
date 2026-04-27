"""
generate_reference.py — generates reference fixture data for regression tests.

Reference system: 3-atom LJ argon in a 10x10x10 box, fixed seed=100.
Uses the same setup as conftest.py simulation_factory (dt=1e-3).

Run via:
    torchrun --nproc_per_node=1 distnnp/tests/fixtures/generate_reference.py

Output: distnnp/tests/fixtures/lj_reference_{integrator_name}.pt containing:
    {
        1:  {"positions": Tensor(n_atoms, 3), "forces": Tensor(n_atoms, 3)},
        ...
    }

One file is generated per integrator (euler, verlet, langevin).
"""

import torch
import torch.distributed as dist
from pathlib import Path

from distnnp_model.empirical.lj_potential_model import lennard_jones_potential
from distnnp_state import radius_graph_pbc_cuda
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

# LJ wrapper for single GPU
class SimpleLJModel:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, state):
        pos = state.local_partition_info.local_atom_positions.clone().detach().requires_grad_(True)
        atom_types = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        sim_config = state.simulation_config
        edge_index, _, _ = radius_graph_pbc_cuda(
            data=type("Data", (object,), {
                "pos": pos, "cell": sim_config.cell.unsqueeze(0),
                "natoms": [pos.shape[0]], "pbc": sim_config.pbc,
            })(),
            radius=sim_config.cutoff,
            max_num_neighbors_threshold=0,
            enforce_max_neighbors_strictly=False,
        )
        energy = lennard_jones_potential(atom_types, pos, edge_index, self.epsilon, self.sigma)
        energy.backward()
        return energy.detach(), -pos.grad.detach()

# The steps at which to save snapshots
SNAPSHOT_STEPS = [1, 5, 10, 15, 20]
OUTPUT_DIR = Path(__file__).parent


def _make_global_state(num_atoms: int, seed: int, device: torch.device) -> MolecularInfo:
    """
    Create a MolecularInfo with random positions in a 10x10x10 box.
    We make this with a fixed seed but set that fixed seed as a parameter.
    """
    torch.manual_seed(seed)
    state = MolecularInfo()
    state.num_atoms = num_atoms
    
    # Generate on CPU to ensure all ranks start with the SAME universe
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long)
    state.atom_positions = torch.rand(num_atoms, 3) * 10.0
    state.atom_velocities = torch.randn(num_atoms, 3) * 0.1
    
    # Now move the consistent data to the GPU
    state.atom_vertex_numbers = state.atom_vertex_numbers.to(device)
    state.atom_positions = state.atom_positions.to(device)
    state.atom_velocities = state.atom_velocities.to(device)
    return state


def _make_sim_config(cutoff: float, device: torch.device) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * 10.0
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
    """Generate and save reference fixtures for a given integrator.

    Args:
        integrator_name: One of "euler", "verlet", or "langevin".
        device: CUDA device to run on.
    """

    # Build MolecularInfo (3-atom system, fixed seed)
    molecular_info = _make_global_state(3, 100, device)

    # Note: 3.5 is not sure if it's a proper cutoff value here
    sim_config = _make_sim_config(3.5, device)

    # Hardcoded values for 1 GPU
    env = _make_env(0, 1, 0)

    # We still need the partitioner but it won't do anything
    partitioner = SimpleSpatialPartitioner()
    partition_info = partitioner.partition(molecular_info, env)
    partitioner.build_halo_maps(partition_info, molecular_info, env, sim_config)

    distnnp_state = DistNNPState()
    distnnp_state.dist_env = env
    distnnp_state.global_molecular_info = molecular_info
    distnnp_state.local_partition_info = partition_info

    # These 1.0 values are fairly arbitrary, not sure if they should be changed
    epsilon = torch.tensor([1.0], dtype=torch.float32, device=device)
    sigma = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Initialize without a shared exchanger to allow independent lazy initialization
    lj_model = SimpleLJModel(epsilon, sigma)

    F = 3
    n_local = partition_info.local_atom_positions.shape[0]

    forces = torch.zeros(n_local, F, dtype=torch.float32, device=device)
    mass = torch.full((n_local,), 1.0, dtype=torch.float32, device=device)
    sim_state = SimState(distnnp_state, sim_config, forces, mass)

    # The constants are fairly arbitrary, not sure if they should be changed
    if integrator_name == "euler":
        integrator = Euler(dt=1e-3)
    elif integrator_name == "verlet":
        integrator = Verlet(dt=1e-3)
    elif integrator_name == "langevin":
        integrator = Langevin(dt=1e-3, gamma=1.0, kT=0.6)
    else:
        raise ValueError("Invalid integrator_name: must be euler, verlet or langevin")

    # This is required for deterministic results under langevin
    # It has some random elements
    torch.cuda.manual_seed(42)

    simulation = Simulation(sim_state, lj_model, integrator)

    # Run simulation and snapshot when needed
    snapshots = {}
    for step in range(1, max(SNAPSHOT_STEPS) + 1):
        energy, forces = simulation.step()
        if step in SNAPSHOT_STEPS:
            snapshots[step] = {
                "positions" : simulation.state.pos.clone(),
                "forces" : forces.clone(),
                "energy" : energy.clone(),
            }

    torch.save(snapshots, OUTPUT_DIR / f"lj_reference_{integrator_name}.pt")


def main():
    """
    Generates and saves reference fixtures for all three 
    integrators separately, since each integrator produces 
    different trajectories from the same initial conditions.
    """
    
    device = torch.device(f"cuda:0")
    
    dist.init_process_group("nccl")

    generate_reference("euler", device)
    generate_reference("verlet", device)
    generate_reference("langevin", device)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()