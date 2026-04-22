"""
generate_reference.py — generates reference fixture data for regression tests.

Reference system: 32-atom LJ argon in a 10x10x10 box, fixed seed=100.
Uses the same setup as conftest.py simulation_factory (dt=1e-3).

Run via:
    torchrun --nproc_per_node=2 -m distnnp.tests.fixtures.generate_reference

Output: distnnp/tests/fixtures/lj_reference_{integrator_name}_rank{r}.pt on each
rank, containing:
    {
        1:  {"positions": Tensor(n_local, 3), "forces": Tensor(n_local, 3)},
        10: {"positions": Tensor(n_local, 3), "forces": Tensor(n_local, 3)},
    }

One file is generated per integrator (euler, verlet, langevin).
"""

import os
import torch
import torch.distributed as dist
from pathlib import Path

from distnnp_model.empirical.lj_potential_model import LennardJonesPotentialModel
from distnnp_model.distnnp_model_impl import FeatureExchanger
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
    torch.manual_seed(seed)
    state = MolecularInfo()
    state.num_atoms = num_atoms
    state.atom_indices = torch.arange(num_atoms, dtype=torch.long, device=device)
    state.atom_positions = torch.rand(num_atoms, 3, device=device) * 10.0
    state.atom_velocities = torch.randn(num_atoms, 3, device=device) * 0.1
    return state


def _make_sim_config(cutoff: float, device: torch.device) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * 10.0
    cfg.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return cfg


def _make_env(rank: int, world_size: int, local_rank: int) -> DistEnvironment:
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = gpus_per_node
    env.num_nodes = world_size // gpus_per_node
    env.local_rank = local_rank
    return env

def generate_reference(integrator_name: str, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
    """Generate and save reference fixtures for a given integrator.

    Args:
        integrator_name: One of "euler", "verlet", or "langevin".
        rank: Global rank of this process.
        local_rank: Local rank (GPU index) on this node.
        world_size: Total number of ranks.
        device: CUDA device for this rank.
    """

    # Build MolecularInfo (32-atom system, fixed seed, same on all ranks)
    molecular_info = _make_global_state(32, 100, device)

    # Note: 3.5 is not sure if it's a proper cutoff value here
    sim_config = _make_sim_config(3.5, device)

    env = _make_env(rank, world_size, local_rank)

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
    lj_model = LennardJonesPotentialModel(epsilon, sigma)

    # LennardJonesPotentialModel stores the exchangers as position_exchanger and atom_type_exchanger (GPUDirectExchanger)
    # We use the position_exchanger's underlying FeatureExchanger for the manual setup in this script.
    exchanger = lj_model.position_exchanger.exchanger
    F = 3
    send_offsets = partition_info.send_offsets
    send_local_indices = partition_info.send_local_indices
    recv_offsets = partition_info.recv_offsets
    n_local = partition_info.local_atom_positions.shape[0]
    total_send = int(send_offsets[-1].item())
    total_recv = int(recv_offsets[-1].item())

    x_buffer = FeatureExchanger.allocate_ipc_buffer(n_local + total_recv, F, local_rank)
    grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(total_send, F, local_rank)

    x_handle = exchanger.export_ipc_handle(x_buffer)
    grad_handle = exchanger.export_ipc_handle(grad_recv_buffer)

    handle_size = len(x_handle)
    x_handle_tensor = torch.tensor(x_handle, dtype=torch.uint8, device=device)
    grad_handle_tensor = torch.tensor(grad_handle, dtype=torch.uint8, device=device)

    all_x_handles_tensor = [torch.zeros(handle_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    all_grad_handles_tensor = [torch.zeros(handle_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(all_x_handles_tensor, x_handle_tensor)
    dist.all_gather(all_grad_handles_tensor, grad_handle_tensor)
    all_x_handles = [t.cpu().tolist() for t in all_x_handles_tensor]
    all_grad_handles = [t.cpu().tolist() for t in all_grad_handles_tensor]

    all_recv_offsets_list = [torch.zeros(world_size + 1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_recv_offsets_list, recv_offsets)
    all_recv_offsets = torch.stack(all_recv_offsets_list)

    all_num_local_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_num_local_list, torch.tensor([n_local], dtype=torch.int64, device=device))
    all_num_local = torch.cat(all_num_local_list)

    exchanger.setup_ipc(
        x_buffer, grad_recv_buffer,
        all_x_handles, all_grad_handles,
        send_offsets, send_local_indices, recv_offsets,
        all_recv_offsets, all_num_local,
        n_local, F, rank, world_size,
    )

    event_handle = exchanger.create_and_export_event()
    event_handle_tensor = torch.tensor(event_handle, dtype=torch.uint8, device=device)
    all_event_handles_tensor = [torch.zeros(len(event_handle), dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(all_event_handles_tensor, event_handle_tensor)
    all_event_handles = [t.cpu().tolist() for t in all_event_handles_tensor]
    exchanger.setup_events(all_event_handles)

    forces = torch.zeros(n_local, F, dtype=torch.float32, device=device)
    # Broadcast mass to (n_local, 3) to support vectorized integration
    mass = torch.full((n_local, 3), 1.0, dtype=torch.float32, device=device)
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
        _, forces = simulation.step()
        if step in SNAPSHOT_STEPS:
            snapshots[step] = {
                "positions" : simulation.state.pos.clone(),
                "forces" : forces.clone(),
            }

    # Save snapshots from each integrator to our directory
    torch.save(snapshots, OUTPUT_DIR / f"lj_reference_{integrator_name}_rank{rank}.pt")


def main():
    """
    Generates and saves reference fixtures for all three 
    integrators separately, since each integrator produces 
    different trajectories from the same initial conditions.
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    generate_reference("euler", rank, local_rank, world_size, device)
    generate_reference("verlet", rank, local_rank, world_size, device)
    generate_reference("langevin", rank, local_rank, world_size, device)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
