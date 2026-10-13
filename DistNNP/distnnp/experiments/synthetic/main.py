from setup_system import generate_synthetic_dataset, create_dist_sim_state
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
from distnnp import SimState
import torch
from dynamis_state import TorchDistEnv
import os
import gc


def log_state(state: SimState, step_num: int):
    # Placeholder writer
    rank = state.dist_env.global_rank

    local_torch_state = {}
    local_torch_state["pos"] = state._local_molecular_info.atom_positions
    local_torch_state["vel"] = state._local_molecular_info.atom_velocities
    local_torch_state["atom"] = state._local_molecular_info.atom_types

    torch.save(local_torch_state, f"rank_{rank}_step_{step_num}.pt")


def reset_gpu():
    """Reset GPU memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def create_and_run_experiment(num_atoms: int):
    torch_distributed_env = TorchDistEnv()

    local_rank = torch_distributed_env.local_rank
    global_rank = torch_distributed_env.global_rank
    world_size = torch_distributed_env.world_size
    device = torch_distributed_env.device

    for i in range(world_size):
        if i == global_rank:
            print(
                f"Hello from global rank {global_rank}, local rank {local_rank}, world size {world_size}, device {device}!"
            )
        torch_distributed_env.barrier()

    sim_config = SimulatorConfig(
        timestep_fs=0.5,
        total_steps=1000,
        cutoff_angstrom=6.0,
        side_length_angstrom=20.0,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        # integrator="verlet",
        edge_weighted_partition=True,
        rebalance_interval=100,
        output_dir="./results",
        traj_file="trajectory.traj",
        traj_interval=100,
        log_interval=10,
        writer_buffer_size=10,
        backend="nccl",
    )

    global_molecular_system = generate_synthetic_dataset(
        num_atoms=1000, box_size=20.0, device=device
    )

    dist_nnp_state = create_dist_sim_state(num_atoms=num_atoms)
    epsilons = torch.rand(100, requires_grad=True).to(device=device)
    sigmas = torch.rand(100, requires_grad=True).to(device=device)
    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
    integrator = Verlet(sim_config.timestep_fs)
    simulator = Simulator(
        model=potential,
        integrator=integrator,
        dist_environment=torch_distributed_env.dist_env,
        config=sim_config,
        device=device,
    )
    simulator.initialize_simulator_state(global_molecular_system, device=device)
    torch_distributed_env.barrier()
    simulator.step(sim_config.timestep_fs)
    torch_distributed_env.barrier()


def main():
    create_and_run_experiment(100)


if __name__ == "__main__":
    main()
