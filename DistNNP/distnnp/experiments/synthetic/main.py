from setup_system import generate_synthetic_dataset
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
from distnnp import SimState
import torch
from distnnp_state import TorchDistEnv


def log_state(state: SimState, step_num: int):
    # Placeholder writer
    rank = state.dist_env.global_rank

    local_torch_state = {}
    local_torch_state["pos"] = state._local_molecular_info.atom_positions
    local_torch_state["vel"] = state._local_molecular_info.atom_velocities
    local_torch_state["atom"] = state._local_molecular_info.atom_types

    torch.save(local_torch_state, f"rank_{rank}_step_{step_num}.pt")


def create_and_run_experiment(num_atoms: int):

    torch_distributed_env = TorchDistEnv()

    device = torch_distributed_env.device

    sim_config = SimulatorConfig(
        timestep_fs=0.5,
        total_steps=1000,
        cutoff_angstrom=6.0,
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

    global_molecular_system = generate_synthetic_dataset(num_atoms=1000, device=device)
    epsilons = torch.rand(100, requires_grad=True).to(device=device)
    sigmas = torch.rand(100, requires_grad=True).to(device=device)
    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
    integrator = Verlet(sim_config.timestep_fs)
    simulator = Simulator(
        model=potential,
        integrator=integrator,
        dist_environment=torch_distributed_env.dist_env,
        config=sim_config,
    )

    simulator.initialize_simulator_state(global_molecular_system)

    for step_num in range(sim_config.total_steps):
        step_size = sim_config.timestep_fs
        simulator.step(step_size)

        if step_num % sim_config.rebalance_interval == 0:
            simulator.sync_and_repartition()

        if step_num % sim_config.log_interval == 0:
            simulator_state = simulator.state
            log_state(simulator_state, step_num)


def main():
    create_and_run_experiment(100)


if __name__ == "__main__":
    main()
