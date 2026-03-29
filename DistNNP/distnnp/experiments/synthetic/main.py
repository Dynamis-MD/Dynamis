from setup_system import create_dist_nnp_state, sync_and_repartition
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
from distnnp_state import DistNNPState
import torch


def log_state(state: DistNNPState, step_num: int):
    # Placeholder writer
    rank = state.dist_env.global_rank

    local_torch_state = {}
    local_torch_state["pos"] = state.local_partition_info.local_atom_positions
    local_torch_state["vel"] = state.local_partition_info.local_atom_velocities
    local_torch_state["atom"] = state.local_partition_info.partition_indices

    torch.save(local_torch_state, f"rank_{rank}_step_{step_num}.pt")


def create_and_run_experiment(num_atoms: int):
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

    dist_nnp_state = create_dist_nnp_state(num_atoms)

    epsilons = torch.rand(100, requires_grad=True).to(device="cuda")
    sigmas = torch.rand(100, requires_grad=True).to(device="cuda")

    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)

    integrator = Verlet(sim_config.timestep_fs)
    simulator = Simulator(potential, integrator, sim_config)

    simulator._set_state(dist_nnp_state)

    for step_num in range(sim_config.total_steps):
        step_size = sim_config.timestep_fs
        simulator.step(step_size)

        if step_num % sim_config.rebalance_interval == 0:
            dist_nnp_state = simulator.state
            repartitioned_state = sync_and_repartition(dist_nnp_state)
            simulator._set_state(repartitioned_state)

        if step_num % sim_config.log_interval == 0:
            log_state(dist_nnp_state, step_num)


def main():
    create_and_run_experiment(100)


if __name__ == "__main__":
    main()
