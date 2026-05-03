from setup_system import generate_synthetic_dataset, Timer
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
from distnnp import SimState
import torch
from distnnp_state import TorchDistEnv
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


def create_and_run_experiment():

    torch_distributed_env = TorchDistEnv()
    local_rank = torch_distributed_env.local_rank
    global_rank = torch_distributed_env.global_rank
    world_size = torch_distributed_env.world_size
    device = torch_distributed_env.device
    side_lengths = []
    density = 0.005

    for i in range(world_size):
        if i == global_rank:
            print(
                f"Hello from global rank {global_rank}, local rank {local_rank}, world size {world_size}, device {device}!"
            )
        torch_distributed_env.barrier()

    def local_print(*args, **kwargs):
        if global_rank == 0:
            print(*args, **kwargs)

    system_sizes = [
        100,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
    ]  # Number of atoms in the system
    for system_size in system_sizes:

        side_length = (system_size / density) ** (
            1 / 3
        )  # atoms per cubic angstrom, assuming a density of 0.05 atoms/Å^3
        side_lengths.append(side_length)

        local_print(
            f"Running experiment for system size {system_size} atoms with side length {side_length:.2f} Å..."
        )

        sim_config = SimulatorConfig(
            timestep_fs=0.5,
            total_steps=1000,
            cutoff_angstrom=6.0,
            side_length_angstrom=side_length,  # Scale cell size with system size
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
            num_atoms=system_size, box_size=side_length, device=device
        )

        local_print(
            f"Global molecular system created with {global_molecular_system.num_atoms} atoms."
        )
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

        partitioner = simulator._get_partitioner()
        internal_simulation_config = simulator._get_simulation_config()

        for num_iterations in range(100):
            # local_print(
            #     f"Rank {global_rank} starting initialization iteration {num_iterations}..."
            # )
            with Timer(
                f"Partition(SFC Morton) - system size {system_size}",
                metadata={
                    "system_size": system_size,
                    "world_size": world_size,
                    "operation": "partition",
                },
            ):
                partition_info = partitioner.partition(
                    global_molecular_system, torch_distributed_env.dist_env
                )

            torch.distributed.barrier()
            torch.cuda.synchronize()

            with Timer(
                f"Build Halo Maps - system size {system_size}",
                metadata={
                    "system_size": system_size,
                    "world_size": world_size,
                    "operation": "build_halo_maps",
                },
            ):
                partitioner.build_halo_maps(
                    partition_info,
                    global_molecular_system,
                    torch_distributed_env.dist_env,
                    internal_simulation_config,
                )

            torch.distributed.barrier()
            torch.cuda.synchronize()

        simulator.initialize_simulator_state(global_molecular_system, device=device)
        # local_print(
        #     f"Simulator state initialized on rank {global_rank} with {simulator.state._local_molecular_info.num_atoms} local atoms."
        # )

    metadata_dict = {
        "world_size": world_size,
        "systen_sizes": system_sizes,
        "side_lengths": side_lengths,
        "density": density,
        "num_iterations": 100,
    }
    Timer.add_meta_data(metadata_dict)
    timing_data_json = Timer.json()

    # local_print(f"Aggregated timing data across all iterations: {timing_data_json}")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(cur_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if global_rank == 0:
        with open(
            os.path.join(results_dir, f"single_layer_timing_data_{world_size}.json"),
            "w",
        ) as f:
            f.write(timing_data_json)

    # for step_num in range(sim_config.total_steps):
    #     step_size = sim_config.timestep_fs
    #     simulator.step(step_size)

    #     if step_num % sim_config.rebalance_interval == 0:
    #         simulator.sync_and_repartition()

    #     if step_num % sim_config.log_interval == 0:
    #         log_state(simulator_state, step_num)


def main():
    create_and_run_experiment()


if __name__ == "__main__":
    main()
