"""
HIV Capsid MD Simulation using DistNNP
=======================================
Runs a short Lennard-Jones molecular dynamics simulation of the full HIV-1
capsid (PDB: 3J3Q, ~2.4 M atoms) distributed across one or more GPUs.

Prerequisites
-------------
1. Run retrieve_data.py in this directory to download and convert the PDB
   structure to 3J3Q.xyz:

       python retrieve_data.py

2. Install all DistNNP sub-packages (from the DistNNP/ repo root):

       pip install -e ./distnnp_state -e ./distnnp_integrator \
                   -e ./distnnp_model -e ./distnnp

Running
-------
Single GPU:
    python main.py

Multi-GPU (torchrun):
    torchrun --nproc_per_node=<NUM_GPUS> main.py

Multi-GPU (SLURM):
    srun --ntasks=<NUM_GPUS> --gpus-per-task=1 python main.py
"""

import gc
import os

import torch

from distnnp_integrator import Verlet
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_state import TorchDistEnv

from setup_system import load_capsid_from_xyz

XYZ_FILE = "3J3Q.xyz"

# Number of MD steps to run. The full capsid is large; 100 steps is enough to
# exercise the distributed pipeline end-to-end.
TOTAL_STEPS = 100


def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def main():
    torch_dist_env = TorchDistEnv()
    global_rank = torch_dist_env.global_rank
    world_size = torch_dist_env.world_size
    device = torch_dist_env.device

    for i in range(world_size):
        if i == global_rank:
            print(f"Rank {global_rank}/{world_size} on {device}")
        torch_dist_env.barrier()

    # Load the HIV capsid structure from the XYZ file. Every rank reads the
    # same file; Simulator.initialize_simulator_state partitions the atoms
    # across ranks internally.
    mol_info, box_size = load_capsid_from_xyz(XYZ_FILE, device)

    sim_config = SimulatorConfig(
        timestep_fs=0.5,
        total_steps=TOTAL_STEPS,
        cutoff_angstrom=6.0,
        side_length_angstrom=box_size,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=50,
        output_dir="./results",
        traj_file="hiv_capsid.traj",
        traj_interval=10,
        log_interval=5,
        writer_buffer_size=5,
        backend="nccl",
    )

    # LJ parameters: 100-element tensors indexed by atom type (atomic number).
    # Random values are used here as stand-ins for real force-field parameters.
    epsilons = torch.rand(100, requires_grad=True).to(device=device)
    sigmas = torch.rand(100, requires_grad=True).to(device=device)
    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
    integrator = Verlet(sim_config.timestep_fs)

    simulator = Simulator(
        model=potential,
        integrator=integrator,
        dist_environment=torch_dist_env.dist_env,
        config=sim_config,
        device=device,
    )

    simulator.initialize_simulator_state(mol_info, device=device)
    torch_dist_env.barrier()
    reset_gpu()

    if global_rank == 0:
        print(
            f"\nHIV capsid simulation: {mol_info.num_atoms:,} atoms | "
            f"{TOTAL_STEPS} steps | box {box_size:.1f} Å | "
            f"{world_size} rank(s)\n"
        )

    for step in range(TOTAL_STEPS):
        simulator.step(sim_config.timestep_fs)

        if step % sim_config.rebalance_interval == 0 and step > 0:
            simulator.sync_and_repartition(device=device)

        if global_rank == 0 and step % sim_config.log_interval == 0:
            print(f"  step {step:4d}/{TOTAL_STEPS}")

    torch_dist_env.barrier()

    if global_rank == 0:
        print(f"\nPeak GPU memory: {get_peak_mb():.1f} MB")
        print("Simulation complete.")


if __name__ == "__main__":
    main()
