"""
HIV Capsid MD Simulation using DistNNP
=======================================
Runs a Lennard-Jones molecular dynamics simulation of the full HIV-1 capsid
(PDB: 3J3Q, ~2.4 M atoms) distributed across one or more GPUs.

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

Limitations
-----------
- Hydrogen atoms are absent (heavy-atom only; see retrieve_data.py).
- No equilibration phase; velocities are initialised from a Maxwell-Boltzmann
  distribution at 310 K. A production simulation would require energy
  minimization, ~1 ns NVT equilibration, and ~1 ns NPT equilibration before
  collecting data (Perilla et al. 2017).
- No barostat; simulation runs NVT Langevin only.
- UFF LJ (Rappe et al. 1992) is used as a placeholder potential. Bonded pairs
  are excluded by distance threshold (< 2.2 Å).
"""

import gc

import torch
import torch.distributed as dist

from distnnp_integrator import Langevin
from distnnp_integrator.masses import TORCH_MASSES
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_state import TorchDistEnv

from setup_system import load_capsid_from_xyz

XYZ_FILE = "3J3Q.xyz"

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
        timestep_fs=1.0,
        total_steps=TOTAL_STEPS,
        cutoff_angstrom=10.0,
        side_length_angstrom=box_size,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        output_dir="./results",
        log_interval=5,
        backend="nccl",
    )

    # LJ parameters: each value is found via this paper - https://pubs.acs.org/doi/10.1021/ja00051a040
    # Not sure if all of these are needed but I just did everything in setup_system.py/ELEMENT_TO_Z to be safe
    _epsilons = { # Unit: amu * Å^2 * fs^-2
        1 : 1.84096e-5, 2 : 2.34304e-5, 3 : 1.046e-5, 4 : 3.5564e-5, 5 : 7.5312e-5,
        6 : 4.3932e-5, 7 : 2.88696e-5, 8 : 2.5104e-5, 9 : 2.092e-5, 10 : 1.75728e-5,
        11 : 1.2552e-5, 12 : 4.64424e-5, 13 : 2.11292e-4, 14 : 1.681968e-4, 15 : 1.27612e-4,
        16 : 1.146416e-4, 17 : 9.49768e-5, 18 : 7.7404e-5, 19 : 1.4644e-5, 20 : 9.95792e-5,
        21 : 7.9496e-6, 22 : 7.1128e-6, 23 : 6.6944e-6, 24 : 6.276e-6, 25 : 5.4392e-6,
        26 : 5.4392e-6, 27 : 5.8576e-6, 28 : 6.276e-6, 29 : 2.092e-6, 30 : 5.18816e-5,
        31 : 1.73636e-4, 32 : 1.585736e-4, 33 : 1.292856e-4, 34 : 1.217544e-4,
        35 : 1.050184e-4, 36 : 9.2048e-5, 53 : 1.418376e-4
    }
    _sigmas = { # Unit: Å
        1 : 2.571, 2 : 2.104, 3 : 2.184, 4 : 2.446, 5 : 3.638,
        6 : 3.431, 7 : 3.261, 8 : 3.118, 9 : 2.997, 10 : 2.889,
        11 : 2.658, 12 : 2.691, 13 : 4.008, 14 : 3.826, 15 : 3.695,
        16 : 3.595, 17 : 3.516, 18 : 3.446, 19 : 2.396, 20 : 3.028,
        21 : 2.936, 22 : 2.829, 23 : 2.801, 24 : 2.693, 25 : 2.638,
        26 : 2.595, 27 : 3.559, 28 : 2.525, 29 : 3.114, 30 : 2.462,
        31 : 3.905, 32 : 3.813, 33 : 3.769, 34 : 3.746, 35 : 3.732,
        36 : 3.689, 53 : 4.009
    }

    epsilons = torch.zeros(100).to(device=device)
    sigmas = torch.zeros(100).to(device=device)
    for z, val in _epsilons.items():
        epsilons[z] = val
    for z, val in _sigmas.items():
        sigmas[z] = val

    # Langevin NVT thermostat parameters.
    # kT at 310 K (physiological) in amu·Å²·fs⁻² — must match setup_system.py.
    KT = 2.577e-4   # amu·Å²·fs⁻²  (k_B · 310 K)
    GAMMA = 0.01    # fs⁻¹  (matches baseline/dist_md.py default), might 
                    # need tuning I don't know if this is good or not

    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
    integrator = Langevin(dt=sim_config.timestep_fs, gamma=GAMMA, kT=KT)

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
            f"NVT Langevin T=310K γ={GAMMA} fs⁻¹ | "
            f"{world_size} rank(s)\n"
        )

    for step in range(TOTAL_STEPS):
        simulator.step(sim_config.timestep_fs)

        local = simulator.state.local_state
        max_vel = local.atom_velocities.abs().max().item()

        if step % sim_config.log_interval == 0:
            # Instantaneous kinetic temperature: T = 2*KE / (Ndof * k_B)
            # k_B in amu·Å²·fs⁻²·K⁻¹
            KB = 8.314e-7
            atom_masses = TORCH_MASSES.to(local.atom_velocities.device)[local.atom_types.reshape(-1)]
            ke_local = (0.5 * atom_masses * (local.atom_velocities ** 2).sum(dim=1)).sum()
            ke_tensor = ke_local.clone()
            if world_size > 1:
                dist.all_reduce(ke_tensor, op=dist.ReduceOp.SUM)
            ndof = 3 * mol_info.num_atoms - 3
            t_kelvin = (2.0 * ke_tensor.item()) / (ndof * KB)

            if global_rank == 0:
                max_frc = local.atom_forces.abs().max().item()
                mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(
                    f"  step {step:4d}/{TOTAL_STEPS} | "
                    f"T={t_kelvin:.1f} K | "
                    f"max_vel={max_vel:.3e} Å/fs | max_frc={max_frc:.3e} | mem={mem_mb:.0f} MB"
                )

    torch_dist_env.barrier()

    if global_rank == 0:
        print(f"\nPeak GPU memory: {get_peak_mb():.1f} MB")
        print("Simulation complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
