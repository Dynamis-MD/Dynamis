from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
import torch
from distnnp_state import TorchDistEnv
import os
import gc

# TODO this code should be in setup
torch_distributed_env = TorchDistEnv()

local_rank = torch_distributed_env.local_rank
global_rank = torch_distributed_env.global_rank
world_size = torch_distributed_env.world_size
device = torch_distributed_env.device

sim_config = SimulatorConfig(
        timestep_fs=0.5,
        total_steps=100,
        cutoff_angstrom=6.0,
        side_length_angstrom=20.0,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=10,
        output_dir="./results",
        traj_file="trajectory.traj",
        traj_interval=5,
        log_interval=10,
        writer_buffer_size=10,
        backend="nccl",
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
    device=device,
)

simulator.setup("../../../baseline/ch450.xyz")
simulator.run()