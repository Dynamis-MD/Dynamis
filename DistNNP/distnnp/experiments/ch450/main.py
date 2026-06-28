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
        rebalance_interval=2,
        output_dir="./results",
        traj_file="trajectory.distnnp",
        traj_interval=1,
        log_interval=1,
        writer_buffer_size=10,
        backend="nccl",
    )

epsilons = torch.full((100,), 0.030, requires_grad=True, device=device)
sigmas   = torch.full((100,), 1.0,   requires_grad=True, device=device)
epsilons.data[1] = 0.030  # H
sigmas.data[1]   = 2.5    # H (Å)
epsilons.data[6] = 0.086  # C
sigmas.data[6]   = 3.4    # C (Å)
potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
integrator = Verlet(sim_config.timestep_fs)
simulator = Simulator(
    model=potential,
    integrator=integrator,
    dist_environment=torch_distributed_env.dist_env,
    config=sim_config,
    device=device,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
simulator.setup(os.path.normpath(os.path.join(_HERE, "../../../../baseline/ch450.xyz")))
simulator.run()