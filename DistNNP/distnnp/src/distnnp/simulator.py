from distnnp_state import (
    SimpleSpatialPartitioner,
    KDTreePartitioner,
    SFCPartitioner,
)
from distnnp_model import DistNNPModel
from distnnp_state import DistNNPState, DistEnvironment, SimState
from distnnp_integrator import Euler, Verlet, Langevin, AVALIABLE_INTEGRATORS
from typing import Union, Literal
from dataclasses import dataclass
import torch
from torch import distributed as dist
import os

POSSIBLE_INTEGRATOR_TYPES = Union[Euler, Verlet, Langevin]

POSSIBLE_PARTITIONERS_TYPES = Union[
    SimpleSpatialPartitioner, KDTreePartitioner, SFCPartitioner
]

VALID_PARTITION_ALGORITHMS = ["simple", "kdtree", "sfc_morton", "sfc_hilbert"]


# Typing check if integrator is one of the available integrators
def check_integrator(integrator):
    if integrator not in AVALIABLE_INTEGRATORS:
        raise ValueError(
            f"Integrator {integrator} is not available. Choose from {AVALIABLE_INTEGRATORS}."
        )


@dataclass
class SimulatorConfig:
    # Physical parameters
    timestep_fs: float = 0.5  # Integration timestep in femtoseconds
    total_steps: int = 1000  # Total number of MD steps to run

    # Spatial decomposition
    cutoff_angstrom: float = 6.0  # NNP neighbour cutoff radius (Å)
    ghost_buffer_angstrom: float = 0.5  # Extra radius for ghost atom selection (Å)
    # Valid values: "sfc_morton" | "sfc_hilbert" | "kdtree" | "simple"
    partition_algorithm: Literal["sfc_morton", "sfc_hilbert", "kdtree", "simple"] = (
        "sfc_morton"
    )

    edge_weighted_partition: bool = True

    # Repartitioning
    rebalance_interval: int = 100  # Repartition every N steps (0 = never)

    # Output
    output_dir: str = "./results"
    traj_file: str = (
        "trajectory.traj"  # Single shared output file (relative to output_dir)
    )
    traj_interval: int = (
        100  # Enqueue a trajectory snapshot every N steps (0 = disabled)
    )
    log_interval: int = 10  # Log energy/timing every N steps
    writer_buffer_size: int = 10  # Max snapshots held in memory before submit() blocks

    # Distributed backend
    backend: str = "nccl"  # PyTorch distributed backend ("nccl" | "gloo")
    """
    Args:
        timestep_fs (float): Integration timestep in femtoseconds.
        total_steps (int): Total number of MD steps to run.
        cutoff_angstrom (float): NNP neighbour cutoff radius in Ångstroms.
        ghost_buffer_angstrom (float): Extra radius for ghost atom selection in Ångstroms.
        partition_algorithm (str): Algorithm for spatial partitioning ("sfc_morton", "sfc_hilbert", "kdtree", "simple").
            "simple": index-order split, cheapest, good only for pre-sorted structures.
            "sfc_morton" / "sfc_hilbert": space-filling curve (recommended).
            "kdtree": recursive KD-tree bisection, best for anisotropic geometries.
        edge_weighted_partition (bool): Whether to weight partitions by edge count instead of atom count.
            When True, partition by dst-edge count (balanced GNN compute). When False, partition by atom count.
        rebalance_interval (int): Number of steps between repartitioning (0 = never).
        output_dir (str): Directory to save output files.
        traj_file (str): Name of the trajectory file relative to output_dir.
            A single shared file is output by default (relative path "trajectory.traj").
        traj_interval (int): Number of steps between trajectory snapshots (0 = disabled).
        log_interval (int): Number of steps between logging energy/timing information.
        writer_buffer_size (int): Maximum number of snapshots held in memory before submit() blocks.
        backend (str): PyTorch distributed backend to use ("nccl" or "gloo").
    """


class Simulator:
    def __init__(
        self,
        model: DistNNPModel,
        integrator: POSSIBLE_INTEGRATOR_TYPES,
        config: SimulatorConfig,
    ):
        self.model = model
        check_integrator(integrator)
        self.integrator = integrator
        self.config = config

        self.initialize_distributed()

    def initialize_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group(backend=self.config.backend)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "NO GPUs detected. CUDA is required for distributed simulation."
            )

        num_gpus_per_node = max(
            torch.cuda.device_count(), int(os.getenv("PROCS_PER_NODE", 0))
        )

        local_rank = dist.get_rank() % num_gpus_per_node if num_gpus_per_node > 0 else 0
        local_device = torch.device(f"cuda:{local_rank}")

        torch.cuda.set_device(local_device)

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        num_nodes = (world_size + num_gpus_per_node - 1) // num_gpus_per_node

        self.local_device = local_device
        self.global_rank = global_rank
        self.world_size = world_size

        self.dist_env = DistEnvironment()
        self.dist_env.global_rank = global_rank
        self.dist_env.world_size = world_size
        self.dist_env.gpus_per_node = num_gpus_per_node
        self.dist_env.local_rank = local_rank
        self.dist_env.num_nodes = num_nodes

    def setup(self, file_path: str) -> None:
        pass

    def _set_state(self, state: SimState):
        self.state = state

    def step(self, step_size: float):
        self.integrator.step(self.state, self.model, step_size)
