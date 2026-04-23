from distnnp_state import (
    SimpleSpatialPartitioner,
    KDTreePartitioner,
    SFCPartitioner,
    SFCCurveType,
    DistEnvironment,
    MolecularInfo,
    SimulationConfig,
)
from distnnp_model import DistNNPModel
from distnnp.simulatorState import SimState
from distnnp_integrator import Euler, Verlet, Langevin, AVALIABLE_INTEGRATORS
from typing import Union, Literal, Optional
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
    if type(integrator) not in [Euler, Verlet, Langevin]:
        raise ValueError(
            f"Integrator {integrator} is not available. Choose from {AVALIABLE_INTEGRATORS}."
        )


@dataclass
class SimulatorConfig:
    # Physical parameters
    timestep_fs: float = 0.5  # Integration timestep in femtoseconds
    total_steps: int = 1000  # Total number of MD steps to run

    # Spatial decomposition
    side_length_angstrom: float = (
        10.0  # Initial side length of spatial partitioning cells (Å)
    )
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
        dist_environment: DistEnvironment,
        config: SimulatorConfig,
    ):
        self.model = model
        check_integrator(integrator)
        self.integrator = integrator
        self.config = config
        self.dist_environment = dist_environment

    def _get_partitioner(self) -> POSSIBLE_PARTITIONERS_TYPES:
        partition_algorithm = self.config.partition_algorithm
        if partition_algorithm not in VALID_PARTITION_ALGORITHMS:
            raise ValueError(
                f"Partition algorithm {partition_algorithm} is not valid. Choose from {VALID_PARTITION_ALGORITHMS}."
            )
        if partition_algorithm == "simple":
            return SimpleSpatialPartitioner()
        elif partition_algorithm == "kdtree":
            return KDTreePartitioner()
        elif partition_algorithm == "sfc_morton":
            return SFCPartitioner(curve_type=SFCCurveType.Morton)
        elif partition_algorithm == "sfc_hilbert":
            return SFCPartitioner(curve_type=SFCCurveType.Hilbert)
        else:
            raise ValueError(
                f"Partition algorithm {partition_algorithm} is not recognized."
            )

    def _get_dist_environment(self) -> DistEnvironment:
        return self.dist_environment

    def _get_simulation_config(self) -> SimulationConfig:
        sim_config = SimulationConfig()

        sim_config.cutoff = self.config.cutoff_angstrom
        sim_config.ghost_cutoff_buffer = self.config.ghost_buffer_angstrom
        sim_config.cell = torch.eye(3) * self.config.side_length_angstrom
        sim_config.pbc = torch.tensor([True, True, True], dtype=torch.bool)

        return sim_config

    def initialize_simulator_state(
        self, molecular_system: MolecularInfo, device: Optional[torch.device] = None
    ) -> None:
        self.state = SimState(
            molecular_system=molecular_system,
            partitioner=self._get_partitioner(),
            simulation_config=self._get_simulation_config(),
            distributed_environment=self._get_dist_environment(),
            device=device,
        )

    def sync_and_repartition(self) -> None:
        self.state.sync_and_repartition()
        # TODO: ADD appropriate barriers/synchronization if needed here

    def _set_state(self, state: SimState):
        self.state = state

    def step(self, step_size: float):
        self.integrator.step(self.state.local_state, self.model, step_size)
