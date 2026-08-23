from distnnp_state import (
    SimpleSpatialPartitioner,
    KDTreePartitioner,
    SFCPartitioner,
    SFCCurveType,
    DistEnvironment,
    MolecularInfo,
    SimulationConfig,
    TrajectoryWriter,
    TorchDistEnv,
)
from distnnp_model import DistNNPModel
from distnnp.simulatorState import SimState
from distnnp_integrator import (
    Euler,
    Verlet,
    Langevin,
    DistNNPIntegrator,
    AVALIABLE_INTEGRATORS,
)
from typing import Union, Literal, Optional
from dataclasses import dataclass
import torch
import uuid

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
        device,
    ):
        self.model = model
        check_integrator(integrator)
        self.integrator: DistNNPIntegrator = integrator
        self.config = config
        self.dist_environment = dist_environment
        self.device = device
        # TODO figure out what author should be
        self.writer: TrajectoryWriter = TrajectoryWriter(
            dist_environment.global_rank,
            uuid.uuid4().bytes,
            "author",
            config.output_dir,
            config.traj_file,
        )

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
        if device is None:
            device = self.device
        self.state = SimState(
            molecular_system=molecular_system,
            partitioner=self._get_partitioner(),
            simulation_config=self._get_simulation_config(),
            distributed_environment=self._get_dist_environment(),
            device=device,
        )
        # Once the state is initialized, we can initialize the model buffers based on the local partition info

        self.model.initialize_buffers(
            self.state.partition_info, device=device
        )

    def setup(self, system_input: Union[str , MolecularInfo]) -> None:
        # (0) Initialize distributed if not already done.
        # Single-GPU / single-process runs skip init_process_group entirely —
        # torchrun sets RANK; SLURM sets SLURM_PROCID. If neither is present we
        # are running standalone and treat this as a world_size=1 job.
        torch_dist_env = TorchDistEnv()
        dist_env = torch_dist_env.dist_env

        sim_config = self._get_simulation_config()

        # (1) initialize from system
        if is_pathlike(system_input):
            assert os.path.exists(system_input), f"{system_input} could not be found"
        global_mol_info = self.writer.initialize(system_input, self.config.timestep_fs, self.config.traj_interval)

        # put tensors on device
        global_mol_info.atom_vertex_numbers = global_mol_info.atom_vertex_numbers.to(device=self.device)
        global_mol_info.atom_types = global_mol_info.atom_types.to(device=self.device)
        global_mol_info.atom_positions = global_mol_info.atom_positions.to(
            device=self.device
        )
        global_mol_info.atom_velocities = global_mol_info.atom_velocities.to(
            device=self.device
        )

        # (2) initialize simulator state from the global system and partition it.
        self.initialize_simulator_state(global_mol_info, device=self.device)

        # (3) Repartition if needed and update writer atom-count metadata.
        self.sync_and_repartition(device=self.device)

        # TODO if velocities are all 0, initialize from Maxwell-Boltzmann dist

    def sync_and_repartition(self, device: Optional[torch.device] = None) -> None:
        self.state.sync_and_repartition()
        # TODO: ADD appropriate barriers/synchronization if needed here
        self.model.initialize_buffers(
            self.state.local_partition_info, device=device if device else self.device
        )
        self.writer.update_atom_counts(self.state.local_partition_info.atom_counts.to(torch.uint32))
        self.integrator.reset()

    def _set_state(self, state: SimState):
        self.state = state

    def step(self, step_size: Optional[float] = None):
        return self.integrator.step(self.state._state, self.model, step_size)

    def should_write_traj(self, step: int) -> bool:
        if self.config.traj_interval == 0:
            return False
        if step % self.config.traj_interval == 0:
            return True
        return False

    def should_rebalance(self, step: int) -> bool:
        if self.config.rebalance_interval == 0:
            return False
        if step % self.config.rebalance_interval == 0:
            return True
        return False

    def run(self) -> None:
        total_done = 0
        while total_done < self.config.total_steps:
            if self.should_rebalance(total_done):
                self.sync_and_repartition()
            energy = self.step(self.config.timestep_fs)
            # if should_log(total_done):
                # _log_energy(total_done, energy)
            if self.should_write_traj(total_done):
                self.state.local_state.atom_vertex_numbers = self.state.local_partition_info.partition_indices
                self.writer.submit(total_done, self.state.local_state)
            total_done += 1
        self.writer.flush()
