from dynamis_state import (
    SimpleSpatialPartitioner,
    KDTreePartitioner,
    SFCPartitioner,
    SFCCurveType,
    DistEnvironment,
    MolecularInfo,
    SimulationConfig,
    TrajectoryReader,
    TrajectoryWriter,
    TorchDistEnv,
    Timer,
    TimingRegistry,
    TimerClass,
    Where,
    Level,
)
from dynamis_state.io import is_pathlike
from distnnp_model import DistNNPModel
from distnnp_model.comm import drift_flags
from distnnp.simulatorState import SimState, DRIFT_TRIP, DRIFT_BREACH
from dynamis_integrator import (
    Euler,
    Verlet,
    Langevin,
    DistNNPIntegrator,
    AVALIABLE_INTEGRATORS,
)
from typing import List, Union, Literal, Optional
from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
import uuid
import warnings

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
    enable_drift_watchdog: bool = (
        True  # Also repartition when any atom has drifted too far
    )
    drift_threshold_angstrom: Optional[float] = (
        None  # None -> ghost_buffer_angstrom / 2
    )

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
        enable_drift_watchdog (bool): Keep the halo maps valid between rebuilds. Ghost
            lists are built at cutoff + ghost_buffer; no edge can be missing while every
            atom has moved at most ghost_buffer / 2 since the build. Each step, on the
            positions the model is about to push, every rank checks its atoms and the
            result is OR-ed over ranks on the IPC push handshake (no extra collective;
            models without IPC pushes fall back to a 1-element all_reduce per step).
            The ghost lists are rebuilt before the next step if some atom could exceed
            the threshold during it (per-atom bound: displacement + dt * velocity
            terms): positions are all-gathered and the halo maps rebuilt, keeping the
            partition (rebalancing stays on rebalance_interval). A step whose graph was
            already built past the threshold is recorded in drift_breach_steps with a
            warning. Not used at W = 1, which has no ghosts.
        drift_threshold_angstrom (Optional[float]): Displacement guarantee in Ångstroms.
            None (default) = ghost_buffer_angstrom / 2, the largest value for which the
            guarantee implies a complete graph.
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

        run_uuid = uuid.uuid4()
        # shared by the timing registry and the trajectory writer (built in setup())
        self.run_uuid = run_uuid

        # Drift watchdog: flags live in the process-wide registry so the model's
        # exchangers carry them without the model knowing (see drift_flags.py).
        # W = 1 has no ghosts, so its graph is always complete: nothing to watch.
        self._drift_flags: Optional[drift_flags.DriftFlags] = (
            drift_flags.enable(device)
            if config.enable_drift_watchdog and dist_environment.world_size > 1
            else None
        )
        self._drift_mode: Optional[str] = None  # "local" | "piggyback" | "all_reduce"
        self._drift_threshold: float = (
            config.ghost_buffer_angstrom / 2
            if config.drift_threshold_angstrom is None
            else config.drift_threshold_angstrom
        )
        self._drift_dt: float = config.timestep_fs
        self._steps_done = 0
        self.drift_rebuild_steps: List[int] = (
            []
        )  # steps preceded by a halo-only rebuild
        self.drift_breach_steps: List[int] = (
            []
        )  # steps whose graph exceeded the guarantee
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        TimingRegistry.instance().init(
            dist_environment.global_rank,
            dist_environment.local_rank,
            dist_environment.world_size,
            run_uuid.bytes,
            device_id,
            config.output_dir,
        )
        Timer.set_level(Level.Basic)
        self.timer = Timer()

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
        self._reset_drift_flags()  # SimState latched fresh reference positions
        # Once the state is initialized, we can initialize the model buffers based on the local partition info

        self.model.initialize_buffers(self.state.partition_info, device=device)

    def setup(self, system_input: Union[str, MolecularInfo]) -> None:
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
            molecular_info, cell, pbc = TrajectoryReader.read(system_input)
        else:
            # in-memory MolecularInfo: no file, so no cell/pbc to read —
            # default to no box/no periodicity.
            molecular_info = system_input
            cell, pbc = torch.zeros(3, 3), torch.zeros(3, dtype=torch.uint8)

        # TODO figure out what author should be
        self.writer: TrajectoryWriter = TrajectoryWriter(
            dist_env.global_rank,
            self.run_uuid.bytes,
            "author",
            self.config.output_dir,
            self.config.traj_file,
            molecular_info,
            cell,
            pbc,
            self.config.timestep_fs,
            self.config.traj_interval,
        )

        global_mol_info = molecular_info

        # put tensors on device
        global_mol_info.atom_vertex_numbers = global_mol_info.atom_vertex_numbers.to(
            device=self.device
        )
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
        self.writer.update_atom_counts(
            self.state.local_partition_info.atom_counts.to(torch.uint32)
        )
        self.integrator.reset()
        self._reset_drift_flags()  # state.sync_and_repartition() re-latched the reference

    def rebuild_halo(self, device: Optional[torch.device] = None) -> None:
        """Refresh the ghost lists on the current positions, keeping the partition.
        Called when the drift watchdog trips; load balancing stays on the
        rebalance_interval schedule (sync_and_repartition). The local atoms, their
        order and velocities are unchanged, so the integrator's cached forces stay
        valid and are kept."""
        self.state.sync_positions_and_rebuild_halo()
        self.model.initialize_buffers(
            self.state.local_partition_info, device=device if device else self.device
        )
        self._reset_drift_flags()

    def _set_state(self, state: SimState):
        self.state = state

    def step(self, step_size: Optional[float] = None):
        if self._drift_flags is None:
            out = self.integrator.step(self.state._state, self.model, step_size)
        else:
            # Every integrator calls the model at exactly the positions the graph is
            # then built on (Verlet/Langevin after halfstep1, Euler before its update),
            # so the drift word is written there and rides on this step's own pushes.
            self._drift_dt = (
                step_size
                if step_size is not None
                else getattr(self.integrator, "dt", self.config.timestep_fs)
            )
            out = self.integrator.step(
                self.state._state, self._model_with_drift_flag, step_size
            )
        self._steps_done += 1
        return out

    def _model_with_drift_flag(self, dist_state, *args, **kwargs):
        self.update_drift_flag()
        return self.model(dist_state, *args, **kwargs)

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

    def _reset_drift_flags(self) -> None:
        if self._drift_flags is not None:
            self._drift_flags.local.zero_()
            self._drift_flags.global_.zero_()

    def update_drift_flag(self) -> None:
        """Write this rank's drift word from the current positions and velocities
        (device-only). step() calls it right before the model pushes; it is exposed for
        callers that drive the model themselves."""
        if self._drift_flags is not None:
            self.state.write_local_drift_flag(
                self._drift_flags.local, self._drift_threshold, self._drift_dt
            )

    def _global_drift_word(self) -> int:
        """OR over all ranks of the drift words written during the last step."""
        flags = self._drift_flags
        if self._drift_mode is None:
            self._drift_mode = self._select_drift_mode()
        if self._drift_mode == "local":
            return int(flags.local.item())
        if self._drift_mode == "all_reduce":
            # NCCL has no bitwise OR: MAX over one element per bit
            w = flags.local
            bits = torch.cat([w & DRIFT_TRIP, (w & DRIFT_BREACH) >> 1])
            dist.all_reduce(bits, op=dist.ReduceOp.MAX)
            b = bits.tolist()
            return b[0] * DRIFT_TRIP + b[1] * DRIFT_BREACH
        return int(flags.global_.item())

    def drift_watchdog_tripped(self) -> bool:
        """Call between steps, after at least one step. True on every rank, or on none,
        if the halo must be rebuilt before the next step: on the positions the last step
        pushed, some atom on some rank could pass the threshold during the next step.
        Also records the last step in drift_breach_steps if its graph was built with
        some atom already past the threshold (should never happen)."""
        if self._drift_flags is None:
            return False
        word = self._global_drift_word()
        if word & DRIFT_BREACH:
            step = self._steps_done - 1
            if not self.drift_breach_steps:
                warnings.warn(
                    f"drift watchdog: the graph of step {step} was built with an atom more "
                    f"than {self._drift_threshold:.3f} A from its halo-build position; edges "
                    "may be missing. Further breaches are recorded in drift_breach_steps.",
                    RuntimeWarning,
                )
            self.drift_breach_steps.append(step)
        return bool(word & DRIFT_TRIP)

    def _select_drift_mode(self) -> str:
        if self.dist_environment.world_size == 1:
            return "local"
        # Piggyback only if every rank's exchangers carried the flags on the steps so
        # far (a model without IPC pushes never wires them). Agreed on collectively,
        # once, so that no rank ends up in a different mode.
        wired = torch.tensor(
            [1 if self._drift_flags.n_wired > 0 else 0],
            dtype=torch.int32,
            device=self._drift_flags.local.device,
        )
        dist.all_reduce(wired, op=dist.ReduceOp.MIN)
        return "piggyback" if wired.item() else "all_reduce"

    def run(self) -> None:
        total_done = 0
        try:
            while total_done < self.config.total_steps:
                Timer.set_step(total_done)
                if self.should_rebalance(total_done):
                    self.timer.start("repartition", TimerClass.Other)
                    self.sync_and_repartition()
                    self.timer.stop()
                # Where.Both: step() launches async CUDA work, so a host-only timer misses it.
                self.timer.start("iteration", TimerClass.Total, Where.Both)
                tripped = total_done > 0 and self.drift_watchdog_tripped()
                if self.should_rebalance(total_done):
                    self.sync_and_repartition()
                elif tripped:
                    self.drift_rebuild_steps.append(total_done)
                    self.rebuild_halo()
                energy = self.step(self.config.timestep_fs)
                self.timer.stop()
                if self.should_write_traj(total_done):
                    self.state.local_state.atom_vertex_numbers = (
                        self.state.local_partition_info.partition_indices
                    )
                    self.writer.submit(total_done, self.state.local_state)
                total_done += 1
            self.writer.flush()
        finally:
            # Runs even on exception/interrupt so timing data isn't lost.
            TimingRegistry.instance().close()
