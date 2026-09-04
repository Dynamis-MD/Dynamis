"""
Shared helpers for thin-adapter MACE trajectory validation.

This module carries over the model-agnostic orchestration and comparison logic
from the UMA trajectory checks. It supports both the distributed thin-adapter
producer and singleton thin or public-calculator reference models.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from distnnp import SimulatorConfig
from distnnp_model.nnp.MACE_Adaptation.distmace_thin_adapter import (
    DistMACEThinAdapter,
    load_raw_mace_model,
)
from distnnp_state import TrajectoryWriter as BoundTrajectoryWriter

try:
    from ase import Atoms
    from ase.build import bulk
    from ase.io import read
    from ase.io import write

    HAS_ASE = True
    _ASE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    Atoms = None  # type: ignore[assignment]
    bulk = None  # type: ignore[assignment]
    read = None  # type: ignore[assignment]
    write = None  # type: ignore[assignment]
    HAS_ASE = False
    _ASE_IMPORT_ERROR = exc

N_STEPS = 100
REBALANCE_INTERVAL = 7
SEED = 1234
SILICON_LATTICE_CONSTANT_ANGSTROM = 5.431
SILICON_SUPERCELL_REPETITIONS = (3, 3, 3)
SILICON_N_ATOMS = 8 * int(np.prod(SILICON_SUPERCELL_REPETITIONS))
SILICON_SIDE_LENGTH_ANGSTROM = (
    SILICON_LATTICE_CONSTANT_ANGSTROM * SILICON_SUPERCELL_REPETITIONS[0]
)
STRUCTURE_PERTURBATION_ANGSTROM = 0.01
HELIUM_N_ATOMS = 64
HELIUM_SIDE_LENGTH_ANGSTROM = 20.0
HELIUM_MIN_DISTANCE_ANGSTROM = 3.0
TEST_SYSTEMS = ("silicon", "helium")
_TEST_SYSTEM_SPECS = {
    "silicon": {
        "filename": "silicon_supercell.xyz",
        "symbol": "Si",
        "atomic_number": 14,
        "n_atoms": SILICON_N_ATOMS,
        "side_length_angstrom": SILICON_SIDE_LENGTH_ANGSTROM,
    },
    "helium": {
        "filename": "helium_box.xyz",
        "symbol": "He",
        "atomic_number": 2,
        "n_atoms": HELIUM_N_ATOMS,
        "side_length_angstrom": HELIUM_SIDE_LENGTH_ANGSTROM,
    },
}
DEFAULT_GHOST_CUTOFF_BUFFER = 0.5

# DistNNP's Verlet kernel uses mass in amu, positions in Angstrom, and time in
# fs, so its force input must be amu * Angstrom / fs^2. MACE returns the
# derivative of its eV energy with respect to Angstrom coordinates (eV / A).
# Keep energy in eV for diagnostics and scale only the force passed to MD.
EV_PER_ANGSTROM_TO_AMU_ANGSTROM_PER_FS2 = 0.0096485332

EARLY_PHASE_STEPS = 4
POS_VEL_TOL = 2e-5
ENERGY_RTOL = 1e-3

_RESOLUTION_FLOOR = 1e-6
_POS_NOISE_RMSD_FLOOR = 1e-6
_POS_NOISE_RMSD_SLOPE_PER_STEP = 5.6e-7
_VEL_NOISE_RMSD_FLOOR = _RESOLUTION_FLOOR
_VEL_NOISE_RMSD_SLOPE_PER_STEP = 3.7e-8
RMSD_MARGIN = 30.0

ENERGY_DRIFT_MARGIN = 3.0
ENERGY_SLOPE_FLOOR_EV_PER_STEP = 1e-3
ENERGY_STD_FLOOR_EV = 1e-2

RDF_LAST_K_FRAMES = 20
RDF_N_BINS = 50
RDF_R_MAX = 8.0
RDF_TOL = 0.3
DIAGNOSTIC_TARGET_VERTEX_ID = 71



def _is_pathlike(file_path: object) -> bool:
    return isinstance(file_path, (str, os.PathLike))


def _infer_atom_counts(partition_info, world_size: int, device: torch.device) -> torch.Tensor:
    atom_counts = getattr(partition_info, "atom_counts", None)
    if atom_counts is not None:
        return atom_counts.to(device=device, dtype=torch.long)

    partition_id = getattr(partition_info, "partition_id", None)
    if partition_id is not None:
        return torch.bincount(
            partition_id.to(device=device, dtype=torch.long),
            minlength=world_size,
        )

    partition_indices = getattr(partition_info, "partition_indices", None)
    if partition_indices is not None and world_size == 1:
        return torch.tensor([int(partition_indices.numel())], device=device, dtype=torch.long)

    raise AttributeError(
        "Could not infer per-rank atom counts: PartitionMessagePassingInfo exposes neither "
        "'atom_counts' nor 'partition_id'."
    )


## Why does this function define like three other functions and creates a programming file within it?>
## Break this one down!
def patch_simulator_trajectory_writer_compat() -> None:
    """Patch distnnp.simulator compatibility gaps for test-only execution.

    Keep the trajectory tests usable while the Python simulator compatibility
    fixes are landing across branches. Current native bindings receive atom
    counts as a ``torch.uint32`` tensor; do not convert that tensor to the
    Python-list contract used by older, stale extensions.

    The same simulator module also calls ``is_pathlike`` and ``os.path.exists``
    inside ``setup()`` without importing either symbol. Patch those missing
    names here with a local helper so the trajectory tests do not depend on
    importing ``distnnp_state.io`` on Python 3.9 environments.

    Some builds also surface non-integer ``atom_vertex_numbers`` /
    ``partition_indices`` tensors at the simulator boundary when loading XYZ /
    EXTXYZ input. The simulator-state path assumes integer indexing tensors, so
    this shim normalizes them to ``torch.long`` before any advanced indexing.
    """
    import distnnp.simulator as simulator_module
    import distnnp.simulatorState as simulator_state_module

    if not getattr(simulator_module, "_mace_test_init_dtype_patch", False):
        original_initialize_simulator_state = simulator_module.Simulator.initialize_simulator_state

        def _compat_initialize_simulator_state(self, molecular_system, device=None):
            target_device = self.device if device is None else device
            if hasattr(molecular_system, "atom_vertex_numbers") and molecular_system.atom_vertex_numbers is not None:
                molecular_system.atom_vertex_numbers = molecular_system.atom_vertex_numbers.to(
                    device=target_device, dtype=torch.long
                )
            return original_initialize_simulator_state(self, molecular_system, device=device)

        simulator_module.Simulator.initialize_simulator_state = _compat_initialize_simulator_state
        simulator_module._mace_test_init_dtype_patch = True

    if not getattr(simulator_module, "_mace_test_sync_patch", False):
        def _compat_sync_and_repartition(self, device=None):
            self.state.sync_and_repartition()
            target_device = device if device else self.device
            self.model.initialize_buffers(self.state.local_partition_info, device=target_device)

            counts = _infer_atom_counts(
                self.state.local_partition_info,
                world_size=self.dist_environment.world_size,
                device=target_device,
            )
            self.writer.update_atom_counts(counts.to(dtype=torch.uint32))

            reset = getattr(self.integrator, "reset", None)
            if callable(reset):
                reset()

        simulator_module.Simulator.sync_and_repartition = _compat_sync_and_repartition
        simulator_module._mace_test_sync_patch = True

    if not getattr(simulator_state_module, "_mace_test_partition_index_patch", False):
        original_setup_mass_and_forces = simulator_state_module.SimState.setup_mass_and_forces
        original_sync_and_repartition = simulator_state_module.SimState.sync_and_repartition

        def _compat_setup_mass_and_forces(self, partition_info, molecular_system):
            partition_info.partition_indices = partition_info.partition_indices.to(
                device=molecular_system.atom_positions.device, dtype=torch.long
            )
            if hasattr(molecular_system, "atom_vertex_numbers") and molecular_system.atom_vertex_numbers is not None:
                molecular_system.atom_vertex_numbers = molecular_system.atom_vertex_numbers.to(
                    device=molecular_system.atom_positions.device, dtype=torch.long
                )
            return original_setup_mass_and_forces(self, partition_info, molecular_system)

        def _compat_sync_and_repartition(self, paritioner=None, comm_backend="nccl"):
            if hasattr(self._state.global_molecular_info, "atom_vertex_numbers"):
                self._state.global_molecular_info.atom_vertex_numbers = (
                    self._state.global_molecular_info.atom_vertex_numbers.to(
                        device=self.device, dtype=torch.long
                    )
                )
            self._state.local_partition_info.partition_indices = (
                self._state.local_partition_info.partition_indices.to(
                    device=self.device, dtype=torch.long
                )
            )
            return original_sync_and_repartition(self, paritioner=paritioner, comm_backend=comm_backend)

        simulator_state_module.SimState.setup_mass_and_forces = _compat_setup_mass_and_forces
        simulator_state_module.SimState.sync_and_repartition = _compat_sync_and_repartition
        simulator_state_module._mace_test_partition_index_patch = True

    current = simulator_module.TrajectoryWriter
    if getattr(current, "__name__", "") == "_CompatTrajectoryWriter":
        simulator_module.is_pathlike = _is_pathlike
        simulator_module.os = os
        return

    class _CompatTrajectoryWriter:
        def __new__(
            cls,
            global_rank: int,
            run_id: bytes,
            author: str,
            output_dir: str = "./trajectory",
            output_file: str = "trajectory.distnnp",
        ):
            try:
                return BoundTrajectoryWriter(global_rank, run_id, author, output_dir, output_file)
            except TypeError as exc:
                message = str(exc)
                if "incompatible constructor arguments" not in message:
                    raise
                return BoundTrajectoryWriter(global_rank, run_id, author, output_dir)

    simulator_module.TrajectoryWriter = _CompatTrajectoryWriter
    simulator_module.is_pathlike = _is_pathlike
    simulator_module.os = os


def _require_ase() -> None:
    if not HAS_ASE:
        raise ImportError("Periodic MACE trajectory tests require ASE.") from _ASE_IMPORT_ERROR


def _pos_rmsd_tolerance(step: int) -> float:
    noise = _POS_NOISE_RMSD_FLOOR + max(0, step - 5) * _POS_NOISE_RMSD_SLOPE_PER_STEP
    return RMSD_MARGIN * noise


def _vel_rmsd_tolerance(step: int) -> float:
    noise = _VEL_NOISE_RMSD_FLOOR + max(0, step - 5) * _VEL_NOISE_RMSD_SLOPE_PER_STEP
    return RMSD_MARGIN * noise


def get_test_system_spec(system: str) -> Dict[str, Any]:
    try:
        return _TEST_SYSTEM_SPECS[system]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trajectory test system {system!r}; choose one of {TEST_SYSTEMS}."
        ) from exc


def _generate_periodic_helium_positions(seed: int) -> np.ndarray:
    """Reproduce the original sparse periodic 64-He trajectory fixture."""
    rng = np.random.default_rng(seed)
    positions = np.empty((HELIUM_N_ATOMS, 3))
    for atom_index in range(HELIUM_N_ATOMS):
        for _ in range(200):
            candidate = rng.uniform(0.0, HELIUM_SIDE_LENGTH_ANGSTROM, size=3)
            if atom_index == 0:
                positions[atom_index] = candidate
                break
            displacements = positions[:atom_index] - candidate
            displacements -= HELIUM_SIDE_LENGTH_ANGSTROM * np.round(
                displacements / HELIUM_SIDE_LENGTH_ANGSTROM
            )
            distances = np.sqrt(np.sum(displacements**2, axis=-1))
            if np.all(distances >= HELIUM_MIN_DISTANCE_ANGSTROM):
                positions[atom_index] = candidate
                break
        else:
            raise RuntimeError(
                f"Could not place helium atom {atom_index}/{HELIUM_N_ATOMS} with "
                f"minimum distance {HELIUM_MIN_DISTANCE_ANGSTROM} Angstrom."
            )
    return positions


def ensure_test_structure(
    path: str | Path,
    seed: int = SEED,
    system: str = "silicon",
) -> None:
    _require_ase()
    spec = get_test_system_spec(system)
    path = str(path)
    if os.path.exists(path):
        atoms = read(path)
        actual_numbers = atoms.get_atomic_numbers()
        expected_cell = np.full(3, spec["side_length_angstrom"])
        if (
            len(atoms) != spec["n_atoms"]
            or not np.all(actual_numbers == spec["atomic_number"])
            or not bool(np.all(atoms.pbc))
            or not np.allclose(atoms.cell.lengths(), expected_cell, atol=1e-6, rtol=0)
        ):
            raise ValueError(
                f"Existing trajectory structure at {path} is not the required {system} "
                "fixture. Use a clean --path directory."
            )
        return

    if system == "silicon":
        atoms = bulk("Si", "diamond", a=SILICON_LATTICE_CONSTANT_ANGSTROM, cubic=True)
        atoms = atoms.repeat(SILICON_SUPERCELL_REPETITIONS)
        atoms.rattle(stdev=STRUCTURE_PERTURBATION_ANGSTROM, seed=seed)
        description = (
            f"periodic {SILICON_N_ATOMS}-atom diamond-Si supercell with "
            f"{STRUCTURE_PERTURBATION_ANGSTROM:.3f} A perturbations"
        )
    else:
        atoms = Atoms(
            ["He"] * HELIUM_N_ATOMS,
            positions=_generate_periodic_helium_positions(seed),
            cell=[HELIUM_SIDE_LENGTH_ANGSTROM] * 3,
            pbc=True,
        )
        description = (
            f"periodic {HELIUM_N_ATOMS}-atom helium box with minimum initial "
            f"separation {HELIUM_MIN_DISTANCE_ANGSTROM:.1f} A"
        )
    atoms.wrap()
    assert len(atoms) == spec["n_atoms"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write(path, atoms, format="extxyz")
    print(f"Wrote {description} to {path}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def real_singleton_torch_dist_env():
    from distnnp_state import TorchDistEnv

    port = free_port()
    return TorchDistEnv(
        is_distributed=True,
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def teardown_singleton_dist_env() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


class MACEForceUnitConversionAdapter:
    """Expose a thin MACE adapter with forces in DistNNP integration units.

    This is deliberately trajectory-test-only: static MACE parity tests compare
    raw MACE eV/Angstrom forces. The potential energy remains in eV; only the
    force handed to the amu/Angstrom/fs Verlet integrator is converted.
    """

    def __init__(self, adapter: DistMACEThinAdapter) -> None:
        self._adapter = adapter
        self._force_diagnostic_limit = 0
        self.force_diagnostics: list[Dict[str, Any]] = []

    @property
    def cutoff(self) -> float:
        return self._adapter.cutoff

    def initialize_buffers(self, partition_info, device: torch.device) -> None:
        self._adapter.initialize_buffers(partition_info, device=device)

    def enable_force_diagnostics(self, max_calls: int = 3) -> None:
        """Capture preflight plus the first two Verlet force evaluations."""
        if max_calls <= 0:
            raise ValueError("max_calls must be positive when force diagnostics are enabled.")
        self._force_diagnostic_limit = max_calls
        self.force_diagnostics = []

    def _graph_diagnostics(self, state) -> Dict[str, np.ndarray | int]:
        """Build test-only edge metadata for the same unchanged local state."""
        view = self._adapter._build_structural_view(state)
        graph = self._adapter._build_owned_graph(view, state)
        destinations = graph.edge_index[1]
        sources = graph.edge_index[0]
        neighbor_count = torch.bincount(destinations, minlength=view.num_local)
        halo_neighbor_count = torch.bincount(
            destinations[sources >= view.num_local], minlength=view.num_local
        )
        return {
            "neighbor_count": neighbor_count.detach().cpu().numpy(),
            "halo_neighbor_count": halo_neighbor_count.detach().cpu().numpy(),
            "halo_atom_count": view.num_total - view.num_local,
        }

    def __call__(self, state, *args: Any, **kwargs: Any):
        energy_ev, forces_ev_per_angstrom = self._adapter(state, *args, **kwargs)
        forces_amu_angstrom_per_fs2 = (
            forces_ev_per_angstrom * EV_PER_ANGSTROM_TO_AMU_ANGSTROM_PER_FS2
        )

        if len(self.force_diagnostics) < self._force_diagnostic_limit:
            graph_diagnostics = self._graph_diagnostics(state)
            partition_info = state.local_partition_info
            num_local = state.local_molecular_info.num_atoms
            self.force_diagnostics.append(
                {
                    "call_index": len(self.force_diagnostics),
                    "energy_ev": float(energy_ev.detach().cpu()),
                    "vid": partition_info.partition_indices[:num_local].detach().cpu().numpy(),
                    "raw_force_ev_per_angstrom": forces_ev_per_angstrom.detach().cpu().numpy(),
                    "force_amu_angstrom_per_fs2": forces_amu_angstrom_per_fs2.detach().cpu().numpy(),
                    **graph_diagnostics,
                }
            )

        return energy_ev, forces_amu_angstrom_per_fs2


def build_thin_mace_model(
    device: torch.device,
    model: str = "medium-mpa-0",
    head_name: Optional[str] = None,
    model_family: str = "mp",
    system: str = "silicon",
) -> MACEForceUnitConversionAdapter:
    spec = get_test_system_spec(system)
    raw_model = load_raw_mace_model(model_family=model_family, model=model, device=device.type)
    supported_atomic_numbers = {int(value) for value in raw_model.atomic_numbers.detach().cpu().tolist()}
    if spec["atomic_number"] not in supported_atomic_numbers:
        raise ValueError(
            f"Selected MACE model_family={model_family!r}, model={model!r} does not support "
            f"{spec['symbol']} (atomic number {spec['atomic_number']}). Supported atomic numbers: "
            f"{sorted(supported_atomic_numbers)}"
        )
    adapter = DistMACEThinAdapter(raw_model.to(device=device).eval(), head_name=head_name or None)
    return MACEForceUnitConversionAdapter(adapter)


class MonolithicMACECalculatorAdapter:
    """Expose the public MACE ASE calculator through DistNNP's model protocol.

    This reference path is intentionally restricted to a singleton simulation:
    ASE receives the complete system and performs MACE-native graph construction,
    independently of the thin adapter's distributed graph and feature exchange.
    """

    def __init__(self, calculator, cutoff: float, device: torch.device) -> None:
        self._calculator = calculator
        self._cutoff = float(cutoff)
        self._device = device

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def initialize_buffers(self, partition_info, device: torch.device) -> None:
        # The monolithic calculator owns no DistNNP communication buffers.
        if device != self._device:
            raise ValueError(
                f"Monolithic MACE calculator was created on {self._device}, "
                f"but the simulator requested {device}."
            )

    def __call__(self, state, *args: Any, **kwargs: Any):
        _require_ase()
        from ase.calculators.calculator import all_changes

        if int(state.dist_env.world_size) != 1:
            raise ValueError(
                "Monolithic MACE trajectory references require a world-size=1 simulator."
            )
        molecular_info = state.local_molecular_info
        positions = molecular_info.atom_positions
        num_atoms = int(molecular_info.num_atoms)
        if positions.shape[0] != num_atoms:
            raise ValueError(
                "Monolithic MACE reference received inconsistent local atom metadata."
            )

        atoms = Atoms(
            numbers=molecular_info.atom_types.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy(),
            cell=state.sim_config.cell.detach().cpu().numpy(),
            pbc=state.sim_config.pbc.detach().cpu().numpy(),
        )
        self._calculator.calculate(
            atoms,
            properties=["energy", "forces"],
            system_changes=all_changes,
        )

        energy_ev = torch.as_tensor(
            self._calculator.results["energy"],
            dtype=positions.dtype,
            device=positions.device,
        )
        forces_ev_per_angstrom = torch.as_tensor(
            self._calculator.results["forces"],
            dtype=positions.dtype,
            device=positions.device,
        )
        if forces_ev_per_angstrom.shape != positions.shape:
            raise ValueError(
                "Public MACE calculator returned forces with shape "
                f"{tuple(forces_ev_per_angstrom.shape)}; expected {tuple(positions.shape)}."
            )

        # Public MACE forces are eV/Angstrom; Verlet expects amu*Angstrom/fs^2.
        converted_forces = (
            forces_ev_per_angstrom * EV_PER_ANGSTROM_TO_AMU_ANGSTROM_PER_FS2
        )
        return energy_ev, converted_forces


def build_monolithic_mace_model(
    device: torch.device,
    model: str = "medium-mpa-0",
    head_name: Optional[str] = None,
    model_family: str = "mp",
    system: str = "silicon",
) -> MonolithicMACECalculatorAdapter:
    """Build an independent public-calculator reference for trajectory tests."""
    spec = get_test_system_spec(system)
    if head_name:
        raise ValueError(
            "The monolithic trajectory reference does not yet support --head-name; "
            "use the selected foundation model's default head."
        )

    try:
        from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Monolithic MACE trajectory references require the public mace.calculators API."
        ) from exc

    loaders = {"mp": mace_mp, "off": mace_off, "omol": mace_omol, "polar": mace_polar}
    calculator = loaders[model_family](
        model=model,
        device=str(device),
        default_dtype="float32",
    )

    calculator_models = getattr(calculator, "models", None)
    if calculator_models is None or len(calculator_models) == 0:
        raise AttributeError("Public MACE calculator does not expose its loaded model list.")
    calculator_model = calculator_models[0]
    cutoff_value = getattr(calculator_model, "r_max", None)
    if cutoff_value is None:
        raise AttributeError("Public MACE calculator model does not expose r_max.")
    cutoff = float(torch.as_tensor(cutoff_value).detach().cpu())

    supported_atomic_numbers = {
        int(value)
        for value in calculator_model.atomic_numbers.detach().cpu().tolist()
    }
    if spec["atomic_number"] not in supported_atomic_numbers:
        raise ValueError(
            f"Selected MACE model_family={model_family!r}, model={model!r} does not support "
            f"{spec['symbol']} (atomic number {spec['atomic_number']}). Supported atomic numbers: "
            f"{sorted(supported_atomic_numbers)}"
        )
    return MonolithicMACECalculatorAdapter(calculator, cutoff=cutoff, device=device)


def make_config(
    output_dir: str,
    cutoff_angstrom: float,
    timestep_fs: float = 0.5,
    rebalance_interval: int = REBALANCE_INTERVAL,
    side_length_angstrom: float = SILICON_SIDE_LENGTH_ANGSTROM,
) -> SimulatorConfig:
    os.makedirs(output_dir, exist_ok=True)
    return SimulatorConfig(
        timestep_fs=timestep_fs,
        total_steps=N_STEPS,
        side_length_angstrom=side_length_angstrom,
        cutoff_angstrom=float(cutoff_angstrom),
        ghost_buffer_angstrom=DEFAULT_GHOST_CUTOFF_BUFFER,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=rebalance_interval,
        traj_interval=0,
        log_interval=1,
        output_dir=output_dir,
        backend="nccl",
    )


def extract_frame(sim, energy) -> Dict[str, Any]:
    partition_info = sim.state.local_partition_info
    mol_info = sim.state.local_state
    vid = partition_info.partition_indices[: mol_info.num_atoms].detach().cpu().numpy()
    order = np.argsort(vid)
    frame = {
        "energy": float(energy),
        "pos": mol_info.atom_positions.detach().cpu().numpy()[order],
        "vel": mol_info.atom_velocities.detach().cpu().numpy()[order],
        "vid": vid[order],
    }
    return frame

# key orchestration point
def run_md_loop(
    sim,
    config: SimulatorConfig,
    steps: int = N_STEPS,
    skip_initial_rebalance: bool = False,
):
    if steps <= 0:
        raise ValueError("Trajectory step count must be positive.")
    frames = []
    for step in range(steps):
        if sim.should_rebalance(step) and not (skip_initial_rebalance and step == 0):
            sim.sync_and_repartition()
        energy, _ = sim.step(config.timestep_fs)
        frames.append(extract_frame(sim, energy))
    return frames


def merge_force_diagnostics(rank_records: list[list[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    """Merge rank-owned diagnostic records into global vertex-ID order."""
    if not rank_records:
        return []
    record_count = len(rank_records[0])
    if any(len(records) != record_count for records in rank_records):
        raise ValueError("Ranks captured a different number of force diagnostic records.")

    merged = []
    array_keys = (
        "vid",
        "raw_force_ev_per_angstrom",
        "force_amu_angstrom_per_fs2",
        "neighbor_count",
        "halo_neighbor_count",
    )
    for call_index in range(record_count):
        records = [rank[call_index] for rank in rank_records]
        combined = {key: np.concatenate([record[key] for record in records]) for key in array_keys}
        order = np.argsort(combined["vid"])
        vids = combined["vid"][order]
        if np.unique(vids).size != vids.size:
            raise ValueError(f"Diagnostic call {call_index} contains duplicate owned vertex IDs.")
        merged.append(
            {
                "call_index": call_index,
                "energy_ev": float(sum(float(record["energy_ev"]) for record in records)),
                "vid": vids,
                "raw_force_ev_per_angstrom": combined["raw_force_ev_per_angstrom"][order],
                "force_amu_angstrom_per_fs2": combined["force_amu_angstrom_per_fs2"][order],
                "neighbor_count": combined["neighbor_count"][order],
                "halo_neighbor_count": combined["halo_neighbor_count"][order],
                "halo_atom_count_by_rank": [int(record["halo_atom_count"]) for record in records],
            }
        )
    return merged


def write_force_diagnostic_report(
    reference_records: list[Dict[str, Any]],
    distributed_records: list[Dict[str, Any]],
    path: str | Path,
) -> None:
    """Write preflight and first-Verlet-call force comparisons without asserting."""
    import csv

    if len(reference_records) != len(distributed_records):
        raise ValueError("Reference and distributed force diagnostic record counts differ.")

    rows = []
    for reference, distributed in zip(reference_records, distributed_records):
        if not np.array_equal(reference["vid"], distributed["vid"]):
            raise ValueError(f"Force diagnostic call {reference['call_index']} has mismatched vertex IDs.")
        raw_diff = distributed["raw_force_ev_per_angstrom"] - reference["raw_force_ev_per_angstrom"]
        converted_diff = (
            distributed["force_amu_angstrom_per_fs2"]
            - reference["force_amu_angstrom_per_fs2"]
        )
        target = np.flatnonzero(reference["vid"] == DIAGNOSTIC_TARGET_VERTEX_ID)
        target_index = int(target[0]) if target.size else None
        rows.append(
            {
                "call_index": int(reference["call_index"]),
                "energy_diff_ev": float(distributed["energy_ev"] - reference["energy_ev"]),
                "raw_force_max_abs_diff_ev_per_angstrom": float(np.max(np.abs(raw_diff))),
                "converted_force_max_abs_diff_amu_angstrom_per_fs2": float(
                    np.max(np.abs(converted_diff))
                ),
                "target_vid": DIAGNOSTIC_TARGET_VERTEX_ID if target_index is not None else "missing",
                "target_raw_force_max_abs_diff_ev_per_angstrom": (
                    float(np.max(np.abs(raw_diff[target_index]))) if target_index is not None else ""
                ),
                "target_neighbor_count_reference": (
                    int(reference["neighbor_count"][target_index]) if target_index is not None else ""
                ),
                "target_neighbor_count_distributed": (
                    int(distributed["neighbor_count"][target_index]) if target_index is not None else ""
                ),
                "target_halo_neighbor_count_distributed": (
                    int(distributed["halo_neighbor_count"][target_index]) if target_index is not None else ""
                ),
                "distributed_halo_atom_count_by_rank": ";".join(
                    str(value) for value in distributed["halo_atom_count_by_rank"]
                ),
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} force diagnostic rows to {path}")


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def rmsd_between_trajectories(traj_a, traj_b) -> Dict[str, list[float]]:
    assert len(traj_a) == len(traj_b)
    return {
        "pos": [rmsd(a["pos"], b["pos"]) for a, b in zip(traj_a, traj_b)],
        "vel": [rmsd(a["vel"], b["vel"]) for a, b in zip(traj_a, traj_b)],
    }


def compute_energy_drift(traj) -> np.ndarray:
    e0 = traj[0]["energy"]
    return np.array([abs(frame["energy"] - e0) for frame in traj])


def energy_drift_slope(traj) -> float:
    energies = np.array([frame["energy"] for frame in traj])
    steps = np.arange(len(energies))
    slope, _ = np.polyfit(steps, energies, 1)
    return float(slope)


def energy_oscillation_std(traj) -> float:
    energies = np.array([frame["energy"] for frame in traj])
    steps = np.arange(len(energies))
    slope, intercept = np.polyfit(steps, energies, 1)
    residual = energies - (slope * steps + intercept)
    return float(np.std(residual))


def compute_rdf(positions: np.ndarray, cell_length: float, r_max: float = RDF_R_MAX, n_bins: int = RDF_N_BINS):
    n = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= cell_length * np.round(diffs / cell_length)
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_dists = pair_dists[pair_dists < r_max]

    hist, edges = np.histogram(pair_dists, bins=n_bins, range=(0, r_max))
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    shell_vol = 4 * np.pi * r**2 * dr
    density = n / (cell_length**3)
    n_pairs_ideal = np.maximum(0.5 * n * density * shell_vol, 1e-12)
    g_r = hist / n_pairs_ideal
    return r, g_r


def averaged_rdf(traj, last_k: int, cell_length: float):
    frames = traj[-last_k:]
    g_accum = None
    r = None
    for frame in frames:
        r, g = compute_rdf(frame["pos"], cell_length)
        g_accum = g if g_accum is None else g_accum + g
    return r, g_accum / len(frames)


def build_comparison_report(ref_traj, dist_traj) -> list[Dict[str, Any]]:
    rows = []
    for step in range(len(ref_traj)):
        ref_frame = ref_traj[step]
        dist_frame = dist_traj[step]
        pos_diff = dist_frame["pos"] - ref_frame["pos"]
        vel_diff = dist_frame["vel"] - ref_frame["vel"]

        pos_per_atom = np.max(np.abs(pos_diff), axis=-1)
        vel_per_atom = np.max(np.abs(vel_diff), axis=-1)
        pos_max_idx = int(np.argmax(pos_per_atom))
        vel_max_idx = int(np.argmax(vel_per_atom))

        vid_match = bool(np.array_equal(ref_frame["vid"], dist_frame["vid"]))
        row = {
            "step": step,
            "is_rebalance_step": step % REBALANCE_INTERVAL == 0,
            "vid_match": vid_match,
            "ref_energy": ref_frame["energy"],
            "dist_energy": dist_frame["energy"],
            "energy_diff": dist_frame["energy"] - ref_frame["energy"],
            "pos_rmsd": rmsd(dist_frame["pos"], ref_frame["pos"]),
            "vel_rmsd": rmsd(dist_frame["vel"], ref_frame["vel"]),
            "pos_max_abs": float(pos_per_atom[pos_max_idx]),
            "pos_max_atom_vid": int(ref_frame["vid"][pos_max_idx]),
            "vel_max_abs": float(vel_per_atom[vel_max_idx]),
            "vel_max_atom_vid": int(ref_frame["vid"][vel_max_idx]),
        }
        rows.append(row)
    return rows


def write_comparison_report_csv(rows: list[Dict[str, Any]], path: str) -> None:
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote full {len(rows)}-step comparison report to {path}")


def print_comparison_report_summary(rows: list[Dict[str, Any]]) -> None:
    """Print the highest-signal diagnostics persisted in comparison_report.csv."""
    if not rows:
        return

    first = rows[0]
    last = rows[-1]
    peak_energy = max(rows, key=lambda row: abs(float(row["energy_diff"])))
    peak_position = max(rows, key=lambda row: float(row["pos_rmsd"]))
    peak_velocity = max(rows, key=lambda row: float(row["vel_rmsd"]))
    rebalance_rows = [row for row in rows if row["is_rebalance_step"]]
    steady_rows = [row for row in rows if not row["is_rebalance_step"]]
    rebalance_pos_peak = max(float(row["pos_rmsd"]) for row in rebalance_rows)
    steady_pos_peak = max(float(row["pos_rmsd"]) for row in steady_rows)
    vertex_ids_match = all(bool(row["vid_match"]) for row in rows)

    print("[report] trajectory comparison summary:")
    print(
        f"[report] vertex IDs match every frame={vertex_ids_match}; "
        f"step 0: dE={float(first['energy_diff']):+.3e} eV, "
        f"pos_rmsd={float(first['pos_rmsd']):.3e} A, "
        f"vel_rmsd={float(first['vel_rmsd']):.3e} A/fs"
    )
    print(
        f"[report] final step {last['step']}: dE={float(last['energy_diff']):+.3e} eV, "
        f"pos_rmsd={float(last['pos_rmsd']):.3e} A, "
        f"vel_rmsd={float(last['vel_rmsd']):.3e} A/fs"
    )
    print(
        f"[report] peak |dE|={abs(float(peak_energy['energy_diff'])):.3e} eV at step "
        f"{peak_energy['step']}; peak position RMSD={float(peak_position['pos_rmsd']):.3e} A "
        f"at step {peak_position['step']} (max component atom vid={peak_position['pos_max_atom_vid']}); "
        f"peak velocity RMSD={float(peak_velocity['vel_rmsd']):.3e} A/fs at step "
        f"{peak_velocity['step']} (max component atom vid={peak_velocity['vel_max_atom_vid']})"
    )
    print(
        f"[report] position RMSD peak: rebalance frames={rebalance_pos_peak:.3e} A, "
        f"non-rebalance frames={steady_pos_peak:.3e} A"
    )


def compare_trajectories(
    ref_traj,
    dist_traj,
    report_path: Optional[str] = None,
    side_length_angstrom: float = SILICON_SIDE_LENGTH_ANGSTROM,
) -> None:
    rows = build_comparison_report(ref_traj, dist_traj)
    if report_path is not None:
        write_comparison_report_csv(rows, report_path)
    print_comparison_report_summary(rows)

    assert len(dist_traj) == len(ref_traj) == N_STEPS, (
        f"frame count mismatch: ref={len(ref_traj)} dist={len(dist_traj)} expected={N_STEPS}"
    )

    for step, (ref_frame, dist_frame) in enumerate(zip(ref_traj, dist_traj)):
        assert np.array_equal(ref_frame["vid"], dist_frame["vid"]), (
            f"step {step}: vertex ID sets diverged between single-rank and distributed runs "
            "(indicates a bug in redistribute_atoms / vertex ID tracking, not numerics)"
        )

    for step in range(0, EARLY_PHASE_STEPS + 1):
        ref_frame = ref_traj[step]
        dist_frame = dist_traj[step]

        np.testing.assert_allclose(
            dist_frame["pos"],
            ref_frame["pos"],
            atol=POS_VEL_TOL,
            rtol=0,
            err_msg=f"[phase 1] positions diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["vel"],
            ref_frame["vel"],
            atol=POS_VEL_TOL,
            rtol=0,
            err_msg=f"[phase 1] velocities diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["energy"],
            ref_frame["energy"],
            rtol=ENERGY_RTOL,
            err_msg=f"[phase 1] total energy diverged beyond {ENERGY_RTOL} relative at step {step}",
        )

    diffs = rmsd_between_trajectories(ref_traj, dist_traj)
    for step in range(EARLY_PHASE_STEPS + 1, N_STEPS):
        pos_tol = _pos_rmsd_tolerance(step)
        vel_tol = _vel_rmsd_tolerance(step)
        assert diffs["pos"][step] <= pos_tol, (
            f"[phase 2] position RMSD-of-difference {diffs['pos'][step]:.6f} exceeds "
            f"{pos_tol:.6f} at step {step}"
        )
        assert diffs["vel"][step] <= vel_tol, (
            f"[phase 2] velocity RMSD-of-difference {diffs['vel'][step]:.6f} exceeds "
            f"{vel_tol:.6f} at step {step}"
        )

    ref_endpoint_drift = compute_energy_drift(ref_traj)
    dist_endpoint_drift = compute_energy_drift(dist_traj)
    print(
        f"[phase 2] energy |E(t)-E(0)| (diagnostic only): "
        f"reference max={float(np.max(ref_endpoint_drift)):.6f} eV, "
        f"distributed max={float(np.max(dist_endpoint_drift)):.6f} eV"
    )

    ref_slope = energy_drift_slope(ref_traj)
    dist_slope = energy_drift_slope(dist_traj)
    ref_std = energy_oscillation_std(ref_traj)
    dist_std = energy_oscillation_std(dist_traj)
    slope_bound = max(ENERGY_DRIFT_MARGIN * abs(ref_slope), ENERGY_SLOPE_FLOOR_EV_PER_STEP)
    std_bound = max(ENERGY_DRIFT_MARGIN * ref_std, ENERGY_STD_FLOOR_EV)
    print(
        f"[phase 2] energy drift slope: reference={ref_slope:.3e} eV/step, "
        f"distributed={dist_slope:.3e} eV/step, bound={slope_bound:.3e} eV/step"
    )
    print(
        f"[phase 2] energy oscillation std (about trend): reference={ref_std:.3e} eV, "
        f"distributed={dist_std:.3e} eV, bound={std_bound:.3e} eV"
    )
    assert abs(dist_slope) <= slope_bound, (
        f"[phase 2] distributed run's energy drift slope ({dist_slope:.3e} eV/step) exceeds "
        f"{ENERGY_DRIFT_MARGIN}x the reference run's own slope ({ref_slope:.3e} eV/step, "
        f"bound={slope_bound:.3e} eV/step)"
    )
    assert dist_std <= std_bound, (
        f"[phase 2] distributed run's energy oscillation std ({dist_std:.3e} eV) exceeds "
        f"{ENERGY_DRIFT_MARGIN}x the reference run's own std ({ref_std:.3e} eV, "
        f"bound={std_bound:.3e} eV)"
    )

    _, g_ref = averaged_rdf(ref_traj, RDF_LAST_K_FRAMES, side_length_angstrom)
    _, g_dist = averaged_rdf(dist_traj, RDF_LAST_K_FRAMES, side_length_angstrom)
    rdf_diff = float(np.mean(np.abs(g_ref - g_dist)))
    print(f"[phase 2] RDF mean abs difference (last {RDF_LAST_K_FRAMES} frames): {rdf_diff:.4f}")
    assert rdf_diff <= RDF_TOL, (
        f"[phase 2] RDF mean abs difference {rdf_diff:.4f} exceeds loose sanity threshold {RDF_TOL}"
    )


def reference_vs_reference_check(
    structure_path: str,
    output_dir_a: str,
    output_dir_b: str,
    model: str = "medium-mpa-0",
    head_name: Optional[str] = None,
    model_family: str = "mp",
):
    from distnnp import Simulator
    from distnnp_integrator import Verlet

    ensure_test_structure(structure_path)

    def _one_run(output_dir: str):
        env = real_singleton_torch_dist_env()
        try:
            dist_env = env.dist_env
            adapter = build_thin_mace_model(
                device=env.device,
                model=model,
                head_name=head_name,
                model_family=model_family,
            )
            integrator = Verlet(dt=0.5)
            config = make_config(output_dir, adapter.cutoff)
            sim = Simulator(adapter, integrator, dist_env, config, env.device)
            sim.setup(structure_path)
            return run_md_loop(sim, config)
        finally:
            teardown_singleton_dist_env()

    traj_a = _one_run(output_dir_a)
    traj_b = _one_run(output_dir_b)
    diffs = rmsd_between_trajectories(traj_a, traj_b)

    print(f"{'step':>4}  {'pos_rmsd':>12}  {'pos_tol':>12}  {'vel_rmsd':>12}  {'vel_tol':>12}")
    for step in range(N_STEPS):
        if step <= EARLY_PHASE_STEPS:
            pos_tol = POS_VEL_TOL
            vel_tol = POS_VEL_TOL
        else:
            pos_tol = _pos_rmsd_tolerance(step)
            vel_tol = _vel_rmsd_tolerance(step)
        flag = (
            "  <-- EXCEEDS TOLERANCE"
            if diffs["pos"][step] > pos_tol or diffs["vel"][step] > vel_tol
            else ""
        )
        print(
            f"{step:>4}  {diffs['pos'][step]:>12.6f}  {pos_tol:>12.6f}  "
            f"{diffs['vel'][step]:>12.6f}  {vel_tol:>12.6f}{flag}"
        )
    return diffs
