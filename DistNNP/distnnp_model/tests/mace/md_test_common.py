"""
Shared helpers for thin-adapter MACE trajectory validation.

This module carries over the model-agnostic orchestration and comparison logic
from the UMA trajectory checks while replacing UMA-specific model setup with a
thin DistMACE adapter factory.
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
    from ase.io import write

    HAS_ASE = True
    _ASE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    Atoms = None  # type: ignore[assignment]
    write = None  # type: ignore[assignment]
    HAS_ASE = False
    _ASE_IMPORT_ERROR = exc

N_STEPS = 100
REBALANCE_INTERVAL = 7
SEED = 1234
N_ATOMS = 64
SIDE_LENGTH_ANGSTROM = 20.0
ELEMENT_SYMBOL = "He"
ELEMENT_ATOMIC_NUMBER = 2
MIN_ATOM_DISTANCE_ANGSTROM = 3.0
DEFAULT_GHOST_CUTOFF_BUFFER = 0.5

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

RANK_ENERGY_ATOL = 1e-6
RANK_ENERGY_RTOL = 1e-6


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


def patch_simulator_trajectory_writer_compat() -> None:
    """Patch distnnp.simulator compatibility gaps for test-only execution.

    Some environments still expose a four-argument pybind constructor
    ``TrajectoryWriter(rank, run_id, author, output_dir)`` even though the
    local Python Simulator passes a fifth ``traj_file`` argument. The trajectory
    tests only need the simulator to construct a writer successfully, so this
    shim retries with the older constructor when needed.

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
            self.writer.update_atom_counts([int(value) for value in counts.detach().cpu().tolist()])

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


def _generate_positions(
    n_atoms: int,
    cell_length: float,
    min_distance: float,
    seed: int,
    max_attempts_per_atom: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.empty((n_atoms, 3))
    for i in range(n_atoms):
        placed = False
        for _ in range(max_attempts_per_atom):
            candidate = rng.uniform(0.0, cell_length, size=3)
            if i == 0:
                positions[i] = candidate
                placed = True
                break
            diffs = positions[:i] - candidate
            diffs -= cell_length * np.round(diffs / cell_length)
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            if np.all(dists >= min_distance):
                positions[i] = candidate
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place atom {i}/{n_atoms} with min_distance={min_distance} "
                f"after {max_attempts_per_atom} attempts each"
            )
    return positions


def ensure_test_structure(
    path: str | Path,
    n_atoms: int = N_ATOMS,
    seed: int = SEED,
    min_distance: float = MIN_ATOM_DISTANCE_ANGSTROM,
) -> None:
    _require_ase()
    path = str(path)
    if os.path.exists(path):
        return
    positions = _generate_positions(n_atoms, SIDE_LENGTH_ANGSTROM, min_distance, seed)
    atoms = Atoms(
        ELEMENT_SYMBOL * n_atoms,
        positions=positions,
        cell=[SIDE_LENGTH_ANGSTROM] * 3,
        pbc=True,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write(path, atoms, format="extxyz")


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


def build_thin_mace_model(
    device: torch.device,
    model: str = "medium-mpa-0",
    head_name: Optional[str] = None,
    model_family: str = "mp",
) -> DistMACEThinAdapter:
    raw_model = load_raw_mace_model(model_family=model_family, model=model, device=device.type)
    supported_atomic_numbers = {int(value) for value in raw_model.atomic_numbers.detach().cpu().tolist()}
    if ELEMENT_ATOMIC_NUMBER not in supported_atomic_numbers:
        raise ValueError(
            f"Selected MACE model_family={model_family!r}, model={model!r} does not support "
            f"{ELEMENT_SYMBOL} (atomic number {ELEMENT_ATOMIC_NUMBER}). Supported atomic numbers: "
            f"{sorted(supported_atomic_numbers)}"
        )
    return DistMACEThinAdapter(raw_model.to(device=device).eval(), head_name=head_name or None)


def make_config(output_dir: str, cutoff_angstrom: float) -> SimulatorConfig:
    os.makedirs(output_dir, exist_ok=True)
    return SimulatorConfig(
        timestep_fs=0.5,
        total_steps=N_STEPS,
        side_length_angstrom=SIDE_LENGTH_ANGSTROM,
        cutoff_angstrom=float(cutoff_angstrom),
        ghost_buffer_angstrom=DEFAULT_GHOST_CUTOFF_BUFFER,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=REBALANCE_INTERVAL,
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
    owned_local_energy = getattr(sim.model, "last_owned_local_energy", None)
    if owned_local_energy is not None:
        frame["owned_local_energy"] = float(owned_local_energy.detach().cpu())
    return frame


def run_md_loop(sim, config: SimulatorConfig):
    frames = []
    for step in range(N_STEPS):
        if sim.should_rebalance(step):
            sim.sync_and_repartition()
        energy, _ = sim.step(config.timestep_fs)
        frames.append(extract_frame(sim, energy))
    return frames


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
        if "owned_local_energy_sum" in dist_frame:
            row["dist_owned_local_energy_sum"] = dist_frame["owned_local_energy_sum"]
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


def compare_trajectories(ref_traj, dist_traj, report_path: Optional[str] = None) -> None:
    if report_path is not None:
        rows = build_comparison_report(ref_traj, dist_traj)
        write_comparison_report_csv(rows, report_path)

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

    _, g_ref = averaged_rdf(ref_traj, RDF_LAST_K_FRAMES, SIDE_LENGTH_ANGSTROM)
    _, g_dist = averaged_rdf(dist_traj, RDF_LAST_K_FRAMES, SIDE_LENGTH_ANGSTROM)
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
