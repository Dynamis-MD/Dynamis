"""Distributed torchrun driver and pytest entrypoints for the thin DistMACE adapter.

This file owns only the distributed adapter path. Standalone monolithic MACE
reference generation lives in ``mace_standalone_reference.py`` so that the
standalone baseline can be exercised without DistNNP graph construction in the
same driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from distnnp_model.nnp.MACE_Adaptation.distmace_thin_adapter import (
    COMPUTE_DTYPE,
    DistMACEThinAdapter,
    canonicalize_nonperiodic_geometry,
    load_raw_mace_model,
    resolve_model_head_index,
    resolve_model_head_name,
)
from distnnp_state import (
    DistNNPState,
    MolecularInfo,
    SFCCurveType,
    SFCPartitioner,
    SimulationConfig,
    TorchDistEnv,
    radius_graph_pbc_cuda,
)
from distnnp_state.masses import TORCH_ATOMIC_MASSES

try:
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

    HAS_ASE = True
except Exception:  # pragma: no cover - optional dependency
    ASE_ATOMIC_NUMBERS = {}
    HAS_ASE = False

ARTIFACT_VERSION = 2
TEST_FILE_DIR = Path(__file__).resolve().parent
ELEMENT_TO_Z = dict(ASE_ATOMIC_NUMBERS)


@dataclass
class PreparedExperiment:
    raw: Dict[str, Any]
    sim: SimulationConfig
    state: DistNNPState
    adapter: DistMACEThinAdapter
    device: torch.device
    model_id: Dict[str, Any]


def natural_chain12() -> Dict[str, Any]:
    """A bent C4H8 fragment with ordinary covalent-scale spacing."""
    carbon = [
        [0.00, 0.00, 0.00],
        [1.52, 0.12, 0.00],
        [3.01, -0.10, 0.18],
        [4.50, 0.08, -0.05],
    ]
    hydrogen = [
        [-0.55, 0.92, 0.20],
        [-0.48, -0.92, -0.20],
        [1.52, 1.08, -0.45],
        [1.55, -0.78, 0.62],
        [3.00, 0.82, 0.72],
        [3.02, -1.05, -0.42],
        [5.03, 0.96, 0.20],
        [5.02, -0.82, -0.28],
    ]
    return {
        "name": "natural_chain12",
        "atomic_numbers": [6] * 4 + [1] * 8,
        "positions": carbon + hydrogen,
        "cell": [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        "pbc": [False, False, False],
    }


BUILTIN_SYSTEMS = {
    "natural_chain12": natural_chain12,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    distributed_parser = subparsers.add_parser(
        "run-distributed",
        help="Run the thin distributed adapter and compare against a detached standalone reference artifact.",
    )
    distributed_parser.add_argument("--builtin-system", choices=BUILTIN_SYSTEMS, default="natural_chain12")
    distributed_parser.add_argument("--xyz-path", type=Path, help="XYZ/extxyz input; overrides --builtin-system.")
    distributed_parser.add_argument("--model-family", choices=["mp", "off", "omol", "polar"], default="mp")
    distributed_parser.add_argument("--model", default="medium-mpa-0")
    distributed_parser.add_argument("--head-name", default="", help="Optional explicit MACE head name.")
    distributed_parser.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    distributed_parser.add_argument(
        "--nonperiodic-padding",
        type=float,
        help="Vacuum padding used before DistNNP halo construction (default: cutoff + buffer).",
    )
    distributed_parser.add_argument(
        "--disable-nonperiodic-canonicalization",
        action="store_true",
        help="Keep raw nonperiodic coordinates for diagnosing raw DistNNP wrapping behavior.",
    )
    distributed_parser.add_argument("--reference-artifact", type=Path, help="Detached standalone reference artifact.")
    distributed_parser.add_argument("--benchmark", action="store_true", help="Time only the distributed adapter loop.")
    distributed_parser.add_argument("--benchmark-warmup", type=int, default=3)
    distributed_parser.add_argument("--benchmark-iterations", type=int, default=10)
    distributed_parser.add_argument("--benchmark-samples", type=int, default=5)
    distributed_parser.add_argument(
        "--time-initialize-buffers",
        action="store_true",
        help="Also report init-only timing for adapter wrapper construction + initialize_buffers.",
    )
    distributed_parser.add_argument("--atol", type=float, default=1e-5)
    distributed_parser.add_argument("--rtol", type=float, default=1e-5)
    distributed_parser.add_argument("--strict-parity", action="store_true")
    distributed_parser.add_argument("--output-json", type=Path)
    return parser


def resolve_xyz_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, TEST_FILE_DIR / path]
        candidates.extend(parent / path for parent in TEST_FILE_DIR.parents)
        if len(path.parts) == 1:
            candidates.append(TEST_FILE_DIR / "data" / path)
    attempted: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        attempted.append(resolved)
        if resolved.is_file():
            return resolved
    attempted_text = "\n".join(f"  - {candidate}" for candidate in attempted)
    raise FileNotFoundError(
        f"XYZ input does not exist: {path}\n"
        f"torchrun working directory: {Path.cwd()}\n"
        f"Attempted paths:\n{attempted_text}"
    )


def load_xyz(path: Path) -> Dict[str, Any]:
    path = resolve_xyz_path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is incomplete: {path}")
    try:
        num_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"XYZ first line must be an atom count: {path}") from exc
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != num_atoms:
        raise ValueError(f"XYZ declares {num_atoms} atoms but contains {len(atom_lines)} rows.")

    atomic_numbers: List[int] = []
    positions: List[List[float]] = []
    for row_index, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"XYZ atom row {row_index} needs symbol and x/y/z coordinates.")
        token = fields[0]
        if token.isdigit():
            atomic_number = int(token)
        elif token in ELEMENT_TO_Z:
            atomic_number = ELEMENT_TO_Z[token]
        else:
            raise ValueError(f"Unsupported element symbol {token!r} on XYZ row {row_index}.")
        atomic_numbers.append(atomic_number)
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])

    comment = lines[1]
    lattice_match = re.search(r'Lattice="([^"]+)"', comment)
    pbc_match = re.search(r'pbc="([^"]+)"', comment, flags=re.IGNORECASE)
    if lattice_match:
        lattice = [float(value) for value in lattice_match.group(1).split()]
        if len(lattice) != 9:
            raise ValueError("extxyz Lattice must contain nine values.")
        cell = [lattice[0:3], lattice[3:6], lattice[6:9]]
    else:
        coordinate_tensor = torch.tensor(positions, dtype=COMPUTE_DTYPE)
        span = coordinate_tensor.max(dim=0).values - coordinate_tensor.min(dim=0).values
        lengths = torch.maximum(span + 20.0, torch.full_like(span, 20.0))
        cell = torch.diag(lengths).tolist()
    if pbc_match:
        tokens = pbc_match.group(1).split()
        if len(tokens) != 3:
            raise ValueError("extxyz pbc must contain three values.")
        pbc = [token.lower() in {"t", "true", "1"} for token in tokens]
    else:
        pbc = [False, False, False]
    return {
        "name": path.stem,
        "source_path": str(path.resolve()),
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
    }


def canonicalize_raw_structure(raw: Dict[str, Any], padding: float) -> Dict[str, Any]:
    positions, cell, translation = canonicalize_nonperiodic_geometry(
        torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE),
        torch.tensor(raw["cell"], dtype=COMPUTE_DTYPE),
        torch.tensor(raw["pbc"], dtype=torch.bool),
        padding,
    )
    adjusted = dict(raw)
    adjusted["positions"] = positions.tolist()
    adjusted["cell"] = cell.tolist()
    adjusted["coordinate_translation"] = translation.tolist()
    adjusted["nonperiodic_padding"] = float(padding)
    return adjusted


def load_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    raw = load_xyz(args.xyz_path) if args.xyz_path else BUILTIN_SYSTEMS[args.builtin_system]()
    if args.disable_nonperiodic_canonicalization:
        return raw
    probe_model = load_raw_mace_model(args.model_family, args.model, "cpu").to(dtype=COMPUTE_DTYPE).eval()
    cutoff = float(probe_model.r_max.detach().cpu())
    padding = args.nonperiodic_padding if args.nonperiodic_padding is not None else cutoff + args.ghost_cutoff_buffer
    return canonicalize_raw_structure(raw, padding)


def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def build_state(
    raw: Dict[str, Any],
    cutoff: float,
    ghost_cutoff_buffer: float,
    torch_env: TorchDistEnv,
) -> Tuple[DistNNPState, SimulationConfig]:
    device = torch_env.device
    numbers = torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device)
    positions = torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE, device=device)
    global_mol = MolecularInfo()
    global_mol.num_atoms = positions.shape[0]
    global_mol.atom_positions = positions
    global_mol.atom_types = numbers
    global_mol.atom_velocities = torch.zeros_like(positions)
    _assign_if_present(global_mol, "atom_forces", torch.zeros_like(positions))
    _assign_if_present(global_mol, "atom_vertex_numbers", torch.arange(positions.shape[0], device=device))
    _assign_if_present(global_mol, "atom_masses", TORCH_ATOMIC_MASSES.to(device)[numbers].float())

    sim = SimulationConfig()
    sim.cutoff = cutoff
    sim.ghost_cutoff_buffer = ghost_cutoff_buffer
    sim.cell = torch.tensor(raw["cell"], dtype=COMPUTE_DTYPE, device=device)
    sim.pbc = torch.tensor(raw["pbc"], dtype=torch.bool, device=device)

    partitioner = SFCPartitioner(SFCCurveType.Morton)
    info = partitioner.partition(global_mol, torch_env.dist_env)
    partitioner.build_halo_maps(info, global_mol, torch_env.dist_env, sim)
    local_indices = info.partition_indices.long()
    local_mol = MolecularInfo()
    local_mol.num_atoms = local_indices.numel()
    local_mol.atom_positions = positions.index_select(0, local_indices)
    local_mol.atom_types = numbers.index_select(0, local_indices)
    local_mol.atom_velocities = torch.zeros_like(local_mol.atom_positions)
    _assign_if_present(local_mol, "atom_forces", torch.zeros_like(local_mol.atom_positions))
    _assign_if_present(local_mol, "atom_vertex_numbers", local_indices)
    _assign_if_present(local_mol, "atom_masses", TORCH_ATOMIC_MASSES.to(device)[local_mol.atom_types].float())

    state = DistNNPState()
    state.dist_env = torch_env.dist_env
    state.sim_config = sim
    state.global_molecular_info = global_mol
    state.local_partition_info = info
    state.local_molecular_info = local_mol
    return state, sim


def radius_graph(
    positions: torch.Tensor,
    sim: SimulationConfig,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = SimpleNamespace(
        pos=positions,
        cell=sim.cell.reshape(1, 3, 3),
        pbc=sim.pbc.reshape(1, 3),
        natoms=torch.tensor([positions.shape[0]], device=positions.device),
    )
    with torch.no_grad():
        edge_index, offsets, _ = radius_graph_pbc_cuda(
            data,
            radius=cutoff,
            max_num_neighbors_threshold=0,
            enforce_max_neighbors_strictly=False,
        )
    return edge_index.long(), -offsets


def edge_records(
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> set[Tuple[int, int, int, int, int]]:
    offsets = unit_shifts.round().long().cpu().tolist()
    pairs = edge_index.T.long().cpu().tolist()
    return {
        (int(pair[0]), int(pair[1]), int(offset[0]), int(offset[1]), int(offset[2]))
        for pair, offset in zip(pairs, offsets)
    }


def explicit_mace_batch_from_tensors(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    model: torch.nn.Module,
    sim: SimulationConfig,
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    head_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    device = positions.device
    model_numbers = model.atomic_numbers.to(device=device, dtype=torch.long)
    node_attrs = numbers[:, None].eq(model_numbers[None, :]).to(COMPUTE_DTYPE)
    cell = sim.cell.reshape(3, 3).to(dtype=COMPUTE_DTYPE)
    unit_shifts = unit_shifts.to(device=device, dtype=COMPUTE_DTYPE)
    head_index = resolve_model_head_index(model, head_name)
    return {
        "positions": positions,
        "edge_index": edge_index.to(device),
        "unit_shifts": unit_shifts,
        "shifts": unit_shifts @ cell,
        "cell": cell,
        "node_attrs": node_attrs,
        "batch": torch.zeros(numbers.numel(), dtype=torch.long, device=device),
        "ptr": torch.tensor([0, numbers.numel()], dtype=torch.long, device=device),
        "head": torch.tensor([head_index], dtype=torch.long, device=device),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _critical_path_seconds(local_seconds: float, device: torch.device) -> float:
    reduce_device = device if dist.is_initialized() and dist.get_backend() == "nccl" else torch.device("cpu")
    elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=reduce_device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.cpu())


def _timing_statistics(seconds_per_step: List[float], num_atoms: int) -> Dict[str, Any]:
    ordered = sorted(seconds_per_step)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    median = statistics.median(ordered)
    mean = statistics.mean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "samples_ms_per_step": [value * 1.0e3 for value in seconds_per_step],
        "median_ms_per_step": median * 1.0e3,
        "mean_ms_per_step": mean * 1.0e3,
        "min_ms_per_step": ordered[0] * 1.0e3,
        "p95_ms_per_step": ordered[p95_index] * 1.0e3,
        "stdev_ms_per_step": stdev * 1.0e3,
        "coefficient_of_variation_percent": 0.0 if mean == 0.0 else 100.0 * stdev / mean,
        "atoms_per_second": 0.0 if median == 0.0 else num_atoms / median,
    }


def _reference_signature(raw: Dict[str, Any], model_family: str, model_name: str, head_name: str) -> str:
    payload = {
        "system": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "atomic_numbers": raw["atomic_numbers"],
        "positions": raw["positions"],
        "cell": raw["cell"],
        "pbc": raw["pbc"],
        "model_family": model_family,
        "model_name": model_name,
        "head_name": head_name,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_reference_artifact(
    artifact_path: Path,
    raw: Dict[str, Any],
    args: argparse.Namespace,
    require_standalone_benchmark: bool = False,
) -> Dict[str, Any]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    expected_head = args.head_name or resolve_model_head_name(
        load_raw_mace_model(args.model_family, args.model, "cpu").to(dtype=COMPUTE_DTYPE).eval(),
        args.head_name or None,
    )
    expected_signature = _reference_signature(raw, args.model_family, args.model, expected_head)
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Reference artifact version mismatch: expected {ARTIFACT_VERSION}, "
            f"got {metadata.get('artifact_version')!r}."
        )
    if metadata.get("signature") != expected_signature:
        raise ValueError("Reference artifact is stale or incompatible with the requested system/model.")
    reference = artifact.get("reference", {})
    required_reference_keys = {
        "calculator_energy",
        "calculator_forces",
        "core_model_energy",
        "core_model_forces",
    }
    missing_reference_keys = sorted(required_reference_keys.difference(reference.keys()))
    if missing_reference_keys:
        raise ValueError(
            "Reference artifact is missing required monolithic reference keys: "
            + ", ".join(missing_reference_keys)
        )
    if require_standalone_benchmark and (
        artifact.get("calculator_benchmark") is None or artifact.get("core_model_benchmark") is None
    ):
        raise ValueError(
            "Benchmark mode requires a reference artifact generated with both calculator and core-model timing stats."
        )
    return artifact


def resolve_reference_artifact_path(args: argparse.Namespace) -> Path:
    if args.reference_artifact is not None:
        return args.reference_artifact
    env_candidates = [
        os.environ.get("DISTMACE_THIN_REFERENCE_ARTIFACT"),
        os.environ.get("DISTMACE_THIN_REFERENCE_DEFAULT_ARTIFACT"),
    ]
    for candidate in env_candidates:
        if candidate:
            return Path(candidate)
    raise ValueError(
        "run-distributed requires --reference-artifact. "
        "For pytest parity runs, set DISTMACE_THIN_REFERENCE_ARTIFACT or a test-specific artifact env var."
    )


def prepare_distributed(args: argparse.Namespace) -> PreparedExperiment:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("Launch this test with torchrun and at least two ranks.")
    if not HAS_ASE and args.xyz_path is not None:
        raise ImportError("ASE is required to decode element symbols from XYZ inputs.")
    torch_env = TorchDistEnv(is_distributed=True)
    if dist.get_world_size() < 2:
        raise RuntimeError("The distributed thin-adapter test requires at least two ranks.")
    device = torch_env.device
    raw = load_raw_structure(args)
    model = load_raw_mace_model(args.model_family, args.model, str(device)).to(device=device).eval()
    cutoff = float(model.r_max.detach().cpu())
    state, sim = build_state(raw, cutoff, args.ghost_cutoff_buffer, torch_env)
    adapter = DistMACEThinAdapter(model, head_name=args.head_name or None)
    adapter.initialize_buffers(state.local_partition_info, device, dist.get_world_size())
    return PreparedExperiment(
        raw=raw,
        sim=sim,
        state=state,
        adapter=adapter,
        device=device,
        model_id={
            "model_family": args.model_family,
            "model_name": args.model,
            "head_name": adapter.head_name,
        },
    )


def gather_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    return [item for item in records if item is not None]


def assemble_forces(records: List[Dict[str, Any]], num_atoms: int) -> torch.Tensor:
    forces = torch.zeros(num_atoms, 3, dtype=COMPUTE_DTYPE)
    for record in records:
        indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        local_forces = torch.tensor(record["owned_local_forces"], dtype=COMPUTE_DTYPE)
        forces.index_copy_(0, indices, local_forces)
    return forces


def run_distributed_adapter_step(prepared: PreparedExperiment) -> Tuple[float, torch.Tensor, List[Dict[str, Any]]]:
    try:
        energy, forces = prepared.adapter(prepared.state)
        record = {
            "rank": dist.get_rank(),
            "local_global_indices": prepared.state.local_partition_info.partition_indices.long().cpu().tolist(),
            "owned_local_energy": float(energy.cpu()),
            "owned_local_forces": forces.detach().cpu().tolist(),
        }
        records = gather_records(record)
        dist.barrier()
        return float(energy.cpu()), forces, records
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if dist.is_initialized() and hasattr(dist, "abort"):
            dist.abort()
        raise


def build_explicit_global_reference(
    prepared: PreparedExperiment,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model = prepared.adapter.model
    cutoff = float(model.r_max.detach().cpu())
    positions = prepared.state.global_molecular_info.atom_positions.detach().clone()
    numbers = prepared.state.global_molecular_info.atom_types.detach().clone()
    explicit_edges, explicit_shifts = radius_graph(positions, prepared.sim, cutoff)
    explicit_batch = explicit_mace_batch_from_tensors(
        positions,
        numbers,
        model,
        prepared.sim,
        explicit_edges,
        explicit_shifts,
        head_name=args.head_name or prepared.adapter.head_name,
    )
    explicit_out = model(
        explicit_batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    explicit_records = edge_records(explicit_edges, explicit_shifts)
    return {
        "energy": float(explicit_out["energy"].detach().cpu().reshape(())),
        "forces": explicit_out["forces"].detach().to(COMPUTE_DTYPE).cpu(),
        "edge_count": len(explicit_records),
    }


def benchmark_distributed(
    prepared: PreparedExperiment,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0 or args.benchmark_samples <= 0:
        raise ValueError("Benchmark warmup must be nonnegative; iterations and samples must be positive.")

    device = prepared.device
    adapter = prepared.adapter
    state = prepared.state
    raw = prepared.raw
    base_model = prepared.adapter.model

    for _ in range(args.benchmark_warmup):
        adapter(state)
    _synchronize(device)
    dist.barrier()

    distributed_samples: List[float] = []
    for _ in range(args.benchmark_samples):
        dist.barrier()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(args.benchmark_iterations):
            adapter(state)
        _synchronize(device)
        local_seconds = (time.perf_counter() - start) / args.benchmark_iterations
        distributed_samples.append(_critical_path_seconds(local_seconds, device))

    init_stats = None
    if args.time_initialize_buffers:
        init_samples: List[float] = []
        for _ in range(args.benchmark_samples):
            dist.barrier()
            _synchronize(device)
            start = time.perf_counter()
            init_adapter = DistMACEThinAdapter(base_model, head_name=args.head_name or None)
            init_adapter.initialize_buffers(state.local_partition_info, device, dist.get_world_size())
            _synchronize(device)
            local_seconds = time.perf_counter() - start
            init_samples.append(_critical_path_seconds(local_seconds, device))
        if dist.get_rank() == 0:
            init_stats = _timing_statistics(init_samples, len(raw["atomic_numbers"]))

    if dist.get_rank() != 0:
        return None
    distributed_stats = _timing_statistics(distributed_samples, len(raw["atomic_numbers"]))
    benchmark = {
        "scope": (
            "steady-state distributed thin-adapter energy+forces; includes position exchange, "
            "owned-graph build, runtime feature exchange, and force recovery; excludes standalone "
            "reference generation, explicit parity reconstruction, and parity comparison"
        ),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "timing_method": "perf_counter with device synchronization; distributed samples use slowest rank",
        "world_size": dist.get_world_size(),
        "warmup_steps": args.benchmark_warmup,
        "iterations_per_sample": args.benchmark_iterations,
        "num_samples": args.benchmark_samples,
        "distributed": distributed_stats,
    }
    if init_stats is not None:
        benchmark["distributed_initialize_buffers"] = init_stats
    return benchmark


def build_summary(
    prepared: PreparedExperiment,
    records: List[Dict[str, Any]],
    explicit_reference: Dict[str, Any],
    artifact: Dict[str, Any],
    args: argparse.Namespace,
    benchmark: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    num_atoms = len(prepared.raw["atomic_numbers"])
    records = sorted(records, key=lambda item: item["rank"])
    distributed_energy = sum(item["owned_local_energy"] for item in records)
    distributed_forces = assemble_forces(records, num_atoms)
    explicit_energy = float(explicit_reference["energy"])
    explicit_forces = explicit_reference["forces"].to(dtype=COMPUTE_DTYPE)
    calculator_energy = float(artifact["reference"]["calculator_energy"])
    calculator_forces = artifact["reference"]["calculator_forces"].to(dtype=COMPUTE_DTYPE)
    core_model_energy = float(artifact["reference"]["core_model_energy"])
    core_model_forces = artifact["reference"]["core_model_forces"].to(dtype=COMPUTE_DTYPE)
    energy_close = torch.isclose(
        torch.tensor(distributed_energy, dtype=COMPUTE_DTYPE),
        torch.tensor(explicit_energy, dtype=COMPUTE_DTYPE),
        atol=args.atol,
        rtol=args.rtol,
    ).item()
    force_close = torch.allclose(
        distributed_forces,
        explicit_forces,
        atol=args.atol,
        rtol=args.rtol,
    )
    classification = (
        "parity_ok"
        if energy_close and force_close
        else "post_graph_model_mismatch"
    )
    explicit_energy_abs = abs(distributed_energy - explicit_energy)
    calculator_energy_abs = abs(distributed_energy - calculator_energy)
    core_model_energy_abs = abs(distributed_energy - core_model_energy)
    explicit_force_delta = distributed_forces - explicit_forces
    calculator_force_delta = distributed_forces - calculator_forces
    core_model_force_delta = distributed_forces - core_model_forces
    summary = {
        "system": prepared.raw["name"],
        "source_path": prepared.raw.get("source_path", f"builtin://{prepared.raw['name']}"),
        "num_atoms": num_atoms,
        "model": prepared.model_id,
        "reference_artifact": artifact["metadata"],
        "mismatch_classification": classification,
        "parity": {
            "distributed_energy": distributed_energy,
            "summed_owned_local_energy": distributed_energy,
            "explicit_distnnp_graph_energy": explicit_energy,
            "calculator_mace_energy": calculator_energy,
            "core_model_mace_energy": core_model_energy,
            "distributed_minus_explicit_energy": distributed_energy - explicit_energy,
            "distributed_minus_calculator_energy": distributed_energy - calculator_energy,
            "distributed_minus_core_model_energy": distributed_energy - core_model_energy,
            "explicit_energy_abs_error": explicit_energy_abs,
            "calculator_energy_abs_error": calculator_energy_abs,
            "core_model_energy_abs_error": core_model_energy_abs,
            "explicit_energy_abs_error_per_atom": explicit_energy_abs / num_atoms,
            "calculator_energy_abs_error_per_atom": calculator_energy_abs / num_atoms,
            "core_model_energy_abs_error_per_atom": core_model_energy_abs / num_atoms,
            "explicit_force_mae": float(explicit_force_delta.abs().mean()),
            "explicit_force_max_abs": float(explicit_force_delta.abs().max()),
            "calculator_force_mae": float(calculator_force_delta.abs().mean()),
            "calculator_force_max_abs": float(calculator_force_delta.abs().max()),
            "core_model_force_mae": float(core_model_force_delta.abs().mean()),
            "core_model_force_max_abs": float(core_model_force_delta.abs().max()),
            "explicit_energy_close": bool(energy_close),
            "explicit_forces_close": bool(force_close),
            "explicit_edge_count": int(explicit_reference["edge_count"]),
            "core_model_edge_count": int(artifact["reference"]["core_model_edge_count"]),
            "calculator_minus_core_model_energy": float(artifact["reference"]["calculator_minus_core_model_energy"]),
            "calculator_minus_core_model_force_mae": float(
                artifact["reference"]["calculator_minus_core_model_force_mae"]
            ),
            "calculator_minus_core_model_force_max_abs": float(
                artifact["reference"]["calculator_minus_core_model_force_max_abs"]
            ),
        },
        "ranks": records,
    }
    if benchmark is not None:
        calculator_benchmark = artifact["calculator_benchmark"]
        core_model_benchmark = artifact["core_model_benchmark"]
        calculator_stats = calculator_benchmark["timings"]
        core_model_stats = core_model_benchmark["timings"]
        calculator_speedup = calculator_stats["median_ms_per_step"] / benchmark["distributed"]["median_ms_per_step"]
        core_model_speedup = core_model_stats["median_ms_per_step"] / benchmark["distributed"]["median_ms_per_step"]
        benchmark["calculator_baseline"] = {
            "scope": calculator_benchmark["scope"],
            "timings": calculator_stats,
            "speedup_over_distributed": calculator_speedup,
            "parallel_efficiency": calculator_speedup / dist.get_world_size(),
        }
        benchmark["core_model_baseline"] = {
            "scope": core_model_benchmark["scope"],
            "timings": core_model_stats,
            "speedup_over_distributed": core_model_speedup,
            "parallel_efficiency": core_model_speedup / dist.get_world_size(),
        }
        summary["benchmark"] = benchmark
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Thin DistMACE Explicit-Graph Experiment")
    print("=" * 60)
    print(f"system: {summary['system']}")
    print(f"source_path: {summary['source_path']}")
    print(f"num_atoms: {summary['num_atoms']}")
    print(f"model: {summary['model']}")
    print(f"mismatch_classification: {summary['mismatch_classification']}")
    print("\n" + "-" * 60 + "\nParity\n" + "-" * 60)
    for key, value in summary["parity"].items():
        print(f"{key}: {value}")
    print("\n" + "-" * 60 + "\nPer-Rank\n" + "-" * 60)
    for record in summary["ranks"]:
        print(
            f"rank {record['rank']}: local_atoms={len(record['local_global_indices'])}, "
            f"owned_local_energy={record['owned_local_energy']}"
        )
    if "benchmark" in summary:
        benchmark = summary["benchmark"]
        print("\n" + "-" * 60 + "\nPerformance Benchmark\n" + "-" * 60)
        print(f"scope: {benchmark['scope']}")
        print(f"device_name: {benchmark['device_name']}")
        print(f"timing_method: {benchmark['timing_method']}")
        print(f"world_size: {benchmark['world_size']}")
        print(f"warmup_steps: {benchmark['warmup_steps']}")
        print(f"iterations_per_sample: {benchmark['iterations_per_sample']}")
        print(f"num_samples: {benchmark['num_samples']}")
        print(f"distributed_median_ms_per_step: {benchmark['distributed']['median_ms_per_step']}")
        print(f"calculator_scope: {benchmark['calculator_baseline']['scope']}")
        print(f"calculator_median_ms_per_step: {benchmark['calculator_baseline']['timings']['median_ms_per_step']}")
        print(f"speedup_calculator_over_distributed: {benchmark['calculator_baseline']['speedup_over_distributed']}")
        print(f"calculator_parallel_efficiency: {benchmark['calculator_baseline']['parallel_efficiency']}")
        print(f"core_model_scope: {benchmark['core_model_baseline']['scope']}")
        print(f"core_model_median_ms_per_step: {benchmark['core_model_baseline']['timings']['median_ms_per_step']}")
        print(f"speedup_core_model_over_distributed: {benchmark['core_model_baseline']['speedup_over_distributed']}")
        print(f"core_model_parallel_efficiency: {benchmark['core_model_baseline']['parallel_efficiency']}")
        if "distributed_initialize_buffers" in benchmark:
            print(
                "distributed_initialize_buffers_median_ms: "
                f"{benchmark['distributed_initialize_buffers']['median_ms_per_step']}"
            )


def run_distributed(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    prepared = prepare_distributed(args)
    artifact_path = resolve_reference_artifact_path(args)
    artifact = load_and_validate_reference_artifact(
        artifact_path,
        prepared.raw,
        args,
        require_standalone_benchmark=args.benchmark,
    )
    _, _, records = run_distributed_adapter_step(prepared)
    benchmark = benchmark_distributed(prepared, args) if args.benchmark else None
    if dist.get_rank() != 0:
        return None
    explicit_reference = build_explicit_global_reference(prepared, args)
    summary = build_summary(prepared, records, explicit_reference, artifact, args, benchmark=benchmark)
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.strict_parity:
        assert summary["mismatch_classification"] == "parity_ok", summary["mismatch_classification"]
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run-distributed":
            run_distributed(args)
        else:  # pragma: no cover - argparse enforces the command
            raise ValueError(f"Unsupported command: {args.command!r}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _require_reference_env(*names: str) -> str:
    import pytest

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    pytest.skip(
        "Set a detached standalone reference artifact env var before running parity tests. "
        f"Tried: {', '.join(names)}"
    )


def test_distmace_thin_adapter_parity_builtin() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env(
        "DISTMACE_THIN_REFERENCE_BUILTIN_ARTIFACT",
        "DISTMACE_THIN_REFERENCE_ARTIFACT",
        "DISTMACE_THIN_REFERENCE_DEFAULT_ARTIFACT",
    )
    args = build_parser().parse_args(
        ["run-distributed", "--reference-artifact", artifact, "--strict-parity"]
    )
    run_distributed(args)


def test_distmace_thin_adapter_parity_halo_fixture() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env(
        "DISTMACE_THIN_REFERENCE_HALO_ARTIFACT",
        "DISTMACE_THIN_REFERENCE_ARTIFACT",
        "DISTMACE_THIN_REFERENCE_DEFAULT_ARTIFACT",
    )
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--xyz-path",
            "nonperiodic_compact_chain12.xyz",
            "--reference-artifact",
            artifact,
            "--strict-parity",
        ]
    )
    run_distributed(args)


def test_distmace_thin_adapter_reinitialize_buffers() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    args = build_parser().parse_args(["run-distributed"])
    prepared = prepare_distributed(args)
    energy0, forces0 = prepared.adapter(prepared.state)
    prepared.adapter.initialize_buffers(
        prepared.state.local_partition_info,
        prepared.device,
        dist.get_world_size(),
    )
    energy1, forces1 = prepared.adapter(prepared.state)
    assert torch.isclose(energy0.cpu(), energy1.cpu(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(forces0.cpu(), forces1.cpu(), atol=1e-6, rtol=1e-6)


def test_reference_artifact_rejects_missing_benchmark_stats(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.pt"
    torch.save(
        {
            "metadata": {
                "artifact_version": ARTIFACT_VERSION,
                "signature": "abc",
            },
            "reference": {
                "calculator_energy": 0.0,
                "calculator_forces": torch.zeros(1, 3, dtype=COMPUTE_DTYPE),
                "core_model_energy": 0.0,
                "core_model_forces": torch.zeros(1, 3, dtype=COMPUTE_DTYPE),
            },
            "calculator_benchmark": None,
            "core_model_benchmark": None,
        },
        artifact_path,
    )
    raw = natural_chain12()
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--reference-artifact",
            str(artifact_path),
            "--benchmark",
        ]
    )
    original_loader = load_raw_mace_model

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("r_max", torch.tensor(1.0))
            self.heads = ["Default"]

    try:
        globals()["load_raw_mace_model"] = lambda *a, **k: _DummyModel()
        import pytest

        with pytest.raises(ValueError, match="stale or incompatible|timing stats"):
            load_and_validate_reference_artifact(
                artifact_path,
                raw,
                args,
                require_standalone_benchmark=True,
            )
    finally:
        globals()["load_raw_mace_model"] = original_loader


if __name__ == "__main__":
    main()
