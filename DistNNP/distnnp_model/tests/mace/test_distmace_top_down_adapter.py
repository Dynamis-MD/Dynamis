"""Distributed integration test and experiment driver for DistMACETopDownAdapter.

Run this file with torchrun. The adapter module intentionally contains no CLI,
fixtures, or monolithic-reference code.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
import torch.distributed as dist

from distnnp_model.nnp.MACE_Adaptation.distmace_top_down_adapter import (
    COMPUTE_DTYPE,
    DistMACETopDownAdapter,
    canonicalize_nonperiodic_geometry,
    load_raw_mace_model,
    resolve_model_head_index,
    resolve_model_head_name,
    summarize_model_contract,
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
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    HAS_MACE = True
except Exception:  # pragma: no cover - cluster dependency
    HAS_MACE = False
    ASE_ATOMIC_NUMBERS = {}


@dataclass
class PreparedExperiment:
    raw: Dict[str, Any]
    sim: SimulationConfig
    state: DistNNPState
    model: Any
    contract: Dict[str, Any]
    adapter: DistMACETopDownAdapter
    device: torch.device
    reference_model: Optional[Any]


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


def natural_stress_bundle48() -> Dict[str, Any]:
    """Four separated, rotated C4H8 fragments that create several boundaries."""
    base = natural_chain12()
    base_positions = torch.tensor(base["positions"], dtype=COMPUTE_DTYPE)
    placements = [
        (0.0, 0.0, 0.0, 0.00),
        (3.0, 3.8, 0.5, 0.35),
        (7.2, 0.7, 1.8, -0.25),
        (9.5, 4.4, 0.1, 0.55),
    ]
    all_positions: List[List[float]] = []
    all_numbers: List[int] = []
    for tx, ty, tz, angle in placements:
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        shifted = base_positions @ rotation.T + torch.tensor([tx, ty, tz])
        all_positions.extend(shifted.tolist())
        all_numbers.extend(base["atomic_numbers"])
    return {
        "name": "natural_stress_bundle48",
        "atomic_numbers": all_numbers,
        "positions": all_positions,
        "cell": [[24.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 16.0]],
        "pbc": [False, False, False],
    }


BUILTIN_SYSTEMS = {
    "natural_chain12": natural_chain12,
    "natural_stress_bundle48": natural_stress_bundle48,
}

ELEMENT_TO_Z = dict(ASE_ATOMIC_NUMBERS)
TEST_FILE_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builtin-system", choices=BUILTIN_SYSTEMS, default="natural_chain12")
    parser.add_argument("--xyz-path", type=Path, help="XYZ/extxyz input; overrides --builtin-system.")
    parser.add_argument("--model-family", choices=["mp", "off", "omol", "polar"], default="mp")
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--comparison-model", default="")
    parser.add_argument(
        "--head-name",
        default="",
        help="Optional explicit MACE head name for multi-head MP/MPA checkpoints.",
    )
    parser.add_argument(
        "--debug-exchange",
        action="store_true",
        help="Print per-rank adapter progress at each interaction and exchange boundary.",
    )
    parser.add_argument(
        "--debug-log-prefix",
        default="distmace-debug",
        help="Prefix used by adapter-side debug tracing.",
    )
    parser.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    parser.add_argument(
        "--nonperiodic-padding",
        type=float,
        help="Vacuum padding used before DistNNP halo construction (default: cutoff + buffer).",
    )
    parser.add_argument(
        "--disable-nonperiodic-canonicalization",
        action="store_true",
        help="Keep raw nonperiodic coordinates for diagnosing DistNNP wrapping behavior.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--strict-parity", action="store_true")
    parser.add_argument(
        "--validation-mode",
        choices=["full", "minimal"],
        default="full",
        help=(
            "Validation scope. 'full' runs monolithic graph/force parity; 'minimal' keeps "
            "the distributed execution and structural diagnostics but skips monolithic references."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="Benchmark validated DistNNP-MACE against single-GPU standalone MACE.",
    )
    benchmark_group.add_argument(
        "--no-benchmark",
        dest="benchmark",
        action="store_false",
        help="Disable benchmarking explicitly.",
    )
    parser.set_defaults(benchmark=False)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument("--benchmark-samples", type=int, default=5)
    return parser


def resolve_xyz_path(path: Path) -> Path:
    """Resolve CLI paths independently of the directory used to launch torchrun."""
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
        f"Attempted paths:\n{attempted_text}\n"
        "If this is a bundled fixture, ensure distnnp_model/tests/mace/data was "
        "copied or committed with the test driver."
    )


def load_xyz(path: Path) -> Dict[str, Any]:
    """Read the atom symbols, positions, Lattice, and pbc fields from XYZ/extxyz."""
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


def canonicalize_raw_structure(
    raw: Dict[str, Any], padding: float
) -> Dict[str, Any]:
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
    global_mol.atom_forces = torch.zeros_like(positions)
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
    local_mol.atom_forces = torch.zeros_like(local_mol.atom_positions)
    _assign_if_present(local_mol, "atom_vertex_numbers", local_indices)
    _assign_if_present(local_mol, "atom_masses", TORCH_ATOMIC_MASSES.to(device)[local_mol.atom_types].float())

    state = DistNNPState()
    state.dist_env = torch_env.dist_env
    state.sim_config = sim
    state.global_molecular_info = global_mol
    state.local_partition_info = info
    state.local_molecular_info = local_mol
    return state, sim


def gather_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    return [item for item in records if item is not None]


def emit_reference_debug(enabled: bool, stage: str, **payload: Any) -> None:
    if not enabled:
        return
    rank = dist.get_rank() if dist.is_initialized() else -1
    message = [f"[distmace-reference][rank={rank}][stage={stage}]"]
    for key, value in payload.items():
        message.append(f"{key}={value}")
    print(" ".join(message), flush=True)


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
    edge_index: torch.Tensor, unit_shifts: torch.Tensor
) -> set[Tuple[int, int, int, int, int]]:
    offsets = unit_shifts.round().long().cpu().tolist()
    pairs = edge_index.T.long().cpu().tolist()
    return {
        (int(pair[0]), int(pair[1]), int(offset[0]), int(offset[1]), int(offset[2]))
        for pair, offset in zip(pairs, offsets)
    }


def reverse_edge_symmetry(
    records: set[Tuple[int, int, int, int, int]],
) -> Tuple[bool, List[List[int]]]:
    """Check that every directed arc has its reverse with the opposite image shift."""
    missing = []
    for src, dst, sx, sy, sz in records:
        reverse = (dst, src, -sx, -sy, -sz)
        if reverse not in records:
            missing.append([src, dst, sx, sy, sz])
    return not missing, sorted(missing)[:12]


def explicit_mace_batch(
    raw: Dict[str, Any],
    model,
    sim: SimulationConfig,
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    positions = torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE, device=device)
    numbers = torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device)
    return explicit_mace_batch_from_tensors(
        positions, numbers, model, sim, edge_index, unit_shifts
    )


def explicit_mace_batch_from_tensors(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    model,
    sim: SimulationConfig,
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    head_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Package an explicit graph without invoking MACE neighbor generation."""
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
    reduce_device = device if dist.get_backend() == "nccl" else torch.device("cpu")
    elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=reduce_device)
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
        "coefficient_of_variation_percent": 100.0 * stdev / mean,
        "atoms_per_second": num_atoms / median,
    }


def benchmark_speedup(
    raw: Dict[str, Any],
    sim: SimulationConfig,
    state: DistNNPState,
    adapter: DistMACETopDownAdapter,
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Compare full distributed steps with equivalent one-GPU global steps."""
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0 or args.benchmark_samples <= 0:
        raise ValueError("Benchmark warmup must be nonnegative; iterations and samples must be positive.")

    rank = dist.get_rank()
    num_atoms = len(raw["atomic_numbers"])
    previous_mode = adapter.execution_mode
    previous_diagnostics = adapter.diagnostics_enabled
    previous_debug = adapter.debug_tracing
    adapter.set_execution_mode("benchmark")
    adapter.set_diagnostics(False)
    adapter.set_debug_tracing(False)
    assert adapter.execution_mode == "benchmark"
    assert not adapter.diagnostics_enabled
    assert not adapter.debug_tracing
    assert not adapter.model._debug_enabled()

    try:
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

        standalone_model = None
        standalone_positions = None
        standalone_numbers = None
        cutoff = adapter.cutoff
        if rank == 0:
            standalone_model = load_raw_mace_model(
                args.model_family, args.model, str(device)
            ).to(device=device, dtype=COMPUTE_DTYPE).eval()
            standalone_positions = torch.tensor(
                raw["positions"], dtype=COMPUTE_DTYPE, device=device
            )
            standalone_numbers = torch.tensor(
                raw["atomic_numbers"], dtype=torch.long, device=device
            )

        def standalone_step() -> None:
            assert standalone_model is not None
            assert standalone_positions is not None
            assert standalone_numbers is not None
            positions = standalone_positions.detach().clone()
            edge_index, unit_shifts = radius_graph(positions, sim, cutoff)
            batch = explicit_mace_batch_from_tensors(
                positions,
                standalone_numbers,
                standalone_model,
                sim,
                edge_index,
                unit_shifts,
                head_name=args.head_name or None,
            )
            standalone_model(
                batch,
                training=False,
                compute_force=True,
                compute_stress=False,
                compute_hessian=False,
            )

        dist.barrier()
        if rank == 0:
            for _ in range(args.benchmark_warmup):
                standalone_step()
            _synchronize(device)
        dist.barrier()

        standalone_samples: List[float] = []
        for _ in range(args.benchmark_samples):
            sample = torch.zeros(
                (),
                dtype=torch.float64,
                device=device if dist.get_backend() == "nccl" else torch.device("cpu"),
            )
            dist.barrier()
            if rank == 0:
                _synchronize(device)
                start = time.perf_counter()
                for _ in range(args.benchmark_iterations):
                    standalone_step()
                _synchronize(device)
                sample.fill_((time.perf_counter() - start) / args.benchmark_iterations)
            dist.broadcast(sample, src=0)
            standalone_samples.append(float(sample.cpu()))

        dist.barrier()
        if rank != 0:
            return None
        distributed_stats = _timing_statistics(distributed_samples, num_atoms)
        standalone_stats = _timing_statistics(standalone_samples, num_atoms)
        speedup = (
            standalone_stats["median_ms_per_step"]
            / distributed_stats["median_ms_per_step"]
        )
        return {
            "scope": (
                "steady-state energy+forces; includes radius graph construction and "
                "adapter communication; excludes model loading, partitioning, halo-map "
                "setup, diagnostics, and parity validation"
            ),
            "standalone_baseline": "single GPU, full global explicit DistNNP-compatible graph",
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "timing_method": "perf_counter with device synchronization; distributed samples use slowest rank",
            "world_size": dist.get_world_size(),
            "warmup_steps": args.benchmark_warmup,
            "iterations_per_sample": args.benchmark_iterations,
            "num_samples": args.benchmark_samples,
            "distributed": distributed_stats,
            "standalone": standalone_stats,
            "speedup_standalone_over_distributed": speedup,
            "parallel_efficiency": speedup / dist.get_world_size(),
        }
    finally:
        adapter.set_execution_mode(previous_mode)
        adapter.set_diagnostics(previous_diagnostics)
        adapter.set_debug_tracing(previous_debug)


def standalone_mace_batch(
    raw: Dict[str, Any], model, cutoff: float, device: torch.device, head_name: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor], set[Tuple[int, int, int, int, int]]]:
    resolved_head = resolve_model_head_name(model, head_name)
    config = Configuration(
        atomic_numbers=np.asarray(raw["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(raw["positions"], dtype=np.float32),
        properties={},
        property_weights={},
        cell=np.asarray(raw["cell"], dtype=np.float32),
        pbc=tuple(raw["pbc"]),
        weight=1.0,
        config_type="DistMACE_reference",
        head=resolved_head,
    )
    model_heads = list(getattr(model, "heads", []))
    z_table = AtomicNumberTable([int(value) for value in model.atomic_numbers.cpu().tolist()])
    atomic_data = AtomicData.from_config(
        config=config,
        z_table=z_table,
        cutoff=cutoff,
        heads=model_heads or [resolved_head],
    )
    records = edge_records(atomic_data.edge_index, atomic_data.unit_shifts)
    batch = Batch.from_data_list([atomic_data]).to(device).to_dict()
    batch = {
        key: value.to(COMPUTE_DTYPE)
        if torch.is_tensor(value) and torch.is_floating_point(value)
        else value
        for key, value in batch.items()
    }
    return batch, records


def monolithic_references(
    raw: Dict[str, Any],
    sim: SimulationConfig,
    args: argparse.Namespace,
    device: torch.device,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    emit_reference_debug(
        args.debug_exchange,
        "reference_begin",
        comparison_model=args.comparison_model or args.model,
    )
    if model is None:
        model = load_raw_mace_model(
            args.model_family, args.comparison_model or args.model, str(device)
        ).to(device=device, dtype=COMPUTE_DTYPE).eval()
    cutoff = float(model.r_max.detach().cpu())
    positions = torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE, device=device)
    dist_edges, dist_shifts = radius_graph(positions, sim, cutoff)
    emit_reference_debug(
        args.debug_exchange,
        "explicit_batch_ready",
        cutoff=cutoff,
        edge_count=int(dist_edges.shape[1]),
    )
    explicit_batch = explicit_mace_batch_from_tensors(
        positions,
        torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device),
        model,
        sim,
        dist_edges,
        dist_shifts,
        head_name=args.head_name or None,
    )
    emit_reference_debug(args.debug_exchange, "explicit_forward_begin")
    explicit_out = model(
        explicit_batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    emit_reference_debug(args.debug_exchange, "explicit_forward_end")
    emit_reference_debug(args.debug_exchange, "standalone_batch_begin")
    standalone_batch, standalone_records = standalone_mace_batch(
        raw,
        model,
        cutoff,
        device,
        head_name=args.head_name or None,
    )
    emit_reference_debug(
        args.debug_exchange,
        "standalone_batch_end",
        edge_count=len(standalone_records),
    )
    dist_records = edge_records(dist_edges, dist_shifts)
    dist_symmetry, dist_missing_reverse = reverse_edge_symmetry(dist_records)
    mace_symmetry, mace_missing_reverse = reverse_edge_symmetry(standalone_records)
    graph_builder_agreement = dist_records == standalone_records
    if graph_builder_agreement:
        standalone_out = explicit_out
    else:
        emit_reference_debug(args.debug_exchange, "standalone_forward_begin")
        standalone_out = model(
            standalone_batch,
            training=False,
            compute_force=True,
            compute_stress=False,
            compute_hessian=False,
        )
        emit_reference_debug(args.debug_exchange, "standalone_forward_end")
    return {
        "cutoff": cutoff,
        "distnnp_graph_records": dist_records,
        "standalone_graph_records": standalone_records,
        "distnnp_edge_count": len(dist_records),
        "standalone_mace_edge_count": len(standalone_records),
        "distnnp_reverse_edge_symmetry": dist_symmetry,
        "standalone_mace_reverse_edge_symmetry": mace_symmetry,
        "distnnp_missing_reverse_preview": dist_missing_reverse,
        "standalone_mace_missing_reverse_preview": mace_missing_reverse,
        "graph_builder_agreement": graph_builder_agreement,
        "distnnp_only_edge_preview": [list(item) for item in sorted(dist_records - standalone_records)[:12]],
        "mace_only_edge_preview": [list(item) for item in sorted(standalone_records - dist_records)[:12]],
        "explicit_energy": float(explicit_out["energy"].detach().cpu().reshape(())),
        "explicit_node_energy": explicit_out["node_energy"].detach().to(COMPUTE_DTYPE).cpu().reshape(-1),
        "explicit_forces": explicit_out["forces"].detach().to(COMPUTE_DTYPE).cpu(),
        "standalone_energy": float(standalone_out["energy"].detach().cpu().reshape(())),
        "standalone_forces": standalone_out["forces"].detach().to(COMPUTE_DTYPE).cpu(),
    }


def assemble_forces(records: List[Dict[str, Any]], num_atoms: int) -> torch.Tensor:
    forces = torch.zeros(num_atoms, 3, dtype=COMPUTE_DTYPE)
    for record in records:
        indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        local_forces = torch.tensor(record["owned_local_force_values"], dtype=COMPUTE_DTYPE)
        forces.index_copy_(0, indices, local_forces)
    return forces


def build_summary(
    raw: Dict[str, Any],
    records: List[Dict[str, Any]],
    reference: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    records = sorted(records, key=lambda item: item["rank"])
    owned_indices = [index for item in records for index in item["local_global_indices"]]
    ownership_ok = sorted(owned_indices) == list(range(len(raw["atomic_numbers"])))
    send_matrix = torch.tensor([item["send_counts_by_peer"] for item in records])
    recv_matrix = torch.tensor([item["recv_counts_by_peer"] for item in records])
    peer_symmetry = torch.equal(send_matrix, recv_matrix.T)
    distributed_energy = sum(item["owned_local_energy"] for item in records)
    distributed_forces = assemble_forces(records, len(raw["atomic_numbers"]))
    distributed_node_energy = torch.zeros(
        len(raw["atomic_numbers"]), dtype=COMPUTE_DTYPE
    )
    for record in records:
        indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        values = torch.tensor(record["owned_node_energy_values"], dtype=COMPUTE_DTYPE)
        distributed_node_energy.index_copy_(0, indices, values)
    halo_builder_issue = any(
        item["graph_verification"]["halo_builder_issue"] for item in records
    )
    local_reconstruction_issue = any(
        item["graph_verification"]["local_graph_reconstruction_issue"]
        for item in records
    )
    structural_ok = all(
        item["graph_verification"]["structural_classification"] == "structural_graph_ok"
        for item in records
    )
    directionality_ok = (
        reference["distnnp_reverse_edge_symmetry"]
        and reference["standalone_mace_reverse_edge_symmetry"]
    )
    explicit_forces = reference["explicit_forces"].to(dtype=COMPUTE_DTYPE)
    standalone_forces = reference["standalone_forces"].to(dtype=COMPUTE_DTYPE)
    explicit_node_energy = reference["explicit_node_energy"].to(dtype=COMPUTE_DTYPE)
    explicit_energy_ok = torch.isclose(
        torch.tensor(distributed_energy, dtype=COMPUTE_DTYPE),
        torch.tensor(reference["explicit_energy"], dtype=COMPUTE_DTYPE),
        atol=args.atol,
        rtol=args.rtol,
    ).item()
    explicit_force_ok = torch.allclose(
        distributed_forces,
        explicit_forces,
        atol=args.atol,
        rtol=args.rtol,
    )
    explicit_node_energy_ok = torch.allclose(
        distributed_node_energy,
        explicit_node_energy,
        atol=args.atol,
        rtol=args.rtol,
    )
    distributed_dtypes_ok = all(
        item["model_parameter_dtype"] == "torch.float32"
        and item["position_dtype"] == "torch.float32"
        and item["force_dtype"] == "torch.float32"
        for item in records
    )
    if halo_builder_issue:
        classification = "halo_builder_issue"
    elif (
        local_reconstruction_issue
        or not directionality_ok
        or not reference["graph_builder_agreement"]
    ):
        classification = "graph_builder_disagreement"
    elif not (
        explicit_energy_ok
        and explicit_force_ok
        and explicit_node_energy_ok
        and distributed_dtypes_ok
    ):
        classification = "post_graph_model_mismatch"
    else:
        classification = "parity_ok"
    explicit_force_delta = distributed_forces - explicit_forces
    explicit_node_energy_delta = distributed_node_energy - explicit_node_energy
    standalone_force_delta = distributed_forces - standalone_forces
    return {
        "system": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "num_atoms": len(raw["atomic_numbers"]),
        "coordinate_translation": raw.get("coordinate_translation", [0.0, 0.0, 0.0]),
        "nonperiodic_padding": raw.get("nonperiodic_padding"),
        "mismatch_classification": classification,
        "pre_validation": {
            "all_atoms_owned_exactly_once": ownership_ok,
            "send_recv_peer_symmetry": peer_symmetry,
            "all_structural_graphs_ok": structural_ok,
            "graph_builder_agreement": reference["graph_builder_agreement"],
            "cutoff": reference["cutoff"],
            "distnnp_edge_count": reference["distnnp_edge_count"],
            "standalone_mace_edge_count": reference["standalone_mace_edge_count"],
            "distnnp_reverse_edge_symmetry": reference["distnnp_reverse_edge_symmetry"],
            "standalone_mace_reverse_edge_symmetry": reference["standalone_mace_reverse_edge_symmetry"],
            "distnnp_missing_reverse_preview": reference["distnnp_missing_reverse_preview"],
            "standalone_mace_missing_reverse_preview": reference["standalone_mace_missing_reverse_preview"],
            "distnnp_only_edge_preview": reference["distnnp_only_edge_preview"],
            "mace_only_edge_preview": reference["mace_only_edge_preview"],
            "rank_graph_verification": [item["graph_verification"] for item in records],
        },
        "parity": {
            "distributed_energy": distributed_energy,
            "explicit_distnnp_graph_energy": reference["explicit_energy"],
            "standalone_mace_energy": reference["standalone_energy"],
            "distributed_minus_explicit_energy": distributed_energy - reference["explicit_energy"],
            "distributed_minus_standalone_energy": distributed_energy - reference["standalone_energy"],
            "explicit_force_mae": float(explicit_force_delta.abs().mean()),
            "explicit_force_max_abs": float(explicit_force_delta.abs().max()),
            "explicit_node_energy_mae": float(explicit_node_energy_delta.abs().mean()),
            "explicit_node_energy_max_abs": float(explicit_node_energy_delta.abs().max()),
            "standalone_force_mae": float(standalone_force_delta.abs().mean()),
            "standalone_force_max_abs": float(standalone_force_delta.abs().max()),
            "explicit_energy_close": bool(explicit_energy_ok),
            "explicit_forces_close": bool(explicit_force_ok),
            "explicit_node_energy_close": bool(explicit_node_energy_ok),
            "all_distributed_dtypes_float32": distributed_dtypes_ok,
        },
        "ranks": records,
    }


def build_minimal_summary(
    raw: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    records = sorted(records, key=lambda item: item["rank"])
    owned_indices = [index for item in records for index in item["local_global_indices"]]
    ownership_ok = sorted(owned_indices) == list(range(len(raw["atomic_numbers"])))
    send_matrix = torch.tensor([item["send_counts_by_peer"] for item in records])
    recv_matrix = torch.tensor([item["recv_counts_by_peer"] for item in records])
    peer_symmetry = torch.equal(send_matrix, recv_matrix.T)
    structural_ok = all(
        item["graph_verification"]["structural_classification"] == "structural_graph_ok"
        for item in records
    )
    all_force_shapes_ok = all(
        tuple(item["returned_force_shape"]) == (len(item["local_global_indices"]), 3)
        for item in records
    )
    classification = (
        "minimal_ok"
        if ownership_ok and peer_symmetry and structural_ok and all_force_shapes_ok
        else "minimal_structural_mismatch"
    )
    return {
        "system": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "num_atoms": len(raw["atomic_numbers"]),
        "coordinate_translation": raw.get("coordinate_translation", [0.0, 0.0, 0.0]),
        "nonperiodic_padding": raw.get("nonperiodic_padding"),
        "validation_mode": "minimal",
        "mismatch_classification": classification,
        "pre_validation": {
            "all_atoms_owned_exactly_once": ownership_ok,
            "send_recv_peer_symmetry": peer_symmetry,
            "all_structural_graphs_ok": structural_ok,
            "graph_builder_agreement": None,
            "cutoff": None,
            "distnnp_edge_count": None,
            "standalone_mace_edge_count": None,
            "distnnp_reverse_edge_symmetry": None,
            "standalone_mace_reverse_edge_symmetry": None,
            "distnnp_missing_reverse_preview": [],
            "standalone_mace_missing_reverse_preview": [],
            "distnnp_only_edge_preview": [],
            "mace_only_edge_preview": [],
            "rank_graph_verification": [item["graph_verification"] for item in records],
        },
        "parity": {
            "distributed_energy": sum(item["owned_local_energy"] for item in records),
            "explicit_distnnp_graph_energy": None,
            "standalone_mace_energy": None,
            "distributed_minus_explicit_energy": None,
            "distributed_minus_standalone_energy": None,
            "explicit_force_mae": None,
            "explicit_force_max_abs": None,
            "explicit_node_energy_mae": None,
            "explicit_node_energy_max_abs": None,
            "standalone_force_mae": None,
            "standalone_force_max_abs": None,
            "explicit_energy_close": None,
            "explicit_forces_close": None,
            "explicit_node_energy_close": None,
            "all_distributed_dtypes_float32": all(
                item["model_parameter_dtype"] == "torch.float32"
                and item["position_dtype"] == "torch.float32"
                and item["force_dtype"] == "torch.float32"
                for item in records
            ),
            "all_returned_force_shapes_local": all_force_shapes_ok,
        },
        "ranks": records,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 60)
    print("DistMACE Explicit-Graph Experiment")
    print("=" * 60)
    print(f"system: {summary['system']}")
    print(f"source_path: {summary['source_path']}")
    print(f"num_atoms: {summary['num_atoms']}")
    print(f"coordinate_translation: {summary['coordinate_translation']}")
    print(f"validation_mode: {summary.get('validation_mode', 'full')}")
    print(f"mismatch_classification: {summary['mismatch_classification']}")
    if "model_contract" in summary:
        print("\n" + "-" * 60 + "\nModel Contract\n" + "-" * 60)
        for key, value in summary["model_contract"].items():
            print(f"{key}: {value}")
    print("\n" + "-" * 60 + "\nPre-Validation\n" + "-" * 60)
    print(f"all_atoms_owned_exactly_once: {summary['pre_validation']['all_atoms_owned_exactly_once']}")
    print(f"send_recv_peer_symmetry: {summary['pre_validation']['send_recv_peer_symmetry']}")
    print(f"all_structural_graphs_ok: {summary['pre_validation']['all_structural_graphs_ok']}")
    if summary["pre_validation"]["graph_builder_agreement"] is not None:
        print(f"graph_builder_agreement: {summary['pre_validation']['graph_builder_agreement']}")
        print(f"cutoff: {summary['pre_validation']['cutoff']}")
        print(f"distnnp_edge_count: {summary['pre_validation']['distnnp_edge_count']}")
        print(f"standalone_mace_edge_count: {summary['pre_validation']['standalone_mace_edge_count']}")
        print(f"distnnp_reverse_edge_symmetry: {summary['pre_validation']['distnnp_reverse_edge_symmetry']}")
        print(f"standalone_mace_reverse_edge_symmetry: {summary['pre_validation']['standalone_mace_reverse_edge_symmetry']}")
    if summary["pre_validation"]["graph_builder_agreement"] is False:
        print(f"distnnp_only_edge_preview: {summary['pre_validation']['distnnp_only_edge_preview']}")
        print(f"mace_only_edge_preview: {summary['pre_validation']['mace_only_edge_preview']}")
    if summary["pre_validation"]["distnnp_reverse_edge_symmetry"] is False:
        print(f"distnnp_missing_reverse_preview: {summary['pre_validation']['distnnp_missing_reverse_preview']}")
    if summary["pre_validation"]["standalone_mace_reverse_edge_symmetry"] is False:
        print(f"standalone_mace_missing_reverse_preview: {summary['pre_validation']['standalone_mace_missing_reverse_preview']}")
    for item in summary["pre_validation"]["rank_graph_verification"]:
        print(
            f"rank {item.get('rank', '?')}: {item['structural_classification']}, "
            f"expected_edges={item['expected_owned_edge_count']}, "
            f"actual_edges={item['actual_owned_edge_count']}, "
            f"missing_halo={item['missing_required_halo_global_indices']}"
        )
    print("\n" + "-" * 60 + "\nParity\n" + "-" * 60)
    for key, value in summary["parity"].items():
        print(f"{key}: {value}")
    print("\n" + "-" * 60 + "\nPer-Rank\n" + "-" * 60)
    for record in summary["ranks"]:
        executed_layers = [
            layer_record["layer_index"]
            for layer_record in record.get("layer_records", [])
            if layer_record.get("executed")
        ]
        print(
            f"rank {record['rank']}: local={record['n_local']}, halo={record['n_halo']}, "
            f"owned_edges={record['owned_destination_edge_count']}, "
            f"feature_exchange_used_ipc={record['feature_exchange_used_ipc']}, "
            f"executed_exchange_layers={executed_layers}"
        )
    if "benchmark" in summary:
        benchmark = summary["benchmark"]
        print("\n" + "-" * 60 + "\nPerformance Benchmark\n" + "-" * 60)
        print(f"scope: {benchmark['scope']}")
        print(f"standalone_baseline: {benchmark['standalone_baseline']}")
        print(f"device_name: {benchmark['device_name']}")
        print(f"timing_method: {benchmark['timing_method']}")
        print(f"world_size: {benchmark['world_size']}")
        print(f"warmup_steps: {benchmark['warmup_steps']}")
        print(f"iterations_per_sample: {benchmark['iterations_per_sample']}")
        print(f"num_samples: {benchmark['num_samples']}")
        print(
            "distributed_median_ms_per_step: "
            f"{benchmark['distributed']['median_ms_per_step']}"
        )
        print(
            "standalone_median_ms_per_step: "
            f"{benchmark['standalone']['median_ms_per_step']}"
        )
        print(f"distributed_p95_ms_per_step: {benchmark['distributed']['p95_ms_per_step']}")
        print(f"standalone_p95_ms_per_step: {benchmark['standalone']['p95_ms_per_step']}")
        print(f"distributed_stdev_ms_per_step: {benchmark['distributed']['stdev_ms_per_step']}")
        print(f"standalone_stdev_ms_per_step: {benchmark['standalone']['stdev_ms_per_step']}")
        print(
            "distributed_coefficient_of_variation_percent: "
            f"{benchmark['distributed']['coefficient_of_variation_percent']}"
        )
        print(
            "standalone_coefficient_of_variation_percent: "
            f"{benchmark['standalone']['coefficient_of_variation_percent']}"
        )
        print(
            "distributed_atoms_per_second: "
            f"{benchmark['distributed']['atoms_per_second']}"
        )
        print(
            "standalone_atoms_per_second: "
            f"{benchmark['standalone']['atoms_per_second']}"
        )
        print(
            "speedup_standalone_over_distributed: "
            f"{benchmark['speedup_standalone_over_distributed']}"
        )
        print(f"parallel_efficiency: {benchmark['parallel_efficiency']}")


def prepare_experiment(args: argparse.Namespace) -> PreparedExperiment:
    if not HAS_MACE:
        raise ImportError("This experiment requires MACE.")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("Launch this test with torchrun and at least two ranks.")
    torch_env = TorchDistEnv(is_distributed=True)
    if dist.get_world_size() < 2:
        raise RuntimeError("The distributed integration test requires at least two ranks.")
    device = torch_env.device
    raw = load_xyz(args.xyz_path) if args.xyz_path else BUILTIN_SYSTEMS[args.builtin_system]()
    model = load_raw_mace_model(args.model_family, args.model, str(device)).to(device).eval()
    contract = summarize_model_contract(model)
    cutoff = float(model.r_max.detach().cpu())
    if not args.disable_nonperiodic_canonicalization:
        padding = args.nonperiodic_padding
        if padding is None:
            padding = cutoff + args.ghost_cutoff_buffer
        raw = canonicalize_raw_structure(raw, padding)
    state, sim = build_state(raw, cutoff, args.ghost_cutoff_buffer, torch_env)
    adapter = DistMACETopDownAdapter(
        model,
        head_name=args.head_name or None,
        debug_tracing=args.debug_exchange,
        debug_prefix=args.debug_log_prefix,
    )
    adapter.initialize_buffers(state.local_partition_info, device, dist.get_world_size())
    reference_model = None
    if dist.get_rank() == 0 and args.validation_mode == "full":
        if args.comparison_model:
            reference_model = load_raw_mace_model(
                args.model_family, args.comparison_model, str(device)
            ).to(device=device, dtype=COMPUTE_DTYPE).eval()
        else:
            reference_model = model
    return PreparedExperiment(
        raw=raw,
        sim=sim,
        state=state,
        model=model,
        contract=contract,
        adapter=adapter,
        device=device,
        reference_model=reference_model,
    )


def run_distributed_adapter_step(
    prepared: PreparedExperiment,
) -> Tuple[float, torch.Tensor, List[Dict[str, Any]]]:
    try:
        energy, forces = prepared.adapter(prepared.state)
        record = prepared.adapter.last_observation | {
            "returned_energy": float(energy.cpu()),
            "returned_force_shape": list(forces.shape),
            "model_contract": prepared.contract,
        }
        record["graph_verification"]["rank"] = dist.get_rank()
        records = gather_records(record)
        dist.barrier()
        return float(energy.cpu()), forces, records
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if dist.is_initialized() and hasattr(dist, "abort"):
            dist.abort()
        raise


def build_validation_summary(
    prepared: PreparedExperiment,
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    summary = None
    classification = None
    try:
        if dist.get_rank() == 0:
            if args.validation_mode == "full":
                reference = monolithic_references(
                    prepared.raw,
                    prepared.sim,
                    args,
                    prepared.device,
                    model=prepared.reference_model,
                )
                summary = build_summary(prepared.raw, records, reference, args)
            else:
                summary = build_minimal_summary(prepared.raw, records)
            summary["model_contract"] = prepared.contract
            classification = summary["mismatch_classification"]
        return summary, classification
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if dist.is_initialized() and hasattr(dist, "abort"):
            dist.abort()
        raise


def run_experiment(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    prepared = prepare_experiment(args)
    _, _, records = run_distributed_adapter_step(prepared)
    summary, classification = build_validation_summary(prepared, records, args)
    classification_message = [classification]
    dist.broadcast_object_list(classification_message, src=0)
    classification = classification_message[0]
    if args.benchmark:
        if args.validation_mode != "full":
            raise RuntimeError("Benchmark requires --validation-mode full.")
        if classification != "parity_ok":
            raise RuntimeError(
                f"Benchmark requires parity_ok pre-validation, got {classification}."
            )
        benchmark = benchmark_speedup(
            prepared.raw,
            prepared.sim,
            prepared.state,
            prepared.adapter,
            args,
            prepared.device,
        )
        if dist.get_rank() == 0:
            assert summary is not None and benchmark is not None
            summary["benchmark"] = benchmark
    if dist.get_rank() != 0:
        return None
    assert summary is not None
    print_summary(summary)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.strict_parity:
        expected = "parity_ok" if args.validation_mode == "full" else "minimal_ok"
        assert summary["mismatch_classification"] == expected, summary["mismatch_classification"]
    return summary


def test_distmace_graph_authority_and_parity() -> None:
    """Pytest entry point; use torchrun for the required distributed environment."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    args = build_parser().parse_args([])
    args.strict_parity = True
    run_experiment(args)


def main() -> None:
    try:
        run_experiment(build_parser().parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
