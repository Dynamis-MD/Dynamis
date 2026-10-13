"""Detached standalone MACE reference writer for MACEModel benchmarks.

This driver avoids Dynamis partitioning and Dynamis graph construction. It
records two monolithic baselines:

1. the public calculator path via ``mace.calculators``
2. a raw ``ScaleShiftMACE`` path with MACE-native graph creation in this file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dynamis_model.nnp.mace import (
    COMPUTE_DTYPE,
    load_raw_mace_model,
    resolve_model_head_name,
)

try:
    from ase import Atoms
    from ase.calculators.calculator import all_changes
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    HAS_STANDALONE_MACE = True
except Exception as exc:  # pragma: no cover - optional dependency
    Atoms = None  # type: ignore[assignment]
    all_changes = None  # type: ignore[assignment]
    ASE_ATOMIC_NUMBERS = {}
    mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None  # type: ignore[assignment]
    HAS_STANDALONE_MACE = False
    _STANDALONE_IMPORT_ERROR = exc
else:
    _STANDALONE_IMPORT_ERROR = None

ARTIFACT_VERSION = 3
TEST_FILE_DIR = Path(__file__).resolve().parent
ELEMENT_TO_Z = dict(ASE_ATOMIC_NUMBERS)


def _require_standalone_mace() -> None:
    if not HAS_STANDALONE_MACE:
        raise ImportError(
            "Standalone MACE reference generation requires ASE and MACE."
        ) from _STANDALONE_IMPORT_ERROR


def natural_chain12() -> Dict[str, Any]:
    carbon = [
        [2.00, 2.00, 2.00],
        [3.52, 2.12, 2.00],
        [5.01, 1.90, 2.18],
        [6.50, 2.08, 1.95],
    ]
    hydrogen = [
        [1.45, 2.92, 2.20],
        [1.52, 1.08, 1.80],
        [3.52, 3.08, 1.55],
        [3.55, 1.22, 2.62],
        [5.00, 2.82, 2.72],
        [5.02, 0.95, 1.58],
        [7.03, 2.96, 2.20],
        [7.02, 1.18, 1.72],
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
    parser.add_argument("--builtin-system", choices=BUILTIN_SYSTEMS, default="natural_chain12")
    parser.add_argument("--xyz-path", type=Path, help="XYZ/extxyz input; overrides --builtin-system.")
    parser.add_argument("--model-family", choices=["mp", "off", "omol", "polar"], default="mp")
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--head-name", default="", help="Optional explicit MACE head name for metadata validation.")
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Enable cuEquivariance for both the public calculator and raw-core reference paths.",
    )
    parser.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True, help="Reference artifact path.")
    parser.add_argument("--benchmark", action="store_true", help="Record standalone timing stats.")
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument("--benchmark-samples", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
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
        f"working directory: {Path.cwd()}\n"
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


def load_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    return load_xyz(args.xyz_path) if args.xyz_path else BUILTIN_SYSTEMS[args.builtin_system]()


def raw_to_atoms(raw: Dict[str, Any]) -> Atoms:
    _require_standalone_mace()
    return Atoms(
        numbers=raw["atomic_numbers"],
        positions=raw["positions"],
        cell=raw["cell"],
        pbc=raw["pbc"],
    )


def load_mace_calculator(model_family: str, model_name: str, device: str, enable_cueq: bool = False):
    _require_standalone_mace()
    loaders = {"mp": mace_mp, "off": mace_off, "omol": mace_omol, "polar": mace_polar}
    if model_family not in loaders:
        raise ValueError(f"Unsupported MACE model family: {model_family}")
    return loaders[model_family](
        model=model_name,
        device=device,
        default_dtype="float32",
        enable_cueq=enable_cueq,
    )


def evaluate_with_calculator(calculator, atoms: Atoms) -> Dict[str, Any]:
    _require_standalone_mace()
    calculator.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
    return {
        "energy": float(calculator.results["energy"]),
        "forces": torch.as_tensor(calculator.results["forces"], dtype=COMPUTE_DTYPE).cpu(),
    }


def _align_batch_dtype(batch_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model_dtype = next(model.parameters()).dtype
    aligned: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            aligned[key] = value.to(dtype=model_dtype)
        else:
            aligned[key] = value
    return aligned


def build_core_model_batch(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    _require_standalone_mace()
    resolved_head = resolve_model_head_name(model, head_name)
    config = Configuration(
        atomic_numbers=np.asarray(raw["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(raw["positions"], dtype=np.float32),
        properties={},
        property_weights={},
        cell=np.asarray(raw["cell"], dtype=np.float32),
        pbc=tuple(raw["pbc"]),
        weight=1.0,
        config_type="dynamis_mace_model_core_model_reference",
        head=resolved_head,
    )
    model_heads = list(getattr(model, "heads", []))
    z_table = AtomicNumberTable([int(value) for value in model.atomic_numbers.detach().cpu().tolist()])
    atomic_data = AtomicData.from_config(
        config=config,
        z_table=z_table,
        cutoff=cutoff,
        heads=model_heads or [resolved_head],
    )
    edge_count = int(atomic_data.edge_index.shape[1])
    batch = Batch.from_data_list([atomic_data]).to(device).to_dict()
    return _align_batch_dtype(batch, model), edge_count


def evaluate_with_core_model(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> Dict[str, Any]:
    batch, edge_count = build_core_model_batch(raw, model, cutoff, device, head_name=head_name)
    out = model(
        batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    return {
        "energy": float(out["energy"].detach().cpu().reshape(())),
        "forces": out["forces"].detach().to(COMPUTE_DTYPE).cpu(),
        "edge_count": edge_count,
    }


def run_core_model_forward(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> None:
    batch, _ = build_core_model_batch(raw, model, cutoff, device, head_name=head_name)
    model(
        batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def benchmark_step_loop(
    step_fn,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
    num_atoms: int,
) -> Dict[str, Any]:
    if warmup < 0 or iterations <= 0 or samples <= 0:
        raise ValueError("Benchmark warmup must be nonnegative; iterations and samples must be positive.")
    for _ in range(warmup):
        step_fn()
    _synchronize(device)

    elapsed_samples: List[float] = []
    for _ in range(samples):
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step_fn()
        _synchronize(device)
        elapsed_samples.append((time.perf_counter() - start) / iterations)
    return _timing_statistics(elapsed_samples, num_atoms)


def _reference_signature(
    raw: Dict[str, Any],
    model_family: str,
    model_name: str,
    head_name: str,
    enable_cueq: bool,
) -> str:
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
        "enable_cueq": enable_cueq,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_artifact_metadata(
    raw: Dict[str, Any],
    args: argparse.Namespace,
    cutoff: float,
    head_name: str,
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "system_name": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "num_atoms": len(raw["atomic_numbers"]),
        "model_family": args.model_family,
        "model_name": args.model,
        "head_name": head_name,
        "enable_cueq": args.enable_cueq,
        "cutoff": cutoff,
        "signature": _reference_signature(raw, args.model_family, args.model, head_name, args.enable_cueq),
    }


def generate_reference_artifact(
    raw: Dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    benchmark: bool,
) -> Dict[str, Any]:
    _require_standalone_mace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = load_raw_mace_model(
        args.model_family,
        args.model,
        str(device),
        enable_cueq=args.enable_cueq,
    ).to(
        device=device,
        dtype=COMPUTE_DTYPE,
    ).eval()
    head_name = resolve_model_head_name(raw_model, args.head_name or None)
    cutoff = float(raw_model.r_max.detach().cpu())
    calculator = load_mace_calculator(
        args.model_family,
        args.model,
        str(device),
        enable_cueq=args.enable_cueq,
    )
    atoms = raw_to_atoms(raw)

    calculator_reference = evaluate_with_calculator(calculator, atoms)
    core_model_reference = evaluate_with_core_model(raw, raw_model, cutoff, device, head_name=head_name)
    calc_core_energy_delta = calculator_reference["energy"] - core_model_reference["energy"]
    calc_core_force_delta = calculator_reference["forces"] - core_model_reference["forces"]

    calculator_benchmark = None
    core_model_benchmark = None
    if benchmark:
        calculator_benchmark = {
            "scope": (
                "single-GPU standalone MACE calculator path; includes MACE internal graph construction, "
                "calculator evaluation, energy extraction, and force extraction; excludes input parsing and model loading"
            ),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "warmup_steps": args.benchmark_warmup,
            "iterations_per_sample": args.benchmark_iterations,
            "num_samples": args.benchmark_samples,
            "timing_method": "perf_counter with device synchronization",
            "timings": benchmark_step_loop(
                lambda: evaluate_with_calculator(calculator, atoms),
                device,
                args.benchmark_warmup,
                args.benchmark_iterations,
                args.benchmark_samples,
                len(raw["atomic_numbers"]),
            ),
        }
        core_model_benchmark = {
            "scope": (
                "single-GPU raw ScaleShiftMACE path; includes MACE-native Configuration creation, "
                "AtomicData.from_config graph construction, Batch.from_data_list packaging, and raw model forward "
                "with force computation; excludes ASE calculator result packaging, host-side output extraction, and model loading"
            ),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "warmup_steps": args.benchmark_warmup,
            "iterations_per_sample": args.benchmark_iterations,
            "num_samples": args.benchmark_samples,
            "timing_method": "perf_counter with device synchronization",
            "timings": benchmark_step_loop(
                lambda: run_core_model_forward(raw, raw_model, cutoff, device, head_name=head_name),
                device,
                args.benchmark_warmup,
                args.benchmark_iterations,
                args.benchmark_samples,
                len(raw["atomic_numbers"]),
            ),
        }

    artifact = {
        "metadata": reference_artifact_metadata(raw, args, cutoff, head_name),
        "reference": {
            "calculator_energy": calculator_reference["energy"],
            "calculator_forces": calculator_reference["forces"],
            "core_model_energy": core_model_reference["energy"],
            "core_model_forces": core_model_reference["forces"],
            "core_model_edge_count": core_model_reference["edge_count"],
            "calculator_minus_core_model_energy": calc_core_energy_delta,
            "calculator_minus_core_model_force_mae": float(calc_core_force_delta.abs().mean()),
            "calculator_minus_core_model_force_max_abs": float(calc_core_force_delta.abs().max()),
        },
        "calculator_benchmark": calculator_benchmark,
        "core_model_benchmark": core_model_benchmark,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return artifact


def print_summary(artifact: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Standalone MACE Reference Artifact")
    print("=" * 60)
    print(f"system: {artifact['metadata']['system_name']}")
    print(f"source_path: {artifact['metadata']['source_path']}")
    print(f"num_atoms: {artifact['metadata']['num_atoms']}")
    print(f"model_family: {artifact['metadata']['model_family']}")
    print(f"model_name: {artifact['metadata']['model_name']}")
    print(f"head_name: {artifact['metadata']['head_name']}")
    print(f"enable_cueq: {artifact['metadata']['enable_cueq']}")
    print(f"cutoff: {artifact['metadata']['cutoff']}")
    print(f"artifact_version: {artifact['metadata']['artifact_version']}")
    print(f"calculator_energy: {artifact['reference']['calculator_energy']}")
    print(f"core_model_energy: {artifact['reference']['core_model_energy']}")
    print(f"calculator_minus_core_model_energy: {artifact['reference']['calculator_minus_core_model_energy']}")
    print(
        "calculator_minus_core_model_force_mae: "
        f"{artifact['reference']['calculator_minus_core_model_force_mae']}"
    )
    if artifact["calculator_benchmark"] is not None:
        calc = artifact["calculator_benchmark"]
        core = artifact["core_model_benchmark"]
        print("\n" + "-" * 60 + "\nPerformance Benchmark\n" + "-" * 60)
        print(f"calculator_scope: {calc['scope']}")
        print(f"core_model_scope: {core['scope']}")
        print(f"device_name: {calc['device_name']}")
        print(f"timing_method: {calc['timing_method']}")
        print(f"warmup_steps: {calc['warmup_steps']}")
        print(f"iterations_per_sample: {calc['iterations_per_sample']}")
        print(f"num_samples: {calc['num_samples']}")
        print(f"calculator_median_ms_per_step: {calc['timings']['median_ms_per_step']}")
        print(f"core_model_median_ms_per_step: {core['timings']['median_ms_per_step']}")


def main() -> None:
    args = build_parser().parse_args()
    raw = load_raw_structure(args)
    artifact = generate_reference_artifact(raw, args, args.output, benchmark=args.benchmark)
    print_summary(artifact)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
