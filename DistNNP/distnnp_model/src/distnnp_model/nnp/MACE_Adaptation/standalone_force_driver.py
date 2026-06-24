"""Standalone force driver built only on MACE's public calculator API.

This file is designed as a portability prototype:

- It avoids importing internal MACE modules such as ``mace.data`` or
  ``mace.modules``.
- It relies only on the public ``mace.calculators`` API plus ASE.
- It can therefore be copied into another Python project, such as DistNNP,
  provided that ``mace-torch`` and ``ase`` are installed in that environment.

Typical usage from the repository root:

    python -m mace.calculators.standalone_force_driver \
        --structure ./example.xyz \
        --model-family mp \
        --model medium-mpa-0 \
        --device cpu

Or with a local checkpoint path:

    python -m mace.calculators.standalone_force_driver \
        --structure ./example.xyz \
        --model-family custom \
        --model ./my_model.model \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read

from mace.calculators import MACECalculator, mace_mp, mace_off, mace_omol, mace_polar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute energies and forces for one atomic configuration using the "
            "public MACE calculator interface only."
        )
    )
    parser.add_argument(
        "--structure",
        help=(
            "Path to an ASE-readable structure file. Mutually exclusive with "
            "--system."
        ),
    )
    parser.add_argument(
        "--system",
        help=(
            "Small built-in test structure. Use molecule:H2O or bulk:Si. "
            "Mutually exclusive with --structure."
        ),
    )
    parser.add_argument(
        "--index",
        default="0",
        help=(
            "ASE index for selecting one structure from a file. "
            "Examples: 0, -1, ':'. This driver expects the selection to resolve "
            "to exactly one structure."
        ),
    )
    parser.add_argument(
        "--model-family",
        choices=["mp", "off", "omol", "polar", "custom"],
        default="mp",
        help=(
            "Public MACE entrypoint to use. "
            "'custom' uses MACECalculator directly with a local checkpoint path."
        ),
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help=(
            "Foundation-model key or local checkpoint path. "
            "For custom mode this must be a local .model path."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device passed into the calculator, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Default dtype passed into the MACE calculator.",
    )
    parser.add_argument(
        "--compute-stress",
        action="store_true",
        help="Also compute and print stress.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save a JSON summary of the computed results.",
    )
    return parser


def load_atoms(args: argparse.Namespace) -> Atoms:
    if bool(args.structure) == bool(args.system):
        raise ValueError("Provide exactly one of --structure or --system.")

    if args.system:
        kind, _, value = args.system.partition(":")
        if not value:
            raise ValueError(
                "System spec must look like molecule:H2O or bulk:Si."
            )
        if kind == "molecule":
            return molecule(value)
        if kind == "bulk":
            return bulk(value)
        raise ValueError(f"Unsupported system kind: {kind}")

    loaded = read(args.structure, index=args.index)
    if isinstance(loaded, list):
        if len(loaded) != 1:
            raise ValueError(
                f"Selection resolved to {len(loaded)} structures; expected exactly 1."
            )
        return loaded[0]
    return loaded


def build_calculator(args: argparse.Namespace):
    common_kwargs = {
        "device": args.device,
        "default_dtype": args.dtype,
    }

    if args.model_family == "mp":
        return mace_mp(model=args.model, dispersion=False, **common_kwargs)
    if args.model_family == "off":
        return mace_off(model=args.model, **common_kwargs)
    if args.model_family == "omol":
        return mace_omol(model=args.model, **common_kwargs)
    if args.model_family == "polar":
        return mace_polar(model=args.model, **common_kwargs)
    if args.model_family == "custom":
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Custom model path does not exist: {model_path}"
            )
        return MACECalculator(model_paths=str(model_path), **common_kwargs)
    raise ValueError(f"Unsupported model family: {args.model_family}")


def result_payload(atoms: Atoms, calc, compute_stress: bool) -> Dict[str, Any]:
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    payload: Dict[str, Any] = {
        "num_atoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(),
        "energy_eV": float(energy),
        "forces_eV_per_A": forces.tolist(),
        "force_norms_eV_per_A": np.linalg.norm(forces, axis=1).tolist(),
        "implemented_properties": list(getattr(calc, "implemented_properties", [])),
        "result_keys": sorted(calc.results.keys()),
    }

    if "node_energy" in calc.results:
        payload["interaction_node_energy_eV"] = np.asarray(
            calc.results["node_energy"]
        ).tolist()
    if "energies" in calc.results:
        payload["total_node_energy_eV"] = np.asarray(calc.results["energies"]).tolist()
    if compute_stress:
        payload["stress_voigt_eV_per_A3"] = atoms.get_stress().tolist()
    return payload


def print_summary(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    print("== Standalone MACE Force Driver ==")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"default_dtype: {args.dtype}")
    print()
    print("== Structure ==")
    print(f"num_atoms: {payload['num_atoms']}")
    print(f"symbols: {payload['symbols']}")
    print()
    print("== Energy and Forces ==")
    print(f"energy_eV: {payload['energy_eV']}")
    print("forces_eV_per_A:")
    print(np.asarray(payload["forces_eV_per_A"]))
    print(f"force_norms_eV_per_A: {payload['force_norms_eV_per_A']}")
    if "stress_voigt_eV_per_A3" in payload:
        print(f"stress_voigt_eV_per_A3: {payload['stress_voigt_eV_per_A3']}")
    print()
    print("== Calculator Metadata ==")
    print(f"implemented_properties: {payload['implemented_properties']}")
    print(f"result_keys: {payload['result_keys']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    atoms = load_atoms(args)
    calc = build_calculator(args)
    atoms.calc = calc

    payload = result_payload(atoms, calc, compute_stress=args.compute_stress)
    print_summary(payload, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
