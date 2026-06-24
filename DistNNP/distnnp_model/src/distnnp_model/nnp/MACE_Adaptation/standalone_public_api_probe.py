"""Inspect the public MACE Python API from outside the repository.

Requirements to install in the target environment
-------------------------------------------------
1. A Python environment with ``mace-torch`` installed.
   Example:
       pip install mace-torch

2. ``ase`` installed if you want to use ``mace.calculators`` with ``ase.Atoms``.
   This is normally pulled in by ``mace-torch``, but it is listed here explicitly
   because calculator-based usage depends on it conceptually.

3. ``torch`` must be available and compatible with the installed ``mace-torch``.
   This is also a runtime dependency of MACE itself.

4. A model checkpoint if you want to run real inference.
   For public foundation models, the convenience helpers in ``mace.calculators``
   can download or locate them. For custom inference, you need a local ``.model``
   file path.

Why this file exists
--------------------
This file is meant to answer a portability question: if copied into another
project, such as DistNNP, what parts of MACE are visible through the installed
Python package surface?

It intentionally imports only public package paths:
    - mace
    - mace.calculators
    - mace.modules
    - mace.data
    - mace.tools

It does not rely on private relative imports into the repository checkout.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List


PUBLIC_MODULES = [
    "mace",
    "mace.calculators",
    "mace.modules",
    "mace.data",
    "mace.tools",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the public Python API exposed by an installed MACE package."
        )
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the discovered public API summary as JSON.",
    )
    return parser


def public_names(module: Any) -> List[str]:
    exports = getattr(module, "__all__", None)
    if exports is not None:
        return list(exports)
    return sorted(name for name in dir(module) if not name.startswith("_"))


def probe_module(module_name: str) -> Dict[str, Any]:
    module = importlib.import_module(module_name)
    payload: Dict[str, Any] = {
        "module": module_name,
        "file": getattr(module, "__file__", None),
        "public_names": public_names(module),
    }
    return payload


def describe_mace_classes() -> Dict[str, Any]:
    modules_pkg = importlib.import_module("mace.modules")
    calculators_pkg = importlib.import_module("mace.calculators")
    data_pkg = importlib.import_module("mace.data")

    mace_cls = getattr(modules_pkg, "MACE", None)
    scaleshift_cls = getattr(modules_pkg, "ScaleShiftMACE", None)
    calc_cls = getattr(calculators_pkg, "MACECalculator", None)

    payload: Dict[str, Any] = {
        "MACE_exposed": mace_cls is not None,
        "ScaleShiftMACE_exposed": scaleshift_cls is not None,
        "MACECalculator_exposed": calc_cls is not None,
        "AtomicData_exposed": hasattr(data_pkg, "AtomicData"),
        "config_from_atoms_exposed": hasattr(data_pkg, "config_from_atoms"),
        "KeySpecification_exposed": hasattr(data_pkg, "KeySpecification"),
    }

    if mace_cls is not None:
        payload["MACE_qualname"] = f"{mace_cls.__module__}.{mace_cls.__name__}"
        payload["MACE_forward_signature"] = str(inspect.signature(mace_cls.forward))
    if scaleshift_cls is not None:
        payload["ScaleShiftMACE_qualname"] = (
            f"{scaleshift_cls.__module__}.{scaleshift_cls.__name__}"
        )
        payload["ScaleShiftMACE_forward_signature"] = str(
            inspect.signature(scaleshift_cls.forward)
        )
    if calc_cls is not None:
        payload["MACECalculator_qualname"] = (
            f"{calc_cls.__module__}.{calc_cls.__name__}"
        )
        payload["MACECalculator_calculate_signature"] = str(
            inspect.signature(calc_cls.calculate)
        )
        payload["MACECalculator_descriptor_signature"] = str(
            inspect.signature(calc_cls.get_descriptors)
        )
    return payload


def print_summary(results: Dict[str, Any]) -> None:
    print("== Public MACE API Probe ==")
    print()
    print("== Top-Level Package ==")
    print(f"mace.__file__: {results['mace_root']}")
    print()
    print("== Importable Public Modules ==")
    for payload in results["modules"]:
        print(f"{payload['module']}:")
        print(f"  file: {payload['file']}")
        print(f"  public_names_sample: {payload['public_names'][:20]}")
    print()
    print("== Model And Calculator Exposure ==")
    exposure = results["class_exposure"]
    for key, value in exposure.items():
        print(f"{key}: {value}")
    print()
    print("== Interpretation ==")
    print(
        "1. The top-level 'mace' package itself is intentionally minimal, but "
        "subpackages such as 'mace.calculators', 'mace.modules', 'mace.data', "
        "and 'mace.tools' are importable from an installed package."
    )
    print(
        "2. The model architecture classes are publicly reachable through "
        "'mace.modules', including 'MACE' and 'ScaleShiftMACE'."
    )
    print(
        "3. The calculator wrapper is publicly reachable through "
        "'mace.calculators.MACECalculator' and the foundation-model helpers."
    )
    print(
        "4. If you want to bypass ASE, the public 'mace.data' package exposes "
        "'config_from_atoms', 'AtomicData', and related graph-building helpers. "
        "That means direct model usage is possible, but you must then manage "
        "graph construction and batching yourself."
    )


def main() -> None:
    args = build_parser().parse_args()

    mace_root = importlib.import_module("mace").__file__
    modules = [probe_module(module_name) for module_name in PUBLIC_MODULES]
    class_exposure = describe_mace_classes()

    results = {
        "mace_root": mace_root,
        "modules": modules,
        "class_exposure": class_exposure,
    }

    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
