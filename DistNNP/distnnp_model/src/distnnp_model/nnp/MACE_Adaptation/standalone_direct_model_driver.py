"""Direct model-level MACE driver using public MACE inputs instead of ASE.

Requirements to install in the target environment
-------------------------------------------------
1. ``mace-torch`` installed in the Python environment.
2. ``torch`` available and compatible with the chosen MACE package.
3. A pretrained MACE checkpoint or a foundation-model key.
4. If you want to use the convenience foundation-model loaders, network access
   may be needed unless the checkpoint is already cached or bundled locally.

What this demonstrates
----------------------
This script bypasses ``ase.Atoms`` and ``MACECalculator`` for inference.
Instead, it uses public MACE package objects directly:

- ``mace.calculators.mace_mp(..., return_raw_model=True)`` or related helpers
- ``mace.data.Configuration``
- ``mace.data.AtomicData.from_config(...)``
- ``mace.tools.AtomicNumberTable``
- ``mace.tools.torch_geometric.batch.Batch``

That makes it a better prototype for mapping DistNNP-owned state into MACE's
native graph input format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
from mace.data import AtomicData, Configuration
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.batch import Batch


BUILTIN_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "h2o": {
        "atomic_numbers": [8, 1, 1],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.758602, 0.0, 0.504284],
            [-0.758602, 0.0, 0.504284],
        ],
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False, False, False],
    },
    "si_dimer": {
        "atomic_numbers": [14, 14],
        "positions": [
            [0.0, 0.0, 0.0],
            [2.35, 0.0, 0.0],
        ],
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False, False, False],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pretrained MACE model directly on a MACE graph input "
            "dictionary, without using ASE or MACECalculator."
        )
    )
    parser.add_argument(
        "--builtin-system",
        choices=sorted(BUILTIN_SYSTEMS.keys()),
        help="Use a built-in raw species/positions example.",
    )
    parser.add_argument(
        "--species",
        help=(
            "Comma-separated atomic numbers, e.g. '8,1,1'. "
            "Required if --builtin-system is not used."
        ),
    )
    parser.add_argument(
        "--positions-json",
        help=(
            "JSON list of positions in Angstrom, e.g. "
            "'[[0,0,0],[0.75,0,0.5],[-0.75,0,0.5]]'. "
            "Required if --builtin-system is not used."
        ),
    )
    parser.add_argument(
        "--cell-json",
        default="[[0,0,0],[0,0,0],[0,0,0]]",
        help="Optional JSON 3x3 cell matrix.",
    )
    parser.add_argument(
        "--pbc-json",
        default="[false,false,false]",
        help="Optional JSON periodic boundary condition triple.",
    )
    parser.add_argument(
        "--model-family",
        choices=["mp", "off", "omol", "polar"],
        default="mp",
        help="Public MACE foundation-model family to load.",
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help="Foundation-model key or checkpoint path understood by the selected loader.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device for the loaded model.",
    )
    parser.add_argument(
        "--compute-stress",
        action="store_true",
        help="Request stress in the direct model forward pass.",
    )
    parser.add_argument(
        "--compute-hessian",
        action="store_true",
        help="Request Hessian in the direct model forward pass.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save a JSON summary.",
    )
    return parser


def parse_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    if args.builtin_system:
        return BUILTIN_SYSTEMS[args.builtin_system]

    if not args.species or not args.positions_json:
        raise ValueError(
            "Provide --builtin-system or both --species and --positions-json."
        )

    atomic_numbers = [int(token.strip()) for token in args.species.split(",") if token.strip()]
    positions = json.loads(args.positions_json)
    cell = json.loads(args.cell_json)
    pbc = json.loads(args.pbc_json)

    if len(atomic_numbers) != len(positions):
        raise ValueError(
            "Number of atomic numbers must match number of position rows."
        )

    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
    }


def load_raw_model(args: argparse.Namespace):
    common = {"model": args.model, "device": args.device, "return_raw_model": True}
    if args.model_family == "mp":
        return mace_mp(**common)
    if args.model_family == "off":
        return mace_off(**common)
    if args.model_family == "omol":
        return mace_omol(**common)
    if args.model_family == "polar":
        return mace_polar(**common)
    raise ValueError(f"Unsupported model family: {args.model_family}")


def build_configuration(raw_structure: Dict[str, Any]) -> Configuration:
    return Configuration(
        atomic_numbers=np.asarray(raw_structure["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(raw_structure["positions"], dtype=np.float64),
        properties={},
        property_weights={},
        cell=np.asarray(raw_structure["cell"], dtype=np.float64),
        pbc=tuple(bool(x) for x in raw_structure["pbc"]),
        weight=1.0,
        config_type="DirectModelProbe",
        head="Default",
    )


def infer_z_table(model) -> AtomicNumberTable:
    atomic_numbers = [int(z) for z in model.atomic_numbers.detach().cpu().tolist()]
    return AtomicNumberTable(atomic_numbers)


def build_batch(config: Configuration, model, device: str):
    z_table = infer_z_table(model)
    cutoff = float(model.r_max.detach().cpu().item())
    heads = getattr(model, "heads", ["Default"])
    atomic_data = AtomicData.from_config(
        config=config,
        z_table=z_table,
        cutoff=cutoff,
        heads=heads,
    )
    batch = Batch.from_data_list([atomic_data]).to(device)
    return batch


def align_batch_dtype(batch_dict: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    model_dtype = next(model.parameters()).dtype
    aligned = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            aligned[key] = value.to(dtype=model_dtype)
        else:
            aligned[key] = value
    return aligned


def summarize_outputs(raw_structure: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "output_keys": sorted(outputs.keys()),
    }
    for key in [
        "energy",
        "node_energy",
        "forces",
        "stress",
        "virials",
        "hessian",
        "edge_forces",
        "node_feats",
        "contributions",
    ]:
        value = outputs.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            payload[f"{key}_shape"] = list(value.shape)
            if value.ndim == 0:
                payload[key] = value.detach().cpu().item()
            else:
                payload[key] = value.detach().cpu().numpy().tolist()
    return payload


def print_summary(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    print("== Direct Model-Level MACE Driver ==")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print()
    print("== Raw MACE Input ==")
    print(f"atomic_numbers: {payload['atomic_numbers']}")
    print(f"positions_A: {payload['positions_A']}")
    print()
    print("== Direct Forward Outputs ==")
    print(f"output_keys: {payload['output_keys']}")
    if "energy" in payload:
        print(f"energy: {payload['energy']}")
    if "forces_shape" in payload:
        print(f"forces_shape: {payload['forces_shape']}")
        print(np.asarray(payload["forces"]))
    if "stress_shape" in payload:
        print(f"stress_shape: {payload['stress_shape']}")
        print(np.asarray(payload["stress"]))
    if "node_energy_shape" in payload:
        print(f"node_energy_shape: {payload['node_energy_shape']}")
    if "node_feats_shape" in payload:
        print(f"node_feats_shape: {payload['node_feats_shape']}")
    if "contributions_shape" in payload:
        print(f"contributions_shape: {payload['contributions_shape']}")


def main() -> None:
    args = build_parser().parse_args()
    raw_structure = parse_raw_structure(args)

    model = load_raw_model(args)
    model.to(args.device)
    model.eval()

    config = build_configuration(raw_structure)
    batch = build_batch(config, model, args.device)

    #I assume that this data is what I need to collect within DistNNP   
    batch_dict = align_batch_dtype(batch.to_dict(), model)

    outputs = model(
        batch_dict,
        training=False,
        compute_force=True,
        compute_stress=args.compute_stress,
        compute_hessian=args.compute_hessian,
    )

    payload = summarize_outputs(raw_structure, outputs)
    print_summary(payload, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
