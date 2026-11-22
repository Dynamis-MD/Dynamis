#!/usr/bin/env python3
"""DistNNP ORB / MACE end-to-end MD benchmark driver.

Launched one rank per GPU via torchrun:

    torchrun --standalone --nproc_per_node=<N> run_distnnp_nnp.py \
        --engine distnnp-orb  --input <xyz> --output-dir <dir> [flags]
    torchrun --standalone --nproc_per_node=<N> run_distnnp_nnp.py \
        --engine distnnp-mace --input <xyz> --output-dir <dir> [flags]

Only the model factory lives here. The spatial decomposition, Langevin
thermostat, warmup/timed loop, OOM capture and JSON schema all come from
distnnp_md.run_md_benchmark -- the same code path run_distnnp_uma.py uses, so
an ORB or MACE scaling curve is directly comparable in methodology to the UMA
one already in results/.

Environments: ORB and MACE cannot share one venv (mace-torch pins e3nn==0.4.4,
fairchem needs e3nn>=0.5), so scripts/strong_scaling_nnp.sbatch selects the
interpreter per engine. Both venvs editable-install this same checkout.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common import parse_bench_args
from distnnp_md import ModelSpec, run_md_benchmark

# `--model` defaults to the UMA checkpoint in bench_common (shared parser). When
# an ORB/MACE cell leaves it untouched, substitute that backend's own default
# rather than trying to load "uma-s-1p2" out of the ORB registry.
_PARSER_DEFAULT_MODEL = "uma-s-1p2"
_DEFAULT_MODEL = {
    "distnnp-orb": "orb_v3_conservative_inf_omat",
    "distnnp-mace": "medium-mpa-0",
}


def _resolve_model_name(args) -> str:
    if args.model != _PARSER_DEFAULT_MODEL:
        return args.model
    return _DEFAULT_MODEL[args.engine]


def build_orb_model(*, device, args, exchanger_cls, rank) -> ModelSpec:
    from distnnp_model.nnp import ORBModel

    model_name = _resolve_model_name(args)
    # ORBModel has no default precision. "float32-high" enables TF32 matmuls,
    # the ORB-side analogue of the UMA `k1` config the exp0 matrix uses
    # (tf32 on); "float32-highest" is the bit-tight setting the ORB parity test
    # runs and is materially slower.
    model = ORBModel(
        model_name=model_name,
        precision=args.orb_precision,
        device=device,
    )
    if rank == 0:
        print(
            f"ORBModel[{model_name}] precision={args.orb_precision} "
            f"radius={model.model_radius} max_neighbors={model.model_max_num_neighbors} "
            f"mp_steps={model.num_mp_steps} hidden={model.hidden_dim}"
        )
        # ORBModel builds its graph from sim_config.cutoff, so a --cutoff-angstrom
        # below the model's own radius silently truncates the receptive field.
        if args.cutoff_angstrom < model.model_radius:
            print(
                f"WARNING: --cutoff-angstrom={args.cutoff_angstrom} is below ORB's "
                f"model radius {model.model_radius}; edges are being truncated."
            )
    return ModelSpec(
        model=model,
        model_name=model_name,
        task="omat",
        settings={
            "backend": "orb",
            "precision": args.orb_precision,
            "model_radius": float(model.model_radius),
            "max_num_neighbors": int(model.model_max_num_neighbors),
            "num_mp_steps": int(model.num_mp_steps),
        },
    )


def build_mace_model(*, device, args, exchanger_cls, rank) -> ModelSpec:
    from distnnp_model.nnp import MACEModel
    from distnnp_model.nnp.mace import load_raw_mace_model

    model_name = _resolve_model_name(args)
    raw = load_raw_mace_model(
        model_family=args.mace_model_family,
        model=model_name,
        device=device.type,
    )
    model = MACEModel(
        raw.to(device=device).eval(),
        head_name=args.mace_head or None,
    )
    if rank == 0:
        print(
            f"MACEModel[{args.mace_model_family}/{model_name}] "
            f"head={model.head_name} r_max={model.cutoff}"
        )
        # MACEModel builds its graph at the checkpoint's own r_max, so the halo
        # the Simulator ships (--cutoff-angstrom + --ghost-buffer-angstrom) must
        # cover it or ghost atoms MACE needs will simply be absent.
        if args.cutoff_angstrom < model.cutoff:
            print(
                f"WARNING: --cutoff-angstrom={args.cutoff_angstrom} is below MACE's "
                f"r_max {model.cutoff}; the halo will not cover the model's graph."
            )
    return ModelSpec(
        model=model,
        model_name=model_name,
        task=args.mace_model_family,
        settings={
            "backend": "mace",
            "model_family": args.mace_model_family,
            "head": model.head_name,
            "r_max": float(model.cutoff),
        },
    )


_FACTORY = {
    "distnnp-orb": build_orb_model,
    "distnnp-mace": build_mace_model,
}


def main() -> None:
    args = parse_bench_args()
    if args.engine not in _FACTORY:
        raise SystemExit(
            f"run_distnnp_nnp.py handles {sorted(_FACTORY)}; got --engine {args.engine}. "
            "UMA cells go through run_distnnp_uma.py."
        )
    run_md_benchmark(args, _FACTORY[args.engine])


if __name__ == "__main__":
    main()
