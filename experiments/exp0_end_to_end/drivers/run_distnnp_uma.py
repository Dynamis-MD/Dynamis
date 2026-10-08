#!/usr/bin/env python3
"""DistNNP-UMA end-to-end MD benchmark driver.

Launched one rank per GPU via torchrun:

    torchrun --standalone --nproc_per_node=<N> run_distnnp_uma.py \
        --input <xyz> --output-dir <dir> [bench_common flags]

Atoms are spatially partitioned (SFC-Morton) across ranks; only ghost
atoms/features are exchanged per message-passing layer. Timing excludes model
load and the warmup window (see bench_common.TimedLoop).

Only the UMA model construction lives here; the shared MD body (partitioning,
Langevin, warmup/timed loop, OOM capture, JSON schema) is in distnnp_md.py so
every DistNNP backend is benchmarked by identical code.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common import (
    build_inference_settings,
    inference_settings_summary,
    parse_bench_args,
)
from distnnp_md import ModelSpec, run_md_benchmark


def build_uma_model(*, device, args, exchanger_cls, rank) -> ModelSpec:
    from distnnp_model.nnp import UMAModel

    # UMAModel's own default is inference_settings="default" -- fairchem's
    # *most general, least optimised* preset (tf32 off, activation checkpointing
    # on, MOLE unmerged, no compile), which also disqualifies the umas_fast_gpu
    # execution backend. Resolve it here instead, from the same helper the
    # FairChem driver uses, so both engines run identical kernels.
    settings = build_inference_settings(args)
    settings_rec = inference_settings_summary(settings)
    if rank == 0:
        print(f"InferenceSettings[{args.inference_settings}] -> {settings_rec}")

    model = UMAModel(
        args.model,
        dataset=args.task,
        device=device.type,
        inference_settings=settings,
        exchanger=exchanger_cls,
    )
    if rank == 0:
        print(f"halo exchanger: {args.exchanger} ({exchanger_cls.__name__})")
    return ModelSpec(
        model=model,
        model_name=args.model,
        task=args.task,
        settings=settings_rec,
    )


def main() -> None:
    run_md_benchmark(parse_bench_args(), build_uma_model)


if __name__ == "__main__":
    main()
