#!/usr/bin/env python3
"""FairChem-UMA end-to-end MD benchmark driver (primary baseline).

Single process + Ray graph-parallel workers:

    python run_fairchem_uma.py --input <xyz> --workers <N> --output-dir <dir>

`--workers 1`  -> plain MLIPPredictUnit (no Ray, no graph parallel), the 1-GPU point.
`--workers N`  -> ParallelMLIPPredictUnit (Ray + ALLGATHER/INDEX_SPLIT graph parallel).

All per-step trajectory I/O from `experiments/run_md.py` is stripped: it
dominates wall time at >=256k atoms and is irrelevant to the throughput claim.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ase
import ase.io
import ase.md.langevin
import ase.md.velocitydistribution
import numpy as np
import torch
from ase import units

from bench_common import (
    BenchRecord,
    RunOutcome,
    TimedLoop,
    capture_oom,
    deadline,
    parse_bench_args,
    peak_mem_mb,
    reset_peak_mem,
    write_result,
)


def _build_predictor(args):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

    # Match DistNNP's `UMAModel` exactly: it builds the predictor with
    # inference_settings="default" (activation_checkpointing=True, merge_mole=False,
    # tf32=True). Using turbo() here instead (activation_checkpointing=False) gave
    # the baseline a memory handicap DistNNP never pays -> FairChem OOM'd at 16k
    # atoms on a 40 GB A100 while DistNNP ran 128k. The only difference we want to
    # measure is the distribution strategy, not the checkpointing policy.
    settings = inference_settings_default()
    # merge_mole=True keeps a persistent NCCL broadcast loop alive in the Ray
    # workers; on this cluster rank0's forward and the workers' broadcast loop
    # desync on the first predict -> 600 s collective timeout -> SIGABRT. default()
    # already sets merge_mole=False (the stable re-dispatch path); --merge-mole 1
    # can force the faster-but-deadlock-prone loop back on.
    settings.merge_mole = bool(args.merge_mole)
    settings.compile = False  # match run_md.py; compile adds minutes of warmup

    predictor = get_predict_unit(
        args.model,
        device="cuda",
        inference_settings=settings,
        workers=args.workers,
    )
    calc = FAIRChemCalculator(predictor, task_name=args.task)
    return predictor, calc


def _worker_peaks(predictor, world_size: int):
    """Best-effort peak memory (MB) per graph-parallel rank.

    rank 0 runs in-process; ranks >=1 are Ray actors. If the remote probe is
    unavailable / brittle we return None and let the caller record only the
    rank-0 peak.
    """
    rank0 = peak_mem_mb()
    if world_size == 1:
        return [rank0]
    try:
        import ray

        workers = getattr(predictor, "workers", None)
        if not workers:
            return None

        def _actor_peak(self):  # runs inside the Ray actor
            import torch as _t

            return _t.cuda.max_memory_allocated() / (1024.0**2)

        remote_peaks = ray.get(
            [w.__ray_call__.remote(_actor_peak) for w in workers]
        )
        return [rank0] + [float(x) for x in remote_peaks]
    except Exception as e:  # noqa: BLE001
        print(f"worker peak-mem probe failed ({e!r}); recording rank-0 peak only")
        return None


def _temperature(atoms) -> float:
    return float(atoms.get_temperature())


def main() -> None:
    args = parse_bench_args()
    os.makedirs(args.output_dir, exist_ok=True)

    world_size = max(1, args.workers)
    gpus_per_node = torch.cuda.device_count() or 1

    atoms = ase.io.read(args.input)
    atoms.set_pbc(True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    num_atoms = len(atoms)
    box = float(atoms.cell.lengths()[0])

    ase.md.velocitydistribution.MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.temperature_k
    )

    state: dict = {}

    # --- phase 1: feasibility. Build the predictor + run ONE step. This is
    # where an OOM lands (as a Ray hang, hence the watchdog). If it fails there
    # is no point timing anything.
    def _probe():
        with deadline(args.build_timeout_s, "fairchem build + feasibility step"):
            predictor, calc = _build_predictor(args)
            state["predictor"] = predictor
            atoms.calc = calc
            dyn = ase.md.langevin.Langevin(
                atoms,
                args.timestep_fs * units.fs,
                temperature_K=args.temperature_k,
                friction=args.friction_inv_fs / units.fs,
            )
            dyn.run(1)
        state["dyn"] = dyn

    probe: RunOutcome = capture_oom(_probe)

    # memory verdict comes from the probe, while the workers are still alive
    peaks = None
    peak_max = peak_mem_mb()
    if probe.status == "ok" and "predictor" in state:
        peaks = _worker_peaks(state["predictor"], world_size)
        if peaks:
            peak_max = max(peaks)

    # --- phase 2: timing (only if the system provably fits)
    loop = None
    timing = probe
    if probe.status == "ok":
        def _time():
            dyn = state["dyn"]
            for _ in range(max(0, args.warmup_steps - 1)):  # 1 step already done
                with deadline(args.step_timeout_s, "fairchem warmup step"):
                    dyn.run(1)
            reset_peak_mem()
            return TimedLoop(
                lambda _i: dyn.run(1),
                0,
                args.timed_steps,
                args.timestep_fs,
                step_timeout_s=args.step_timeout_s,
            )

        timing = capture_oom(_time)
        loop = timing.payload
        if timing.status == "ok" and loop is not None:
            peak_max = max(peak_max, peak_mem_mb())

    status = probe.status if probe.status != "ok" else timing.status
    error = probe.error or timing.error

    mean_temp = _temperature(atoms) if status == "ok" else None
    if args.nvt_check:
        print(f"NVT check: mean kinetic temperature = {mean_temp} K "
              f"(target {args.temperature_k} K)")
    rec = BenchRecord(
        engine="fairchem-uma",
        model=args.model,
        task=args.task,
        integrator="langevin",
        num_atoms=num_atoms,
        box_angstrom=box,
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        nodes=1,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        timestep_fs=args.timestep_fs,
        temperature_k=args.temperature_k,
        friction_inv_fs=args.friction_inv_fs,
        status=status,
        input_file=os.path.abspath(args.input),
        tag=args.tag,
        elapsed_s=loop.elapsed_s if loop else None,
        steps_per_sec=loop.steps_per_sec if loop else None,
        ns_per_day=loop.ns_per_day if loop else None,
        peak_mem_mb_per_rank=peaks,
        peak_mem_mb_max=peak_max,
        mean_temperature_k=mean_temp,
        error=error,
    )
    path = write_result(args.output_dir, rec)
    print(f"wrote {path}  status={status}")
    if error:
        print(error)

    # Ray actors otherwise keep the job alive past the Python exit.
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
