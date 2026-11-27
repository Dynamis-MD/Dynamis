#!/usr/bin/env python3
"""Dynamis + ORB distributed MD scaling benchmark driver.

Emits per-run results directly in the canonical Dynamis-experiments-log layout:
    <output-dir>/<study>/<run_id>/{result.json, timings/timing_rank*.jsonl.gz}

Single-node (1/2/4 GPUs inside a 4-GPU allocation):
    torchrun --standalone --nproc_per_node=<W> run_orb_scaling.py \
        --input <xyz> --model orb_v3_direct_20_omat \
        --exchanger gpudirect --study strong --tag strong1m --output-dir <dir>

Multi-node (8/16/32 GPUs via srun + torchrun):
    srun --ntasks=<N> --ntasks-per-node=1 --cpu-bind=none \
      torchrun --nnodes=<N> --nproc-per-node=4 --rdzv-backend=c10d ... \
        run_orb_scaling.py --input <xyz> --model orb_v3_direct_20_omat \
        --exchanger gpudirect --study weak --tag weak32k --output-dir <dir>
"""
from __future__ import annotations

import inspect
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist

from bench_common import (
    BenchRecord,
    RunOutcome,
    TimedLoop,
    build_base_parser,
    capture_oom,
    compress_timings,
    deadline,
    detect_use_fabric,
    env_versions,
    gather_rank_metrics,
    inject_cell,
    kT_native,
    kinetic_temperature,
    load_system,
    make_run_id,
    peak_mem_mb,
    reset_peak_mem,
    resolve_exchanger_cls,
    run_directory,
    wrap_scalable_halo,
    write_result,
)

ORB_MODELS = [
    "orb_v3_direct_inf_omat",
    "orb_v3_direct_inf_mpa",
    "orb_v3_direct_20_omat",
    "orb_v3_direct_20_mpa",
    "orb_v3_conservative_inf_omat",
    "orb_v3_conservative_inf_mpa",
    "orb_v3_conservative_20_omat",
    "orb_v3_conservative_20_mpa",
]


def parse_args(argv: list[str] | None = None):
    p = build_base_parser(
        description="Dynamis + ORB strong/weak/hero scaling benchmark cell"
    )
    p.add_argument("--model", default="orb_v3_direct_inf_omat", choices=ORB_MODELS)
    p.add_argument(
        "--orb-precision",
        default="float32-high",
        choices=["float32-high", "float32-highest", "float64"],
    )
    p.add_argument(
        "--edge-chunk-size",
        type=int,
        default=0,
        help="Edge chunk size for Orb-Direct (0 disables chunking; requires PR #156)",
    )
    p.add_argument(
        "--mixed-precision",
        default="none",
        choices=["none", "bf16"],
        help="Optional mixed precision for Orb-Direct ('bf16' or 'none'; requires PR #156)",
    )
    p.add_argument(
        "--use-fused-kernels",
        default="false",
        choices=["auto", "true", "false"],
        help="Blackwell sm_100a CuTeDSL+Triton fused kernels for Orb-Direct (requires PR #158)",
    )
    return p.parse_args(argv)


def _build_orb_model(args, device, exchanger_cls):
    from dynamis_model.nnp.orb import ORBModel

    edge_chunk_size = (
        args.edge_chunk_size if args.edge_chunk_size and args.edge_chunk_size > 0 else None
    )
    mixed_precision = None if args.mixed_precision == "none" else args.mixed_precision
    use_fused_kernels = {"auto": "auto", "true": True, "false": False}[
        args.use_fused_kernels
    ]

    sig_params = inspect.signature(ORBModel.__init__).parameters
    kwargs: dict = {
        "model_name": args.model,
        "precision": args.orb_precision,
        "device": device.type if isinstance(device, torch.device) else device,
        "exchanger": exchanger_cls,
    }

    if "edge_chunk_size" in sig_params:
        kwargs["edge_chunk_size"] = edge_chunk_size
    elif edge_chunk_size is not None:
        raise RuntimeError(
            f"--edge-chunk-size={edge_chunk_size} requested, but installed ORBModel "
            "does not support edge_chunk_size (requires PR #156)."
        )

    if "mixed_precision" in sig_params:
        kwargs["mixed_precision"] = mixed_precision
    elif mixed_precision is not None:
        raise RuntimeError(
            f"--mixed-precision={mixed_precision} requested, but installed ORBModel "
            "does not support mixed_precision (requires PR #156)."
        )

    if "use_fused_kernels" in sig_params:
        kwargs["use_fused_kernels"] = use_fused_kernels
    elif use_fused_kernels is True:
        raise RuntimeError(
            "--use-fused-kernels=true requested, but installed ORBModel "
            "does not support use_fused_kernels (requires PR #158)."
        )

    return ORBModel(**kwargs), edge_chunk_size, mixed_precision, use_fused_kernels


def main() -> None:
    args = parse_args()

    from dynamis import Simulator, SimulatorConfig
    from dynamis_integrator import Langevin
    from dynamis_state import Level, Timer, TimerClass, TimingRegistry, TorchDistEnv, Where

    exchanger_cls = resolve_exchanger_cls(args.exchanger)

    torch_dist_env = TorchDistEnv(is_distributed=True)
    dist_env = torch_dist_env.dist_env
    device = torch_dist_env.device
    rank = torch_dist_env.global_rank
    world_size = torch_dist_env.world_size

    spec = load_system(
        args.input,
        pbc_mode=args.pbc,
        cutoff_angstrom=args.cutoff_angstrom,
        ghost_buffer_angstrom=args.ghost_buffer_angstrom,
        nonperiodic_pad=args.nonperiodic_pad,
        resample_velocities=args.resample_velocities,
    )
    box = spec.box_angstrom
    num_atoms = spec.num_atoms

    eff_tag = args.tag or args.study
    run_id = make_run_id(
        args.model, args.exchanger, args.study, num_atoms, world_size, args.tag
    )
    rdir = run_directory(args.output_dir, args.study, run_id)
    timings_path = os.path.join(rdir, "timings")
    os.makedirs(timings_path, exist_ok=True)

    torch.manual_seed(1234)
    model, edge_chunk_size, mixed_precision, use_fused_kernels = _build_orb_model(
        args, device, exchanger_cls
    )
    max_neighbors = int(model.model_max_num_neighbors)

    if rank == 0:
        print(
            f"[rank 0] run_id={run_id} model={args.model} precision={args.orb_precision} "
            f"mixed_precision={mixed_precision} edge_chunk_size={edge_chunk_size} "
            f"use_fused_kernels={use_fused_kernels} "
            f"exchanger={args.exchanger} W={world_size} nodes={torch_dist_env.num_nodes} "
            f"atoms={num_atoms} ({num_atoms / world_size:.0f}/GPU) "
            f"scalable_halo={args.scalable_halo}"
        )
        print(f"[rank 0] cell={spec.cell} pbc={spec.pbc}")

    integ = Langevin(
        dt=args.timestep_fs,
        gamma=args.friction_inv_fs,
        kT=kT_native(args.temperature_k),
    )
    cfg = SimulatorConfig(
        timestep_fs=args.timestep_fs,
        total_steps=args.warmup_steps + args.timed_steps,
        initial_temperature_K=args.temperature_k,
        velocity_seed=args.velocity_seed,
        side_length_angstrom=box,
        cutoff_angstrom=args.cutoff_angstrom,
        ghost_buffer_angstrom=args.ghost_buffer_angstrom,
        partition_algorithm=args.partition_algorithm,
        edge_weighted_partition=True,
        rebalance_interval=args.rebalance_interval,
        enable_drift_watchdog=args.drift_watchdog,
        traj_interval=0,
        log_interval=0,
        output_dir=timings_path,
        backend="nccl",
    )

    sim_holder: dict = {}
    step_ctr = {"i": 0}
    in_timed = {"flag": False}

    def _probe():
        with deadline(args.build_timeout_s, "ORB build + setup + feasibility step"):
            sim = Simulator(model, integ, dist_env, cfg, device)
            inject_cell(sim, spec.cell, spec.pbc)
            if args.scalable_halo:
                wrap_scalable_halo(sim, rank, world_size)
            sim.setup(spec.info)
            sim_holder["sim"] = sim
            # Keep TimingRegistry off during warmup so timing_rank*.jsonl reflects timed window
            Timer.set_level(Level.Off)
            sim.step(cfg.timestep_fs)
            step_ctr["i"] = 1

    probe: RunOutcome = capture_oom(_probe)
    my_peak = peak_mem_mb()

    use_fabric = (
        detect_use_fabric(model, args.exchanger) if probe.status == "ok" else None
    )

    loop = None
    timing = probe
    if probe.status == "ok":

        def _time():
            sim = sim_holder["sim"]
            timer = sim.timer

            def _enable_timing():
                in_timed["flag"] = True
                Timer.set_level(Level.Basic)

            def one_step(_i: int) -> None:
                s = step_ctr["i"]
                step_ctr["i"] += 1
                if in_timed["flag"]:
                    Timer.set_step(s)
                    timer.start("iteration", TimerClass.Total, Where.Both)
                if sim.should_rebalance(s):
                    sim.sync_and_repartition()
                sim.step(cfg.timestep_fs)
                if in_timed["flag"]:
                    timer.stop()

            reset_peak_mem()
            return TimedLoop(
                one_step,
                max(0, args.warmup_steps - 1),
                args.timed_steps,
                args.timestep_fs,
                step_timeout_s=args.step_timeout_s,
                on_timed_start=_enable_timing,
            )

        timing = capture_oom(_time)
        loop = timing.payload
        if timing.status == "ok":
            my_peak = max(my_peak, peak_mem_mb())

    try:
        TimingRegistry.instance().close()
    except Exception:
        pass

    status = probe.status if probe.status != "ok" else timing.status
    error = probe.error or timing.error

    peaks: list[float] = [my_peak]
    local_counts: list[int] | None = None
    ghost_counts: list[int] | None = None
    hostnames: list[str] = [socket.gethostname()]

    if status == "ok" and "sim" in sim_holder:
        peaks, local_counts, ghost_counts, hostnames = gather_rank_metrics(
            sim_holder["sim"], my_peak, world_size
        )

    mean_temp = None
    if status == "ok" and "sim" in sim_holder:
        try:
            mean_temp = kinetic_temperature(sim_holder["sim"])
        except Exception as e:
            print(f"[rank {rank}] temperature probe failed: {e!r}")

    # Synchronize before rank 0 gzips all ranks' timing_rank*.jsonl traces
    if dist.is_initialized() and status == "ok":
        try:
            with deadline(60, "pre-compress barrier"):
                dist.barrier()
        except Exception:
            pass

    fused_kernels_active = bool(getattr(model, "_fused_state", None) is not None)

    if rank == 0:
        has_timings = status == "ok" and compress_timings(timings_path)
        run_metrics: dict = {
            "fused_kernels_active": fused_kernels_active,
            "use_fused_kernels": args.use_fused_kernels,
        }
        if edge_chunk_size is not None:
            run_metrics["edge_chunk_size"] = edge_chunk_size
        if mixed_precision is not None:
            run_metrics["mixed_precision"] = mixed_precision

        rec = BenchRecord(
            run_id=run_id,
            study=args.study,
            tag=eff_tag,
            model=args.model,
            status=status,
            num_atoms=num_atoms,
            world_size=world_size,
            precision=args.orb_precision,
            max_num_neighbors=max_neighbors,
            exchanger=args.exchanger,
            use_fabric=use_fabric,
            partition_algorithm=args.partition_algorithm,
            scalable_halo=bool(args.scalable_halo),
            integrator="langevin",
            atoms_per_gpu=num_atoms / float(world_size),
            box_angstrom=box,
            gpus_per_node=torch_dist_env.gpus_per_node,
            nodes=torch_dist_env.num_nodes,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            timestep_fs=args.timestep_fs,
            temperature_k=args.temperature_k,
            friction_inv_fs=args.friction_inv_fs,
            dataset_file=os.path.basename(args.input),
            timings_dir="timings" if has_timings else None,
            elapsed_s=loop.elapsed_s if loop else None,
            steps_per_sec=loop.steps_per_sec if loop else None,
            ns_per_day=loop.ns_per_day if loop else None,
            peak_mem_mb_per_rank=peaks,
            peak_mem_mb_max=max(peaks) if peaks else None,
            mean_temperature_k=mean_temp,
            local_atoms_per_rank=local_counts,
            ghost_atoms_per_rank=ghost_counts,
            metrics=run_metrics,
            hostnames=hostnames,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            error=error,
            env=env_versions(["orb_models"]),
        )
        out_path = write_result(args.output_dir, rec)
        print(
            f"[rank 0] wrote {out_path}  status={status}  "
            f"use_fabric={use_fabric}  fused={fused_kernels_active}  "
            f"sps={rec.steps_per_sec}  peakMB={rec.peak_mem_mb_max}  "
            f"T={mean_temp}"
        )
        if error:
            print(error)

    if dist.is_initialized():
        try:
            if status == "ok":
                with deadline(60, "final barrier"):
                    dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
