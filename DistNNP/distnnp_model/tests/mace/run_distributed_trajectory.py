"""
Run the distributed thin-adapter MACE simulation and write a merged trajectory.

Usage:
    torchrun --nproc_per_node=2 distnnp_model/tests/mace/run_distributed_trajectory.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from distnnp import Simulator
from distnnp_integrator import Verlet
from distnnp_state import TorchDistEnv

if __package__:
    from .md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        TEST_SYSTEMS,
        build_thin_mace_model,
        ensure_test_structure,
        get_test_system_spec,
        make_config,
        merge_force_diagnostics,
        patch_simulator_trajectory_writer_compat,
        run_md_loop,
    )
else:
    from md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        TEST_SYSTEMS,
        build_thin_mace_model,
        ensure_test_structure,
        get_test_system_spec,
        make_config,
        merge_force_diagnostics,
        patch_simulator_trajectory_writer_compat,
        run_md_loop,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--head-name", default="")
    parser.add_argument("--model-family", default="mp", choices=["mp", "off", "omol", "polar"])
    parser.add_argument("--system", default="silicon", choices=TEST_SYSTEMS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--disable-rebalance-after-setup", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0 or args.timestep_fs <= 0:
        raise ValueError("--steps and --timestep-fs must be positive.")
    patch_simulator_trajectory_writer_compat()

    system_spec = get_test_system_spec(args.system)
    structure = Path(args.path) / system_spec["filename"]
    output_dir = Path(args.path) / "dist_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    try:
        torch_dist_env = TorchDistEnv(is_distributed=True)
        dist_env = torch_dist_env.dist_env

        if dist_env.global_rank == 0:
            ensure_test_structure(structure, system=args.system)
        dist.barrier()

        torch.manual_seed(SEED)
        model = build_thin_mace_model(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system=args.system,
        )
        integrator = Verlet(dt=args.timestep_fs)
        config = make_config(
            str(output_dir),
            model.cutoff,
            timestep_fs=args.timestep_fs,
            rebalance_interval=0 if args.disable_rebalance_after_setup else REBALANCE_INTERVAL,
            side_length_angstrom=system_spec["side_length_angstrom"],
        )

        sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
        sim.setup(str(structure))

        skip_initial_rebalance = False
        if args.diagnostics:
            # Make this force evaluation observe the exact partition used by MD step 0.
            if not args.disable_rebalance_after_setup:
                sim.sync_and_repartition()
                skip_initial_rebalance = True
            model.enable_force_diagnostics()
            model(sim.state._state)

        local_frames = run_md_loop(
            sim,
            config,
            steps=args.steps,
            skip_initial_rebalance=skip_initial_rebalance,
        )
        local_payload = [
            (
                frame["energy"],
                frame["vid"],
                frame["pos"],
                frame["vel"],
            )
            for frame in local_frames
        ]

        dist.barrier()
        gathered = [None] * dist_env.world_size
        dist.all_gather_object(gathered, local_payload)

        gathered_diagnostics = None
        if args.diagnostics:
            gathered_diagnostics = [None] * dist_env.world_size
            dist.all_gather_object(gathered_diagnostics, model.force_diagnostics)

        if dist_env.global_rank == 0:
            merged = []
            for step in range(len(local_payload)):
                rank_local_energies = [float(gathered[rank][step][0]) for rank in range(dist_env.world_size)]
                vids = np.concatenate([gathered[rank][step][1] for rank in range(dist_env.world_size)])
                pos = np.concatenate([gathered[rank][step][2] for rank in range(dist_env.world_size)])
                vel = np.concatenate([gathered[rank][step][3] for rank in range(dist_env.world_size)])
                order = np.argsort(vids)
                frame = {
                    "energy": float(sum(rank_local_energies)),
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
                merged.append(frame)
            torch.save(merged, dist_dump)
            print(f"Wrote merged {len(merged)}-step distributed trajectory to {dist_dump}")
            if gathered_diagnostics is not None:
                diagnostic_dump = Path(args.path) / "distributed_force_diagnostics.pt"
                torch.save(merge_force_diagnostics(gathered_diagnostics), diagnostic_dump)
                print(f"Wrote distributed force diagnostics to {diagnostic_dump}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
