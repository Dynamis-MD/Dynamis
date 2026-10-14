"""
Run the distributed MACEModel simulation and write a merged trajectory.

Usage:
    torchrun --nproc_per_node=2 dynamis_model/tests/mace/run_distributed_trajectory.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dynamis import Simulator
from dynamis_integrator import Verlet
from dynamis_state import TorchDistEnv

if __package__:
    from .md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        build_mace_model,
        ensure_test_structure,
        install_trajectory_simulator_import_compatibility,
        make_config,
        run_md_loop,
        structure_sha256,
    )
else:
    from md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        build_mace_model,
        ensure_test_structure,
        install_trajectory_simulator_import_compatibility,
        make_config,
        run_md_loop,
        structure_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--head-name", default="")
    parser.add_argument(
        "--model-family", default="mp", choices=["mp", "off", "omol", "polar"]
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Convert the distributed raw MACE model to cuEquivariance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    install_trajectory_simulator_import_compatibility()
    structure = Path(args.path) / "silicon_supercell.xyz"
    output_dir = Path(args.path) / "dist_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    try:
        torch_dist_env = TorchDistEnv(is_distributed=True)
        dist_env = torch_dist_env.dist_env

        if dist_env.global_rank == 0:
            ensure_test_structure(structure)
        dist.barrier()

        torch.manual_seed(SEED)
        model = build_mace_model(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system="silicon",
            enable_cueq=args.enable_cueq,
        )
        integrator = Verlet(dt=0.5)
        config = make_config(
            str(output_dir),
            model.cutoff,
            rebalance_interval=REBALANCE_INTERVAL,
        )

        sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
        sim.setup(str(structure))

        local_frames = run_md_loop(sim, config)
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

        if dist_env.global_rank == 0:
            merged = []
            for step in range(len(local_payload)):
                rank_local_energies = [
                    float(gathered[rank][step][0])
                    for rank in range(dist_env.world_size)
                ]
                vids = np.concatenate(
                    [gathered[rank][step][1] for rank in range(dist_env.world_size)]
                )
                pos = np.concatenate(
                    [gathered[rank][step][2] for rank in range(dist_env.world_size)]
                )
                vel = np.concatenate(
                    [gathered[rank][step][3] for rank in range(dist_env.world_size)]
                )
                order = np.argsort(vids)
                frame = {
                    "energy": float(sum(rank_local_energies)),
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
                merged.append(frame)
            artifact = {
                "metadata": {
                    "artifact_version": 2,
                    "model": args.model,
                    "model_family": args.model_family,
                    "head_name": args.head_name,
                    "enable_cueq": args.enable_cueq,
                    "structure_sha256": structure_sha256(structure),
                    "timestep_fs": config.timestep_fs,
                    "step_count": N_STEPS,
                    "rebalance_interval": REBALANCE_INTERVAL,
                    "world_size": dist_env.world_size,
                    "adapter_identity": "MACEModel",
                    "force_unit": "amu*angstrom/fs^2",
                },
                "frames": merged,
            }
            torch.save(artifact, dist_dump)
            print(
                f"Wrote merged {len(merged)}-step distributed trajectory to {dist_dump}"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
