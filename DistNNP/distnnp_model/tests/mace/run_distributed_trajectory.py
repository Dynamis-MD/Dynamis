"""
Run the distributed thin-adapter MACE simulation and write a merged trajectory.

Usage:
    torchrun --nproc_per_node=2 run_distributed_trajectory.py --path <temp dir path>
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
        RANK_ENERGY_ATOL,
        RANK_ENERGY_RTOL,
        SEED,
        build_thin_mace_model,
        ensure_test_structure,
        make_config,
        patch_simulator_trajectory_writer_compat,
        run_md_loop,
    )
else:
    from md_test_common import (
        RANK_ENERGY_ATOL,
        RANK_ENERGY_RTOL,
        SEED,
        build_thin_mace_model,
        ensure_test_structure,
        make_config,
        patch_simulator_trajectory_writer_compat,
        run_md_loop,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--head-name", default="")
    parser.add_argument("--model-family", default="mp", choices=["mp", "off", "omol", "polar"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patch_simulator_trajectory_writer_compat()

    structure = Path(args.path) / "system.xyz"
    output_dir = Path(args.path) / "dist_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    try:
        torch_dist_env = TorchDistEnv(is_distributed=True)
        dist_env = torch_dist_env.dist_env

        if dist_env.global_rank == 0:
            ensure_test_structure(structure)
        dist.barrier()

        torch.manual_seed(SEED)
        model = build_thin_mace_model(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
        )
        integrator = Verlet(dt=0.5)
        config = make_config(str(output_dir), model.cutoff)

        sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
        sim.setup(str(structure))

        local_frames = run_md_loop(sim, config)
        local_payload = [
            (
                frame["energy"],
                frame.get("owned_local_energy"),
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
                rank_global_energies = [float(gathered[rank][step][0]) for rank in range(dist_env.world_size)]
                representative_energy = rank_global_energies[0]
                if not all(
                    np.isclose(energy, representative_energy, rtol=RANK_ENERGY_RTOL, atol=RANK_ENERGY_ATOL)
                    for energy in rank_global_energies[1:]
                ):
                    raise AssertionError(
                        f"rank-returned global energies disagree at step {step}: {rank_global_energies}"
                    )

                owned_energy_values = [
                    float(gathered[rank][step][1])
                    for rank in range(dist_env.world_size)
                    if gathered[rank][step][1] is not None
                ]
                vids = np.concatenate([gathered[rank][step][2] for rank in range(dist_env.world_size)])
                pos = np.concatenate([gathered[rank][step][3] for rank in range(dist_env.world_size)])
                vel = np.concatenate([gathered[rank][step][4] for rank in range(dist_env.world_size)])
                order = np.argsort(vids)
                frame = {
                    "energy": representative_energy,
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
                if owned_energy_values:
                    frame["owned_local_energy_sum"] = float(sum(owned_energy_values))
                merged.append(frame)
            torch.save(merged, dist_dump)
            print(f"Wrote merged {len(merged)}-step distributed trajectory to {dist_dump}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
