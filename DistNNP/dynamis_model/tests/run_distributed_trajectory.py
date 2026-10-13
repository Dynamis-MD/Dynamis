"""
Runs the (multi-rank) distributed UMA simulation and dumps the merged,
global-vertex-ID-ordered trajectory to disk. Run this first, then run
test_uma_md.py pointing --path at the same path.

Usage:
    torchrun --nproc_per_node=2 run_distributed_trajectory.py \\
        --path <temp dir path>
"""
import argparse

import torch
import torch.distributed as dist

from dynamis import Simulator
from dynamis_state import TorchDistEnv
from dynamis_integrator import Verlet
from dynamis_model.nnp import UMAModel

from md_test_common import (
    ensure_test_structure,
    make_config,
    run_md_loop,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    structure = f"{args.path}/system.xyz"
    output_dir = f"{args.path}/dist_out"
    dist_dump = f"{args.path}/dist_traj.pt"

    torch_dist_env = TorchDistEnv(is_distributed=True)
    dist_env = torch_dist_env.dist_env

    if dist_env.global_rank == 0:
        ensure_test_structure(structure)
    dist.barrier()  # other ranks must not read the file until rank 0 has written it

    torch.manual_seed(1234)
    model = UMAModel("uma-s-1p2", dataset="omol", device=torch_dist_env.device)
    integrator = Verlet(dt=0.5)
    config = make_config(output_dir)

    sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
    sim.setup(structure)

    local_frames = run_md_loop(sim, config)
    # pack into plain tensors/lists so all_gather_object doesn't need to pickle
    # anything GPU-resident
    local_payload = [
        (f["energy"], f["vid"], f["pos"], f["vel"]) for f in local_frames
    ]

    dist.barrier()
    gathered = [None] * dist_env.world_size
    dist.all_gather_object(gathered, local_payload)

    if dist_env.global_rank == 0:
        merged = []
        for step in range(len(local_payload)):
            total_energy = sum(gathered[r][step][0] for r in range(dist_env.world_size))
            import numpy as np

            vids = np.concatenate([gathered[r][step][1] for r in range(dist_env.world_size)])
            pos = np.concatenate([gathered[r][step][2] for r in range(dist_env.world_size)])
            vel = np.concatenate([gathered[r][step][3] for r in range(dist_env.world_size)])
            order = np.argsort(vids)
            merged.append(
                {
                    "energy": total_energy,
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
            )
        torch.save(merged, dist_dump)
        print(f"Wrote merged {len(merged)}-step distributed trajectory to {dist_dump}")

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
