"""
Runs a world_size=1 UMA simulation as ground truth, then compares it
against a trajectory previously dumped by run_distributed_trajectory.py.

Usage:
    python test_uma_md.py --path <temp dir path>

Exits 0 and prints PASSED on success; exits 1 and prints the failing
assertion on mismatch.
"""
import argparse
import os
import sys

import torch

from distnnp import Simulator
from distnnp_integrator import Verlet
from distnnp_model.nnp import UMAModel

from md_test_common import (
    compare_trajectories,
    ensure_test_structure,
    make_config,
    real_singleton_torch_dist_env,
    run_md_loop,
    teardown_singleton_dist_env,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    structure = f"{args.path}/system.xyz"
    output_dir = f"{args.path}/ref_out"
    dist_dump = f"{args.path}/dist_traj.pt"

    ensure_test_structure(structure)  # no-ops if run_distributed_trajectory.py already wrote it

    torch_dist_env = real_singleton_torch_dist_env()
    dist_env = torch_dist_env.dist_env
    assert dist_env.world_size == 1  # sanity: this run must be the true single-process baseline

    torch.manual_seed(1234)
    model = UMAModel("uma-s-1p2", dataset="omol", device=torch_dist_env.device.type)
    integrator = Verlet(dt=0.5)
    config = make_config(output_dir)

    sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
    sim.setup(structure)

    ref_traj = run_md_loop(sim, config)
    teardown_singleton_dist_env()

    dist_traj = torch.load(dist_dump, weights_only=False)

    report_path = os.path.join(output_dir, "comparison_report.csv")
    try:
        compare_trajectories(ref_traj, dist_traj, report_path=report_path)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print(f"PASSED: {len(ref_traj)}-step trajectory matches within tolerance schedule")
    sys.exit(0)


if __name__ == "__main__":
    main()
