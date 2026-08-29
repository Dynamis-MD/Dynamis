"""
Run the world-size=1 thin-adapter MACE simulation as ground truth and compare
it against a trajectory written by run_distributed_trajectory.py.

Usage:
    python test_md.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from distnnp import Simulator
from distnnp_integrator import Verlet

if __package__:
    from .md_test_common import (
        SEED,
        build_thin_mace_model,
        compare_trajectories,
        ensure_test_structure,
        make_config,
        patch_simulator_trajectory_writer_compat,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
    )
else:
    from md_test_common import (
        SEED,
        build_thin_mace_model,
        compare_trajectories,
        ensure_test_structure,
        make_config,
        patch_simulator_trajectory_writer_compat,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
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
    output_dir = Path(args.path) / "ref_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    ensure_test_structure(structure)

    try:
        torch_dist_env = real_singleton_torch_dist_env()
        dist_env = torch_dist_env.dist_env
        assert dist_env.world_size == 1

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

        ref_traj = run_md_loop(sim, config)
        assert dist_dump.exists(), f"Distributed trajectory artifact not found: {dist_dump}"
        dist_traj = torch.load(dist_dump, weights_only=False)

        report_path = os.path.join(output_dir, "comparison_report.csv")
        compare_trajectories(ref_traj, dist_traj, report_path=report_path)
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    finally:
        teardown_singleton_dist_env()

    print(f"PASSED: {len(ref_traj)}-step trajectory matches within tolerance schedule")
    sys.exit(0)


if __name__ == "__main__":
    main()
