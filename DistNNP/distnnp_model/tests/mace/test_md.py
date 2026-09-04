"""
Run a world-size=1 MACE reference simulation and compare it against a thin-
adapter trajectory written by run_distributed_trajectory.py. The reference can
use either the thin adapter or MACE's independent public ASE calculator.

Usage:
    python distnnp_model/tests/mace/test_md.py --path <temp dir path>
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
        N_STEPS,
        REBALANCE_INTERVAL,
        TEST_SYSTEMS,
        build_comparison_report,
        build_monolithic_mace_model,
        build_thin_mace_model,
        compare_trajectories,
        ensure_test_structure,
        get_test_system_spec,
        make_config,
        patch_simulator_trajectory_writer_compat,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
        write_comparison_report_csv,
        write_force_diagnostic_report,
    )
else:
    from md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        TEST_SYSTEMS,
        build_comparison_report,
        build_monolithic_mace_model,
        build_thin_mace_model,
        compare_trajectories,
        ensure_test_structure,
        get_test_system_spec,
        make_config,
        patch_simulator_trajectory_writer_compat,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
        write_comparison_report_csv,
        write_force_diagnostic_report,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--head-name", default="")
    parser.add_argument("--model-family", default="mp", choices=["mp", "off", "omol", "polar"])
    parser.add_argument("--system", default="silicon", choices=TEST_SYSTEMS)
    parser.add_argument(
        "--reference-mode",
        choices=["thin", "monolithic"],
        default="thin",
        help=(
            "Singleton reference implementation. 'thin' preserves the existing diagnostic "
            "path; 'monolithic' uses MACE's public ASE calculator and native graph builder."
        ),
    )
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--disable-rebalance-after-setup", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0 or args.timestep_fs <= 0:
        raise ValueError("--steps and --timestep-fs must be positive.")
    if args.steps != N_STEPS and not args.diagnostic_only:
        raise ValueError("Short trajectory runs require --diagnostic-only so acceptance criteria remain unchanged.")
    if args.reference_mode == "monolithic" and args.diagnostics:
        raise ValueError(
            "--diagnostics currently records thin-adapter internals; use "
            "--reference-mode thin for that existing diagnostic path."
        )
    patch_simulator_trajectory_writer_compat()

    system_spec = get_test_system_spec(args.system)
    structure = Path(args.path) / system_spec["filename"]
    output_dir = Path(args.path) / "ref_out"
    dist_dump = Path(args.path) / "dist_traj.pt"
    distributed_diagnostic_dump = Path(args.path) / "distributed_force_diagnostics.pt"

    ensure_test_structure(structure, system=args.system)

    try:
        torch_dist_env = real_singleton_torch_dist_env()
        dist_env = torch_dist_env.dist_env
        assert dist_env.world_size == 1

        torch.manual_seed(SEED)
        model_factory = (
            build_monolithic_mace_model
            if args.reference_mode == "monolithic"
            else build_thin_mace_model
        )
        model = model_factory(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system=args.system,
        )
        print(
            f"Running {args.reference_mode} singleton MACE reference trajectory "
            f"for {args.system}."
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
            # Match the distributed producer's preflight partition exactly.
            if not args.disable_rebalance_after_setup:
                sim.sync_and_repartition()
                skip_initial_rebalance = True
            model.enable_force_diagnostics()
            model(sim.state._state)

        ref_traj = run_md_loop(
            sim,
            config,
            steps=args.steps,
            skip_initial_rebalance=skip_initial_rebalance,
        )
        assert dist_dump.exists(), f"Distributed trajectory artifact not found: {dist_dump}"
        dist_traj = torch.load(dist_dump, weights_only=False)

        report_path = os.path.join(output_dir, "comparison_report.csv")
        if args.diagnostics:
            assert distributed_diagnostic_dump.exists(), (
                f"Distributed force diagnostics not found: {distributed_diagnostic_dump}"
            )
            force_report_path = output_dir / "force_diagnostic_report.csv"
            write_force_diagnostic_report(
                model.force_diagnostics,
                torch.load(distributed_diagnostic_dump, weights_only=False),
                force_report_path,
            )

        if args.diagnostic_only:
            rows = build_comparison_report(ref_traj, dist_traj)
            write_comparison_report_csv(rows, report_path)
            print("DIAGNOSTIC ONLY: skipped trajectory acceptance assertions.")
        else:
            compare_trajectories(
                ref_traj,
                dist_traj,
                report_path=report_path,
                side_length_angstrom=system_spec["side_length_angstrom"],
            )
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    finally:
        teardown_singleton_dist_env()

    if args.diagnostic_only:
        print(f"COMPLETED: {len(ref_traj)}-step diagnostic trajectory")
    else:
        print(f"PASSED: {len(ref_traj)}-step trajectory matches within tolerance schedule")
    sys.exit(0)


if __name__ == "__main__":
    main()
