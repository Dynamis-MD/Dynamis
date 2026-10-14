"""
Run a world-size=1 MACE reference simulation and compare it against a MACEModel
trajectory written by run_distributed_trajectory.py. The reference can use
either MACEModel or MACE's independent public ASE calculator.

Usage:
    python dynamis_model/tests/mace/test_md.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dynamis import Simulator
from dynamis_integrator import Verlet

if __package__:
    from .md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        build_monolithic_mace_model,
        build_mace_model,
        compare_trajectories,
        ensure_test_structure,
        install_trajectory_simulator_import_compatibility,
        make_config,
        real_singleton_torch_dist_env,
        run_md_loop,
        structure_sha256,
        teardown_singleton_dist_env,
    )
else:
    from md_test_common import (
        SEED,
        N_STEPS,
        REBALANCE_INTERVAL,
        build_monolithic_mace_model,
        build_mace_model,
        compare_trajectories,
        ensure_test_structure,
        install_trajectory_simulator_import_compatibility,
        make_config,
        real_singleton_torch_dist_env,
        run_md_loop,
        structure_sha256,
        teardown_singleton_dist_env,
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
        help="Use cuEquivariance in both the singleton reference and distributed artifact.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["mace", "monolithic"],
        default="monolithic",
        help=(
            "Singleton reference implementation. 'monolithic' is the acceptance oracle; "
            "'mace' is retained only for explicit developer comparisons."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    install_trajectory_simulator_import_compatibility()
    structure = Path(args.path) / "silicon_supercell.xyz"
    output_dir = Path(args.path) / "ref_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    ensure_test_structure(structure)

    try:
        torch_dist_env = real_singleton_torch_dist_env()
        dist_env = torch_dist_env.dist_env
        assert dist_env.world_size == 1

        torch.manual_seed(SEED)
        model_factory = (
            build_monolithic_mace_model
            if args.reference_mode == "monolithic"
            else build_mace_model
        )
        model = model_factory(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system="silicon",
            enable_cueq=args.enable_cueq,
        )
        print(
            f"Running {args.reference_mode} singleton MACE reference trajectory for silicon."
        )
        integrator = Verlet(dt=0.5)
        config = make_config(
            str(output_dir),
            model.cutoff,
            rebalance_interval=REBALANCE_INTERVAL,
        )

        sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
        sim.setup(str(structure))

        ref_traj = run_md_loop(sim, config)
        assert (
            dist_dump.exists()
        ), f"Distributed trajectory artifact not found: {dist_dump}"
        artifact = torch.load(dist_dump, weights_only=False)
        if not isinstance(artifact, dict) or set(artifact) != {"metadata", "frames"}:
            raise ValueError(
                "Distributed trajectory artifact must contain exactly 'metadata' and 'frames'."
            )
        expected_metadata = {
            "artifact_version": 2,
            "model": args.model,
            "model_family": args.model_family,
            "head_name": args.head_name,
            "enable_cueq": args.enable_cueq,
            "structure_sha256": structure_sha256(structure),
            "timestep_fs": config.timestep_fs,
            "step_count": N_STEPS,
            "rebalance_interval": REBALANCE_INTERVAL,
            "world_size": 2,
            "adapter_identity": "MACEModel",
            "force_unit": "amu*angstrom/fs^2",
        }
        if artifact["metadata"] != expected_metadata:
            raise ValueError(
                "Distributed trajectory metadata does not match the singleton reference: "
                f"expected {expected_metadata}, got {artifact['metadata']}"
            )
        compare_trajectories(
            ref_traj,
            artifact["frames"],
            report_path=str(output_dir / "comparison_report.csv"),
        )
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    finally:
        teardown_singleton_dist_env()

    print(f"PASSED: {len(ref_traj)}-step trajectory matches within tolerance schedule")
    sys.exit(0)


if __name__ == "__main__":
    main()
