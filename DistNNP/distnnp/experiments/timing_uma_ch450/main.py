"""
Timing sanity-check: UMA model on the CH450 system, single run at a fixed
world size.

Purpose: confirm the distributed pipeline runs end-to-end under UMA at 1/2/4
GPUs and produce per-rank timing_rank{N}.jsonl files (via timing_level="detail",
so halo_exchange_wait is recorded separately from halo_exchange). This is the
first step of testing_plan.md — once this is working, a larger system
(experiments/exp_datasets/) replaces ch450.xyz for the real scaling study.

Launch with torchrun (see run_timing_sweep.sh for the 1/2/4 GPU sweep):
    torchrun --standalone --nproc_per_node=<N> main.py --output-dir results/timing/<N>gpu
"""
import argparse
import os

from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_state import TorchDistEnv
from distnnp_integrator import Verlet
from distnnp_model.nnp import UMAModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_STRUCTURE = os.path.normpath(os.path.join(_HERE, "../../../../baseline/ch450.xyz"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="./results",
                   help="Where timing_rank{N}.jsonl and trajectory files are written.")
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--timing-level", default="detail", choices=["off", "basic", "detail"])
    p.add_argument("--model-name", default="uma-s-1p2")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch_dist_env = TorchDistEnv()
    device = torch_dist_env.device

    sim_config = SimulatorConfig(
        timestep_fs=0.5,
        total_steps=args.total_steps,
        cutoff_angstrom=6.0,
        side_length_angstrom=20.0,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        # Fixed partition for the whole run: rebalancing mid-run would inject
        # partition/build_halo_maps cost into an arbitrary subset of steps and
        # skew the per-step timing comparison across GPU counts.
        rebalance_interval=0,
        output_dir=args.output_dir,
        traj_interval=0,  # trajectory writer not under test here
        log_interval=1,
        timing_level=args.timing_level,
        backend="nccl",
    )

    model = UMAModel(args.model_name, torch_dist_env.global_rank, device=device)
    integrator = Verlet(sim_config.timestep_fs)
    simulator = Simulator(
        model=model,
        integrator=integrator,
        dist_environment=torch_dist_env.dist_env,
        config=sim_config,
        device=device,
    )
    simulator.setup(_STRUCTURE)
    simulator.run()


if __name__ == "__main__":
    main()
