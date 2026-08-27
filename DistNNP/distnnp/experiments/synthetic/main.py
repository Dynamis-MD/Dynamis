import argparse
import json
import os

from setup_system import generate_synthetic_dataset, Timer
from distnnp_model.empirical import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator import Verlet
import torch
from distnnp_state import TorchDistEnv
import gc


def reset_gpu():
    """Reset GPU memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic LJ baseline experiment")
    parser.add_argument("--num-atoms", type=int, default=1000)
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--rebalance-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--traj-interval",
        type=int,
        default=0,
        help="Write a trajectory snapshot every N steps (0 = disabled, recommended for sweeps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    return parser.parse_args()


def create_and_run_experiment(args):
    torch_distributed_env = TorchDistEnv()

    local_rank = torch_distributed_env.local_rank
    global_rank = torch_distributed_env.global_rank
    world_size = torch_distributed_env.world_size
    device = torch_distributed_env.device

    for i in range(world_size):
        if i == global_rank:
            print(
                f"Hello from global rank {global_rank}, local rank {local_rank}, world size {world_size}, device {device}!"
            )
        torch_distributed_env.barrier()

    def local_print(*args, **kwargs):
        if global_rank == 0:
            print(*args, **kwargs)

    sim_config = SimulatorConfig(
        timestep_fs=args.timestep_fs,
        total_steps=args.steps,
        cutoff_angstrom=6.0,
        side_length_angstrom=args.box_size,
        ghost_buffer_angstrom=0.5,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=args.rebalance_interval,
        output_dir=args.output_dir,
        traj_file="trajectory.traj",
        traj_interval=args.traj_interval,
        log_interval=args.log_interval,
        writer_buffer_size=10,
        backend="nccl",
    )

    global_molecular_system = generate_synthetic_dataset(
        num_atoms=args.num_atoms, box_size=args.box_size, device=device
    )

    # Pre-create output_dir on every rank before Simulator/TrajectoryWriter
    # construction: the C++ TrajectoryWriter does stat()-then-mkdir() with no
    # EEXIST handling, so two ranks racing to create the same directory can
    # hard-exit(1) the loser. Creating it up front (idempotent) avoids the race.
    os.makedirs(args.output_dir, exist_ok=True)

    epsilons = torch.rand(100, requires_grad=True).to(device=device)
    sigmas = torch.rand(100, requires_grad=True).to(device=device)
    potential = LennardJonesPotentialModel(epsilon=epsilons, sigma=sigmas)
    integrator = Verlet(sim_config.timestep_fs)
    simulator = Simulator(
        model=potential,
        integrator=integrator,
        dist_environment=torch_distributed_env.dist_env,
        config=sim_config,
        device=device,
    )
    simulator.initialize_simulator_state(global_molecular_system, device=device)
    torch_distributed_env.barrier()

    reset_gpu()
    timer = Timer(
        "synthetic_baseline",
        metadata={
            "world_size": world_size,
            "num_atoms": args.num_atoms,
            "box_size": args.box_size,
            "steps": args.steps,
        },
    )
    with timer:
        simulator.run()
    torch_distributed_env.barrier()

    elapsed_s = timer.local_total_time / 1000.0
    steps_per_sec = args.steps / elapsed_s
    peak_mb = get_peak_mb() if device.type == "cuda" else None

    local_print(
        f"Completed {args.steps} steps on {world_size} rank(s), {args.num_atoms} atoms: "
        f"{elapsed_s:.3f}s ({steps_per_sec:.2f} steps/s), peak mem {peak_mb:.1f} MB"
        if peak_mb is not None
        else f"Completed {args.steps} steps on {world_size} rank(s), {args.num_atoms} atoms: "
        f"{elapsed_s:.3f}s ({steps_per_sec:.2f} steps/s)"
    )

    Timer.add_meta_data(
        {
            "world_size": world_size,
            "num_atoms": args.num_atoms,
            "box_size": args.box_size,
            "steps": args.steps,
            "elapsed_s": elapsed_s,
            "steps_per_sec": steps_per_sec,
            "peak_memory_mb": peak_mb,
        }
    )

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(
            args.output_dir,
            f"synthetic_baseline_{world_size}gpu_{args.num_atoms}atoms.json",
        )
        with open(result_path, "w") as f:
            f.write(Timer.json())
        print(f"Wrote results to {result_path}")


def main():
    args = parse_args()
    create_and_run_experiment(args)


if __name__ == "__main__":
    main()
