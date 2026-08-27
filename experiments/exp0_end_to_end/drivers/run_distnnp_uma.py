#!/usr/bin/env python3
"""DistNNP-UMA end-to-end MD benchmark driver.

Launched one rank per GPU via torchrun:

    torchrun --standalone --nproc_per_node=<N> run_distnnp_uma.py \
        --input <xyz> --output-dir <dir> [bench_common flags]

Atoms are spatially partitioned (SFC-Morton) across ranks; only ghost
atoms/features are exchanged per message-passing layer. Timing excludes model
load and the warmup window (see bench_common.TimedLoop).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ase.io
import torch
import torch.distributed as dist

from bench_common import (
    BenchRecord,
    RunOutcome,
    TimedLoop,
    capture_oom,
    deadline,
    kT_native,
    parse_bench_args,
    peak_mem_mb,
    reset_peak_mem,
    write_result,
)


def _global_num_atoms(sim) -> int:
    try:
        gmi = sim.state._state.global_molecular_info
        return int(gmi.atom_positions.shape[0])
    except Exception:
        return 0


def _kinetic_temperature(sim) -> float:
    """Mean instantaneous temperature (K) over all ranks, from local
    velocities + tabulated masses. velocities are Å/fs, masses amu, so
    KE is in amu*Å^2/fs^2 and we divide by k_B in the same native units."""
    from distnnp_integrator.masses import TORCH_MASSES
    from bench_common import BOLTZMANN_EV_PER_K, EV_TO_AMU_A2_PER_FS2

    local = sim.state.local_state
    n_local = int(local.num_atoms)
    vel = local.atom_velocities[:n_local]
    masses = TORCH_MASSES.to(vel.device)[local.atom_types[:n_local].reshape(-1).long()]
    ke_local = 0.5 * torch.sum(masses.unsqueeze(-1) * vel * vel)
    ke = ke_local.detach().clone()
    n = torch.tensor(float(n_local), device=vel.device)
    if dist.is_initialized():
        dist.all_reduce(ke, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
    kb_native = BOLTZMANN_EV_PER_K * EV_TO_AMU_A2_PER_FS2
    dof = 3.0 * float(n.item())
    return float((2.0 * ke.item()) / (dof * kb_native))


def _thermalize(sim, temperature_k: float, seed: int) -> None:
    """Draw fresh Maxwell-Boltzmann velocities on this rank's local atoms.

    The synthetic CH4/O2 .xyz carry arbitrary `momenta` (generator sets
    MOMENTA_STD=0.5 and documents them as physically meaningless, to be
    re-thermalized by whatever engine runs them). The FairChem driver already
    calls `MaxwellBoltzmannDistribution`; do the equivalent here so both
    engines start from the same ~T ensemble instead of a ~1e6 K soup.
    """
    from distnnp_integrator.masses import TORCH_MASSES
    from bench_common import BOLTZMANN_EV_PER_K, EV_TO_AMU_A2_PER_FS2

    local = sim.state.local_state
    n_local = int(local.num_atoms)
    dev = local.atom_velocities.device
    masses = TORCH_MASSES.to(dev)[local.atom_types[:n_local].reshape(-1).long()].unsqueeze(-1)
    kb_native = BOLTZMANN_EV_PER_K * EV_TO_AMU_A2_PER_FS2
    sigma = torch.sqrt(torch.as_tensor(kb_native * temperature_k, device=dev) / masses)  # Å/fs
    g = torch.Generator(device=dev).manual_seed(seed)
    local.atom_velocities[:n_local] = torch.randn((n_local, 3), generator=g, device=dev) * sigma


def main() -> None:
    args = parse_bench_args()

    from distnnp import Simulator, SimulatorConfig
    from distnnp_integrator import Langevin
    from distnnp_model.nnp import UMAModel
    from distnnp_state import TorchDistEnv

    torch_dist_env = TorchDistEnv(is_distributed=True)
    dist_env = torch_dist_env.dist_env
    device = torch_dist_env.device
    rank = torch_dist_env.global_rank
    world_size = torch_dist_env.world_size

    # Every rank makes the output dir — TrajectoryWriter mkdir's it and racing
    # ranks otherwise trip over each other.
    os.makedirs(args.output_dir, exist_ok=True)

    # DistNNP's Simulator ignores the xyz `Lattice=` header and builds its cell
    # from side_length_angstrom, so read the box explicitly.
    _atoms0 = ase.io.read(args.input)
    box = float(_atoms0.cell.lengths()[0])
    file_num_atoms = len(_atoms0)

    torch.manual_seed(1234)
    model = UMAModel(args.model, dataset=args.task, device=device.type)
    integ = Langevin(
        dt=args.timestep_fs,
        gamma=args.friction_inv_fs,
        kT=kT_native(args.temperature_k),
    )
    cfg = SimulatorConfig(
        timestep_fs=args.timestep_fs,
        total_steps=args.warmup_steps + args.timed_steps,
        side_length_angstrom=box,
        cutoff_angstrom=args.cutoff_angstrom,
        ghost_buffer_angstrom=args.ghost_buffer_angstrom,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=args.rebalance_interval,
        traj_interval=0,
        log_interval=0,
        output_dir=args.output_dir,
        backend="nccl",
    )

    sim_holder: dict = {}
    step_ctr = {"i": 0}

    # --- phase 1: feasibility. Build + partition + ONE step. An OOM here shows
    # up as an NCCL/collective hang, so the whole block is watchdog-bounded.
    def _probe():
        with deadline(args.build_timeout_s, "distnnp build + setup + feasibility step"):
            sim = Simulator(model, integ, dist_env, cfg, device)
            sim.setup(args.input)
            sim_holder["sim"] = sim
            if sim.should_rebalance(0):
                sim.sync_and_repartition()
            # Reset the synthetic file's meaningless momenta to a real MB ensemble
            # (post-repartition, so it lands on the final local atom set).
            _thermalize(sim, args.temperature_k, seed=1234 + rank)
            sim.step(cfg.timestep_fs)
            step_ctr["i"] = 1

    probe: RunOutcome = capture_oom(_probe)
    my_peak = peak_mem_mb()  # feasibility-phase peak

    # --- phase 2: timing (only if the partition provably fits on this GPU count)
    loop = None
    timing = probe
    if probe.status == "ok":
        def _time():
            sim = sim_holder["sim"]

            def one_step(_i: int) -> None:
                s = step_ctr["i"]
                step_ctr["i"] += 1
                if sim.should_rebalance(s):
                    sim.sync_and_repartition()
                sim.step(cfg.timestep_fs)

            reset_peak_mem()
            return TimedLoop(
                one_step,
                max(0, args.warmup_steps - 1),  # 1 step already spent in the probe
                args.timed_steps,
                args.timestep_fs,
                step_timeout_s=args.step_timeout_s,
            )

        timing = capture_oom(_time)
        loop = timing.payload
        if timing.status == "ok":
            my_peak = max(my_peak, peak_mem_mb())

    status = probe.status if probe.status != "ok" else timing.status
    error = probe.error or timing.error

    # ---- gather peak memory from every rank (only safe when all ranks agree
    # they finished cleanly; on failure the ranks may be desynced, so rank 0
    # just reports its own) ----
    peaks: list = [my_peak]
    if status == "ok" and dist.is_initialized():
        peaks = [None] * world_size
        with deadline(60, "peak all_gather"):
            dist.all_gather_object(peaks, my_peak)
    peaks = [float(x) for x in peaks if x is not None] or [my_peak]

    mean_temp = None
    if status == "ok" and "sim" in sim_holder:
        try:
            mean_temp = _kinetic_temperature(sim_holder["sim"])
        except Exception as e:  # noqa: BLE001
            print(f"[rank {rank}] temperature probe failed: {e!r}")

    if args.nvt_check and rank == 0:
        print(f"NVT check: mean kinetic temperature = {mean_temp} K "
              f"(target {args.temperature_k} K)")

    if rank == 0:
        rec = BenchRecord(
            engine="distnnp-uma",
            model=args.model,
            task=args.task,
            integrator="langevin",
            num_atoms=(_global_num_atoms(sim_holder["sim"]) if "sim" in sim_holder else 0) or file_num_atoms,
            box_angstrom=box,
            world_size=world_size,
            gpus_per_node=torch_dist_env.gpus_per_node,
            nodes=torch_dist_env.num_nodes,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            timestep_fs=args.timestep_fs,
            temperature_k=args.temperature_k,
            friction_inv_fs=args.friction_inv_fs,
            status=status,
            input_file=os.path.abspath(args.input),
            tag=args.tag,
            elapsed_s=loop.elapsed_s if loop else None,
            steps_per_sec=loop.steps_per_sec if loop else None,
            ns_per_day=loop.ns_per_day if loop else None,
            peak_mem_mb_per_rank=peaks,
            peak_mem_mb_max=max(peaks) if peaks else None,
            mean_temperature_k=mean_temp,
            error=error,
        )
        path = write_result(args.output_dir, rec)
        print(f"[rank 0] wrote {path}  status={status}")
        if error:
            print(error)

    if dist.is_initialized():
        try:
            if status == "ok":
                with deadline(60, "final barrier"):
                    dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
