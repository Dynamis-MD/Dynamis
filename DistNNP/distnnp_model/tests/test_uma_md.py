"""
Trajectory-equivalence test: global (world_size=1) UMA vs. distributed UMA.

Strategy
--------
Velocity Verlet is used instead of Langevin so the reference run is fully
deterministic (no RNG stream to keep in sync across ranks). Both runs start
from the identical structure file and the identical model state_dict, so any
divergence traces back to the distributed code path (partitioning, halo
exchange, ghost-gradient accumulation) rather than to initialization.

Comparison is done in *global vertex-ID order*, not rank order, since
partitioning may reassign atoms to different ranks (and does, once
`rebalance_interval` triggers a repartition mid-run).

Because MD is chaotic, float32 all_reduce order-of-operations differences
between the single-process and multi-rank halo-sum paths can compound. We
therefore assert tight agreement on the early steps (where forces/positions
should match near machine precision) and a looser, monotonically-relaxed
bound on later steps, while also checking that energy stays consistent
between the two runs at every logged step (a cheap invariant that catches
gross bugs like a mis-wired ghost-scatter-reduce even if positions have
drifted).
"""
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist

from distnnp import Simulator, SimulatorConfig
from distnnp_state import TorchDistEnv
from distnnp_integrator import Verlet
from distnnp_model.nnp import UMAModel  # test-config UMA wrapper, e.g. a small checkpoint

N_STEPS = 25
REBALANCE_INTERVAL = 7  # ensures at least 2 repartitions occur within N_STEPS
SEED = 1234
N_ATOMS = 64
CELL_LENGTH = 20.0

# tolerance schedule: (step_index_inclusive_upper_bound, atol)
POS_TOL_SCHEDULE = [(4, 1e-5), (12, 1e-3), (N_STEPS, 5e-2)]
VEL_TOL_SCHEDULE = [(4, 1e-5), (12, 1e-3), (N_STEPS, 5e-2)]
ENERGY_RTOL = 1e-3

def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    nnodes = int(os.environ.get("SLURM_NNODES", 1))
 
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
 
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
 
    return rank, local_rank, world_size, nnodes


def _tol_for_step(step: int, schedule):
    for upper, tol in schedule:
        if step <= upper:
            return tol
    return schedule[-1][1]


def _write_test_structure(path: str, n_atoms: int = N_ATOMS, seed: int = SEED):
    from ase import Atoms
 
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, CELL_LENGTH, size=(n_atoms, 3))
    atoms = Atoms("C" * n_atoms, positions=positions, cell=[CELL_LENGTH] * 3, pbc=True)
    atoms.write(path)


def _make_config(tmpdir: str) -> SimulatorConfig:
    return SimulatorConfig(
        timestep_fs=0.5,
        total_steps=N_STEPS,
        cutoff_angstrom=6.0,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=REBALANCE_INTERVAL,
        traj_interval=0, # trajectory writer not under test here
        log_interval=1,
        output_dir=tmpdir,
        backend="nccl",
    )


def _extract_frame(sim, energy) -> dict:
    partition_info = sim.state._state.local_partition_info
    mol_info = sim.state._state.local_molecular_info
    vid = partition_info.partition_indices[: mol_info.num_atoms].cpu().numpy()
    order = np.argsort(vid)
    return {
        "energy": float(energy),
        "pos": mol_info.atom_positions.detach().cpu().numpy()[order],
        "vel": mol_info.atom_velocities.detach().cpu().numpy()[order],
        "vid": vid[order],
    }


def _run_local_reference(structure_path: str, dist_env: TorchDistEnv, output_dir: str, local_rank: int):
    """world_size == 1 run; acts as ground truth."""
    saved_env = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "1"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        torch.manual_seed(SEED)
 
        model = UMAModel("uma-s-1p2", 0, device=device)
        integrator = Verlet(dt=0.5)
        sim = Simulator(model=model, integrator=integrator, dist_environment=dist_env, config=_make_config(output_dir), device=device)
        sim.setup(structure_path)
 
        assert not dist.is_initialized(), (
            "Simulator.setup() initialized a process group even with WORLD_SIZE=1 — "
            "the 'reference' run is no longer isolated from the real distributed group"
        )
 
        traj = []
        for _ in range(N_STEPS):
            energy = sim.step()
            traj.append(_extract_frame(sim, energy))
        return traj
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
 
 
def _run_distributed_and_gather(structure_path: str, dist_env: TorchDistEnv, output_dir: str):
    rank, local_rank, world_size, _ = setup_distributed()
    device = "cuda"
 
    torch.manual_seed(SEED)
    model = UMAModel("uma-s-1p2", rank, device=device)
    integrator = Verlet(dt=0.5)
    sim = Simulator(model=model, integrator=integrator, dist_environment=dist_env, config=_make_config(output_dir), device=device)
    sim.setup(structure_path)
 
    local_frames = []
    for step in range(N_STEPS):
        if sim.should_rebalance():
            sim._repartition()
        local_energy = sim.step()
        frame = _extract_frame(sim, local_energy)
        local_frames.append((step, frame["energy"], frame["vid"], frame["pos"], frame["vel"]))
 
    dist.barrier()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_frames)
 
    merged = None
    if rank == 0:
        merged = []
        for step in range(N_STEPS):
            total_energy = sum(gathered[r][step][1] for r in range(world_size))
            vids = np.concatenate([gathered[r][step][2] for r in range(world_size)])
            pos = np.concatenate([gathered[r][step][3] for r in range(world_size)])
            vel = np.concatenate([gathered[r][step][4] for r in range(world_size)])
            order = np.argsort(vids)
            merged.append(
                {
                    "energy": total_energy,
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
            )
 
    dist.barrier()
    return rank, merged
 
 
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU(s)")
def test_distributed_matches_global_trajectory(tmp_path):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 2:
        pytest.skip("run via torchrun --nproc_per_node>=2 to exercise this test")
 
    true_rank = int(os.environ.get("RANK", 0))
    true_local_rank = int(os.environ.get("LOCAL_RANK", 0))
 
    structure_path = str(tmp_path / "system.xyz")
    _write_test_structure(structure_path)
 
    ref_traj = None
    if true_rank == 0:
        ref_traj = _run_local_reference(
            structure_path, TorchDistEnv(is_distributed=False).dist_env, str(tmp_path / "ref_out"), true_local_rank
        )
 
    rank, dist_traj = _run_distributed_and_gather(structure_path, TorchDistEnv().dist_env, str(tmp_path / "dist_out"))
 
    if rank != 0:
        return  # participated in the collectives above; assertions run on rank 0 only
 
    assert len(dist_traj) == len(ref_traj) == N_STEPS
 
    for step, (ref_frame, dist_frame) in enumerate(zip(ref_traj, dist_traj)):
        assert np.array_equal(ref_frame["vid"], dist_frame["vid"]), (
            f"step {step}: vertex ID sets diverged between global and distributed runs "
            "(indicates a bug in redistribute_atoms / vertex ID tracking, not numerics)"
        )
 
        pos_tol = _tol_for_step(step, POS_TOL_SCHEDULE)
        vel_tol = _tol_for_step(step, VEL_TOL_SCHEDULE)
 
        np.testing.assert_allclose(
            dist_frame["pos"], ref_frame["pos"], atol=pos_tol, rtol=0,
            err_msg=f"positions diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["vel"], ref_frame["vel"], atol=vel_tol, rtol=0,
            err_msg=f"velocities diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["energy"], ref_frame["energy"], rtol=ENERGY_RTOL,
            err_msg=f"total energy diverged beyond {ENERGY_RTOL} relative at step {step}",
        )

