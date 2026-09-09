"""
Runs a world_size=1 UMA simulation as ground truth, then compares it
against a trajectory previously dumped by run_distributed_trajectory.py.

Usage:
    python test_uma_md.py --path <temp dir path>

Exits 0 and prints PASSED on success; exits 1 and prints the failing
assertion on mismatch.
"""
import argparse
import numpy as np
import os
import pytest
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ase import Atoms

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False

from distnnp import Simulator
from distnnp_integrator import Verlet
from distnnp_model.nnp import UMAModel
from distnnp_model.nnp.uma import _EV_PER_A_TO_AMU_A_PER_FS2
from distnnp_model.potential import DistNNPModel
from distnnp_state import TorchDistEnv

from md_test_common import (
    compare_trajectories,
    ensure_test_structure,
    free_port,
    make_config,
    real_singleton_torch_dist_env,
    run_md_loop,
    teardown_singleton_dist_env,
)


class LocalUMAModel(DistNNPModel):
    def __init__(self, model_name, dataset, device):
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device.type)
        self.calc = FAIRChemCalculator(predictor, task_name=dataset)
        self.device = device

    def __call__(self, distributed_state):
        local_mol_info = distributed_state.local_molecular_info
        sim_config = distributed_state.sim_config
        atom_types = local_mol_info.atom_types.detach().clone().cpu().numpy()
        atom_positions = local_mol_info.atom_positions.detach().clone().cpu().numpy()
        cell = sim_config.cell.detach().clone().cpu().numpy()
        pbc = sim_config.pbc.detach().clone().cpu().numpy()
        atoms = Atoms(atom_types, positions=atom_positions, cell=cell, pbc=pbc, info={"charge": 0, "spin": 1})
        return torch.tensor(self.calc.get_potential_energy(atoms) * _EV_PER_A_TO_AMU_A_PER_FS2, device=self.device), torch.tensor(self.calc.get_forces(atoms) * _EV_PER_A_TO_AMU_A_PER_FS2, device=self.device)

    def initialize_buffers(self, partition_info, device):
        pass


def _distributed_worker(rank: int, world_size: int, path: str, master_port: int, log: bool = False) -> None:
    if log:
        log_path = f"/tmp/rank_{rank}.log"
     
        def log(msg):
            with open(log_path, "a") as f:
                f.write(f"{msg}\n")
     
        log(f"starting: rank={rank} world_size={world_size}")
    else:
        def log(msg):
            pass
 
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
 
    try:
        torch_dist_env = TorchDistEnv(is_distributed=True)
        dist_env = torch_dist_env.dist_env
        log(f"joined process group: global_rank={dist_env.global_rank}")
 
        structure = f"{path}/system.xyz"
        output_dir = f"{path}/dist_out"
        dist_dump = f"{path}/dist_traj.pt"
 
        if dist_env.global_rank == 0:
            ensure_test_structure(structure)
        dist.barrier()
 
        torch.manual_seed(1234)
        model = UMAModel("uma-s-1p2", dataset="omol", device=torch_dist_env.device.type)
        integrator = Verlet(dt=0.5)
        config = make_config(output_dir)
 
        sim = Simulator(model, integrator, dist_env, config, torch_dist_env.device)
        sim.setup(structure)
        log("setup complete, starting MD loop")
 
        local_frames = run_md_loop(sim, config)
        log(f"MD loop complete: {len(local_frames)} steps")
        local_payload = [(f["energy"], f["vid"], f["pos"], f["vel"]) for f in local_frames]
 
        dist.barrier()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_payload)
        log("gather complete")
 
        if dist_env.global_rank == 0:
            merged = []
            for step in range(len(local_payload)):
                total_energy = sum(gathered[r][step][0] for r in range(world_size))
                vids = np.concatenate([gathered[r][step][1] for r in range(world_size)])
                pos = np.concatenate([gathered[r][step][2] for r in range(world_size)])
                vel = np.concatenate([gathered[r][step][3] for r in range(world_size)])
                order = np.argsort(vids)
                merged.append({
                    "energy": total_energy,
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                })
            torch.save(merged, dist_dump)
            log(f"wrote {len(merged)}-step trajectory to {dist_dump}")
 
        dist.barrier()
        log("done")
    except Exception as e:
        log(f"EXCEPTION: {type(e).__name__}: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_comparison(md_test_path):
    structure = f"{md_test_path}/system.xyz"
    output_dir = f"{md_test_path}/ref_out"
    dist_dump = f"{md_test_path}/dist_traj.pt"

    ensure_test_structure(structure)  # no-ops if run_distributed_trajectory.py already wrote it

    torch_dist_env = real_singleton_torch_dist_env()
    dist_env = torch_dist_env.dist_env
    assert dist_env.world_size == 1  # sanity: this run must be the true single-process baseline

    torch.manual_seed(1234)
    model = LocalUMAModel("uma-s-1p2", dataset="omol", device=torch_dist_env.device)
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
 
 
@pytest.mark.skipif(not HAS_FAIRCHEM, reason="fairchem-core is not installed")
def test_uma_md(md_test_path):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        pytest.skip("requires >=2 GPUs")
 
    port = free_port()
    mp.spawn(_distributed_worker, args=(world_size, md_test_path, port), nprocs=world_size, join=True)
 
    _run_comparison(md_test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    test_uma_md(args.path)
