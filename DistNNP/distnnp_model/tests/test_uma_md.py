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
import pytest

import torch

from ase import Atoms

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False

from distnnp import Simulator
from distnnp_integrator import Verlet
from distnnp_model.nnp import UMAModel
from distnnp_model.potential import DistNNPModel

from md_test_common import (
    compare_trajectories,
    ensure_test_structure,
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
        return torch.tensor(self.calc.get_potential_energy(atoms), device=self.device), torch.tensor(self.calc.get_forces(atoms), device=self.device)

    def initialize_buffers(self, partition_info, device):
        pass


@pytest.mark.skipif(not HAS_FAIRCHEM, reason="fairchem-core is not installed")
def test_uma_md(md_test_path):
    structure = f"{md_test_path}/system.xyz"
    output_dir = f"{md_test_path}/ref_out"
    dist_dump = f"{md_test_path}/dist_traj.pt"

    ensure_test_structure(structure)  # no-ops if run_distributed_trajectory.py already wrote it

    torch_dist_env = real_singleton_torch_dist_env()
    dist_env = torch_dist_env.dist_env
    assert dist_env.world_size == 1  # sanity: this run must be the true single-process baseline

    torch.manual_seed(1234)
    # model = UMAModel("uma-s-1p2", dataset="omol", device=torch_dist_env.device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    test_uma_md(args.path)
