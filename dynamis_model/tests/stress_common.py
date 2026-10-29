"""Systems and state construction shared by the per-backend stress tests."""

from __future__ import annotations

import os
import socket

import numpy as np
import torch

from dynamis_state import (
    DynamisState,
    MolecularInfo,
    SFCCurveType,
    SFCPartitioner,
    SimulationConfig,
)
from dynamis_state.masses import TORCH_ATOMIC_MASSES

GHOST_BUFFER = 0.5


def deformed_silicon():
    """64-atom rattled Si with a non-trivial, non-cubic strain so every stress
    component is non-zero."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=7)
    eps = np.array([[0.010, 0.005, 0.000], [0.005, -0.008, 0.003], [0.0, 0.003, 0.004]])
    atoms.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return atoms


def scaled_silicon(scale):
    """64-atom rattled Si whose lattice constant is ``scale`` times 5.43 A, cubic."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43 * scale, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.02, seed=1)
    return atoms


def slab_silicon():
    """Two Si layers, periodic in x/y only, with vacuum along z."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 1))
    atoms.center(vacuum=6.0, axis=2)
    atoms.set_pbc([True, True, False])
    atoms.rattle(0.05, seed=3)
    eps = np.array([[0.010, 0.004, 0.0], [0.004, -0.006, 0.0], [0.0, 0.0, 0.0]])
    atoms.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return atoms


def raw_from(atoms):
    return {
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def _assign_if_present(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def build_state(raw, cutoff, torch_env, dtype=torch.float32, ghost_buffer=GHOST_BUFFER):
    """Partition ``raw`` over ``torch_env`` with an SFC partitioner and return
    the rank's DynamisState."""
    device = torch_env.device
    numbers = torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device)
    positions = torch.tensor(raw["positions"], dtype=dtype, device=device)
    global_mol = MolecularInfo()
    global_mol.num_atoms = positions.shape[0]
    global_mol.atom_positions = positions
    global_mol.atom_types = numbers
    global_mol.atom_velocities = torch.zeros_like(positions)
    _assign_if_present(global_mol, "atom_forces", torch.zeros_like(positions))
    _assign_if_present(
        global_mol, "atom_vertex_numbers", torch.arange(positions.shape[0], device=device)
    )
    _assign_if_present(
        global_mol, "atom_masses", TORCH_ATOMIC_MASSES.to(device)[numbers].float()
    )

    sim = SimulationConfig()
    sim.cutoff = cutoff
    sim.ghost_cutoff_buffer = ghost_buffer
    sim.cell = torch.tensor(raw["cell"], dtype=dtype, device=device)
    sim.pbc = torch.tensor(raw["pbc"], dtype=torch.bool, device=device)

    partitioner = SFCPartitioner(SFCCurveType.Morton)
    info = partitioner.partition(global_mol, torch_env.dist_env)
    partitioner.build_halo_maps(info, global_mol, torch_env.dist_env, sim)
    local_indices = info.partition_indices.long()
    local_mol = MolecularInfo()
    local_mol.num_atoms = local_indices.numel()
    local_mol.atom_positions = positions.index_select(0, local_indices)
    local_mol.atom_types = numbers.index_select(0, local_indices)
    local_mol.atom_velocities = torch.zeros_like(local_mol.atom_positions)
    _assign_if_present(
        local_mol, "atom_forces", torch.zeros_like(local_mol.atom_positions)
    )
    _assign_if_present(local_mol, "atom_vertex_numbers", local_indices)
    _assign_if_present(
        local_mol,
        "atom_masses",
        TORCH_ATOMIC_MASSES.to(device)[local_mol.atom_types].float(),
    )

    state = DynamisState()
    state.dist_env = torch_env.dist_env
    state.sim_config = sim
    state.global_molecular_info = global_mol
    state.local_partition_info = info
    state.local_molecular_info = local_mol
    return state


def in_plane_stress_ok(got, ref):
    """Slab check: in-plane block equals the reference, vacuum axis is exactly 0."""
    block = np.abs(ref[:2, :2]).max()
    assert block > 1e-3, "reference in-plane stress is trivial"
    np.testing.assert_allclose(got[:2, :2], ref[:2, :2], atol=1e-2 * block)
    assert not got[2, :].any() and not got[:, 2].any()


def single_rank_process_group_env():
    """Env vars for a one-rank process group, for backends whose exchangers
    assert a live group even at world size 1 (ORB, UMA). A no-op under torchrun."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
