"""Tests for io integration"""

import torch
import pytest

from distnnp_state import (MolecularInfo, load_state)

def test_xyz_extended_io():
    """Extended xyz test"""
    env = load_state("tests/extended_xyz.xyz")
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape(6, 3)
    velocities = torch.concat((values, values), dim=0)
    velocities = velocities.reshape(6, 3)

    assert env.num_atoms == 6
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(velocities, env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1), env.atom_indices)

def test_xyz_io():
    """Regular xyz test"""
    env = load_state("tests/xyz.xyz")
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape([6, 3])
    assert env.num_atoms == 6
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(torch.zeros([6, 3]), env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1), env.atom_indices)
