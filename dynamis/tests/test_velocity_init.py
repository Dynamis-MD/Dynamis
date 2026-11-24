"""
test_velocity_init.py — which velocities Simulator.setup() starts from.

Pure CPU, so it runs without torchrun:
    pytest dynamis/tests/test_velocity_init.py -v
"""

import warnings

import pytest
import torch

from dynamis.simulator import _initialize_velocities
from dynamis_state import MolecularInfo, sample_maxwell_boltzmann

T_K = 300.0
_TYPES = torch.tensor([8, 1, 1, 6, 1, 1, 1, 1], dtype=torch.int64)


@pytest.fixture(scope="session", autouse=True)
def dist_init():
    """Overrides the conftest fixture: nothing here is distributed."""
    yield


def _molecular_info(velocities: torch.Tensor) -> MolecularInfo:
    info = MolecularInfo()
    info.num_atoms = len(_TYPES)
    info.atom_types = _TYPES.clone()
    info.atom_velocities = velocities
    return info


def test_zero_velocities_are_sampled_without_warning():
    info = _molecular_info(torch.zeros(len(_TYPES), 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _initialize_velocities(info, T_K, seed=5)
    assert torch.equal(
        info.atom_velocities, sample_maxwell_boltzmann(_TYPES, T_K, 5)
    )


def test_nonzero_velocities_are_kept_without_warning():
    velocities = torch.full((len(_TYPES), 3), 0.25)
    info = _molecular_info(velocities.clone())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _initialize_velocities(info, T_K, seed=0)
    assert torch.equal(info.atom_velocities, velocities)


def test_zero_temperature_keeps_a_zero_start():
    info = _molecular_info(torch.zeros(len(_TYPES), 3))
    _initialize_velocities(info, 0.0, seed=0)
    assert torch.equal(info.atom_velocities, torch.zeros(len(_TYPES), 3))
