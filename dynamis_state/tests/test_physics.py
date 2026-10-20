"""Tests for the shared element/physics data"""

import torch

from dynamis_state import ATOMIC_MASSES
from dynamis_state.dynamis_state_impl import mass_table


def test_mass_table_indexed_by_atomic_number():
    assert ATOMIC_MASSES.shape == (119,)
    assert ATOMIC_MASSES.dtype == torch.float32
    assert ATOMIC_MASSES[0] == 0.0
    # every element up to Oganesson has a mass
    assert torch.all(ATOMIC_MASSES[1:] > 0)
    assert ATOMIC_MASSES[1].item() == torch.tensor(1.008).item()
    assert ATOMIC_MASSES[6].item() == torch.tensor(12.011).item()
    assert ATOMIC_MASSES[18].item() == torch.tensor(39.948).item()


def test_mass_table_returns_a_fresh_copy():
    """Mutating one result must not leak into the integrators' masses."""
    table = mass_table()
    table[1] = -1.0
    assert mass_table()[1] > 0
