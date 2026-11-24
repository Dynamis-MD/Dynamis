"""Tests for the shared element/physics data"""

import pytest
import torch

from dynamis_state import ATOMIC_MASSES, sample_maxwell_boltzmann
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


# ─── Maxwell-Boltzmann sampling ────────────────────────────────────────────

# Boltzmann constant in amu * A^2 / (fs^2 * K), matching physics.hpp
K_B = 8.314462618e-7
T_K = 300.0

# O, H, H, C, H, H, H, H
_SMALL_SYSTEM = torch.tensor([8, 1, 1, 6, 1, 1, 1, 1], dtype=torch.int64)


def test_sampled_velocities_shape_and_dtype():
    v = sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 0)
    assert v.shape == (len(_SMALL_SYSTEM), 3)
    assert v.dtype == torch.float32
    assert torch.all(v.norm(dim=1) > 0)


def test_sampling_is_seeded():
    a = sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 7)
    b = sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 7)
    c = sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sampling_leaves_global_rng_untouched():
    rng_state = torch.get_rng_state()
    sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 0)
    assert torch.equal(rng_state, torch.get_rng_state())


def test_zero_temperature_samples_zero_velocities():
    v = sample_maxwell_boltzmann(_SMALL_SYSTEM, 0.0, 0)
    assert torch.equal(v, torch.zeros(len(_SMALL_SYSTEM), 3))


def test_sampled_velocities_have_zero_total_momentum():
    types = torch.tensor([1, 6, 8] * 300, dtype=torch.int64)
    v = sample_maxwell_boltzmann(types, T_K, 0)
    momentum = (ATOMIC_MASSES[types].unsqueeze(1) * v).sum(dim=0)
    assert torch.allclose(momentum, torch.zeros(3), atol=1e-4)


def test_single_atom_samples_zero_velocity():
    """COM removal leaves nothing for a lone atom."""
    v = sample_maxwell_boltzmann(torch.tensor([6]), T_K, 0)
    assert torch.allclose(v, torch.zeros(1, 3))


def test_empty_system_samples_nothing():
    v = sample_maxwell_boltzmann(torch.zeros(0, dtype=torch.int64), T_K, 0)
    assert v.shape == (0, 3)


def test_sampled_kinetic_energy_matches_temperature():
    """Per species, <1/2 m v^2> = 3/2 k_B T, which also checks the 1/sqrt(m) scaling.
    With ~10k atoms per species the relative std is ~0.8%, so 5% is loose."""
    types = torch.tensor([1, 8] * 10_000, dtype=torch.int64)
    v = sample_maxwell_boltzmann(types, T_K, 0)

    masses = ATOMIC_MASSES[types].double()
    kinetic = 0.5 * masses * (v.double() ** 2).sum(dim=1)
    expected = 1.5 * K_B * T_K
    for z in (1, 8):
        assert kinetic[types == z].mean().item() == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("bad_type", [0, 119, -1])
def test_sampling_unknown_atom_type_raises(bad_type):
    types = torch.tensor([6, bad_type, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match=f"atom 1 has unknown atom type {bad_type}"):
        sample_maxwell_boltzmann(types, T_K, 0)


@pytest.mark.parametrize("temperature", [-1.0, float("nan")])
def test_invalid_temperature_raises(temperature):
    with pytest.raises(ValueError, match="non-negative"):
        sample_maxwell_boltzmann(_SMALL_SYSTEM, temperature, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sampling_matches_across_devices():
    """Sampling happens on the CPU, so a CUDA input gets the same values back on CUDA."""
    cpu = sample_maxwell_boltzmann(_SMALL_SYSTEM, T_K, 3)
    cuda = sample_maxwell_boltzmann(_SMALL_SYSTEM.cuda(), T_K, 3)
    assert cuda.is_cuda
    assert torch.equal(cpu, cuda.cpu())
