"""
test_simulation_distributed.py — integration tests for the Simulation class.

These tests verify that Simulation can be constructed and stepped without error
in a multi-GPU distributed setting.  Run via:

    torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v
"""

import torch
import pytest


def test_simulation_constructs(simulation_factory):
    """Simulation(state, model, integrator) completes without error on 2+ ranks."""
    simulation_factory()


def test_simulation_step_runs(simulation_factory):
    """simulation.step() completes without error."""
    simulation = simulation_factory()
    simulation.step()


def test_energy_is_scalar(simulation_factory):
    """Energy returned from step() is a scalar (rank-0 or single-element tensor)."""
    simulation = simulation_factory()
    energy, _ = simulation.step()
    assert energy.numel() == 1


def test_forces_shape(simulation_factory):
    """Forces have shape (n_local_atoms, 3) on each rank."""
    simulation = simulation_factory()
    _, forces = simulation.step()
    n_local_atoms = simulation.state.pos.shape[0]
    assert forces.shape[0] == n_local_atoms
    assert forces.shape[1] == 3


def test_energy_forces_finite(simulation_factory):
    """No NaN or Inf values in energy or forces after a step."""
    simulation = simulation_factory()
    energy, forces = simulation.step()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()

def test_multi_step_no_crash(simulation_factory):
    """10 consecutive steps complete without error or state corruption."""
    simulation = simulation_factory()
    for _ in range(10):
        simulation.step()
