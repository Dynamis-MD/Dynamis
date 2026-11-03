"""Stress from MACEModel(compute_stress=True) against two independent references.

* MACE's own ASE calculator on the same weights (no Dynamis code involved).
* Central finite differences of MACEModel's energy under explicit strains.

The single-process tests run with plain ``pytest``. The two-rank test checks
that the all-reduced stress equals the ASE reference on every rank, which is
the multi-rank claim in ``dynamis_model.strain``; run it with::

    torchrun --nproc-per-node=2 -m pytest dynamis_model/tests/mace/test_mace_stress.py

float32 compute (``COMPUTE_DTYPE``) limits the finite-difference test to ~1e-2
of the stress scale.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist

from stress_common import (
    build_state,
    deformed_silicon,
    in_plane_stress_ok,
    raw_from,
    slab_silicon,
)

from dynamis_model.nnp.mace import COMPUTE_DTYPE, MACEModel, load_raw_mace_model
from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD
from dynamis_state import TorchDistEnv

try:
    import ase  # noqa: F401

    HAS_ASE = True
except ImportError:  # pragma: no cover
    HAS_ASE = False

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
DEVICE = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAS_ASE, reason="ASE not installed"),
]


@pytest.fixture(scope="module")
def raw_model():
    try:
        return load_raw_mace_model("mp", "medium-mpa-0", DEVICE).to(DEVICE).eval()
    except Exception as exc:  # no network / weights
        pytest.skip(f"MACE-MP weights unavailable: {exc}")


def ase_reference_stress(atoms):
    """eV/A^3, tensile positive, from MACE's own autograd path."""
    from mace.calculators import mace_mp

    atoms = atoms.copy()
    atoms.calc = mace_mp(
        model="medium-mpa-0", device=DEVICE, default_dtype="float32"
    )
    return atoms.get_stress(voigt=False)


def dynamis_stress(raw_model, atoms, env, compute_stress=True):
    state = build_state(raw_from(atoms), float(raw_model.r_max), env)
    model = MACEModel(raw_model)
    model.initialize_buffers(
        state.local_partition_info, env.device, env.world_size
    )
    return model(state, compute_stress=compute_stress)


def strained(atoms, delta, h):
    out = atoms.copy()
    out.set_cell(out.cell.array @ (np.eye(3) + h * delta), scale_atoms=True)
    return out


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_stress_matches_ase_calculator(raw_model):
    atoms = deformed_silicon()
    env = TorchDistEnv(is_distributed=False)
    _, _, stress = dynamis_stress(raw_model, atoms, env)
    got = stress.cpu().numpy() / EV_TO_MD
    ref = ase_reference_stress(atoms)
    assert np.abs(ref).max() > 1e-3, "reference stress is trivially zero"
    np.testing.assert_allclose(got, ref, atol=1e-2 * np.abs(ref).max())
    np.testing.assert_allclose(got, got.T, atol=1e-7)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_stress_matches_finite_difference_of_energy(raw_model):
    atoms = deformed_silicon()
    env = TorchDistEnv(is_distributed=False)
    _, _, stress = dynamis_stress(raw_model, atoms, env)
    stress = stress.cpu().numpy() / EV_TO_MD
    volume = abs(np.linalg.det(atoms.cell.array))
    rng = np.random.default_rng(0)
    h = 2e-3
    for _ in range(3):
        delta = rng.standard_normal((3, 3))
        delta = 0.5 * (delta + delta.T)
        e_plus, _ = dynamis_stress(raw_model, strained(atoms, delta, h), env, False)
        e_minus, _ = dynamis_stress(raw_model, strained(atoms, delta, -h), env, False)
        fd = (e_plus.item() - e_minus.item()) / (2 * h) / EV_TO_MD
        analytic = float((stress * delta).sum() * volume)
        assert fd == pytest.approx(analytic, rel=5e-2, abs=5e-2)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_forces_unchanged_by_stress_request(raw_model):
    atoms = deformed_silicon()
    env = TorchDistEnv(is_distributed=False)
    e0, f0 = dynamis_stress(raw_model, atoms, env, False)
    e1, f1, _ = dynamis_stress(raw_model, atoms, env, True)
    assert torch.allclose(e0, e1)
    assert torch.allclose(f0, f1, atol=1e-7)


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs torchrun --nproc-per-node>=2")
def test_two_rank_stress_matches_ase_reference(raw_model):
    atoms = deformed_silicon()
    env = TorchDistEnv(is_distributed=True)
    state = build_state(raw_from(atoms), float(raw_model.r_max), env)
    model = MACEModel(raw_model)
    model.initialize_buffers(state.local_partition_info, env.device, env.world_size)
    _, _, stress = model(state, compute_stress=True)
    got = stress.cpu().numpy() / EV_TO_MD

    # Same reference on every rank; the ASE calculator uses no Dynamis code.
    ref = ase_reference_stress(atoms)
    assert np.abs(ref).max() > 1e-3
    np.testing.assert_allclose(got, ref, atol=1e-2 * np.abs(ref).max())

    # The all-reduce must leave every rank with the same tensor.
    gathered = [torch.zeros_like(stress) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stress.contiguous())
    for other in gathered:
        assert torch.equal(other, gathered[0])


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_slab_in_plane_stress_matches_ase_and_vacuum_axis_is_zero(raw_model):
    atoms = slab_silicon()
    env = TorchDistEnv(is_distributed=False)
    _, _, stress = dynamis_stress(raw_model, atoms, env)
    in_plane_stress_ok(stress.cpu().numpy() / EV_TO_MD, ase_reference_stress(atoms))


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs torchrun --nproc-per-node>=2")
def test_two_rank_slab_stress_matches_ase_reference(raw_model):
    atoms = slab_silicon()
    env = TorchDistEnv(is_distributed=True)
    state = build_state(raw_from(atoms), float(raw_model.r_max), env)
    model = MACEModel(raw_model)
    model.initialize_buffers(state.local_partition_info, env.device, env.world_size)
    _, _, stress = model(state, compute_stress=True)
    in_plane_stress_ok(stress.cpu().numpy() / EV_TO_MD, ase_reference_stress(atoms))
