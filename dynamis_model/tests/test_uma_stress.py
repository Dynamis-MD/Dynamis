"""Stress from UMAModel(compute_stress=True) against fairchem's own UMA stress.

The reference is ``FAIRChemCalculator`` (the same weights and task, no Dynamis
code). The single-process tests run with plain ``pytest``; the two-rank tests
need::

    torchrun --nproc-per-node=2 -m pytest dynamis_model/tests/test_uma_stress.py
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
    single_rank_process_group_env,
    slab_silicon,
)

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate import FAIRChemCalculator

    from dynamis_model.nnp.uma import UMAModel

    HAS_FAIRCHEM = True
except ImportError:  # pragma: no cover
    HAS_FAIRCHEM = False

from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD
from dynamis_state import TorchDistEnv

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
MODEL_NAME = "uma-s-1p2"
DATASET = "omat"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAS_FAIRCHEM, reason="fairchem-core not installed"),
]


@pytest.fixture(scope="module")
def env():
    if WORLD_SIZE == 1:
        single_rank_process_group_env()
    return TorchDistEnv(is_distributed=True)  # also selects this rank's GPU


@pytest.fixture(scope="module")
def model(env):
    try:
        return UMAModel(MODEL_NAME, dataset=DATASET, device="cuda")
    except Exception as exc:  # no network / weights
        pytest.skip(f"UMA weights unavailable: {exc}")


@pytest.fixture(scope="module")
def reference(env):
    calculator = FAIRChemCalculator(
        pretrained_mlip.get_predict_unit(MODEL_NAME, device="cuda"), task_name=DATASET
    )

    def stress(atoms):
        # The slab's vacuum gap (12 A) exceeds the 6 A cutoff, so treating it as
        # fully periodic changes no interaction; only the in-plane block is compared.
        atoms = atoms.copy()
        atoms.pbc = True
        atoms.calc = calculator
        return atoms.get_stress(voigt=False)

    return stress


def dynamis_stress(model, env, atoms, compute_stress=True):
    cutoff = float(model.main_model.backbone.cutoff)
    state = build_state(raw_from(atoms), cutoff, env)
    model.initialize_buffers(state.local_partition_info, "cuda")
    # The backbone caches its MOLE mixing coefficients on the first call and reuses
    # them for the rest of an MD run (composition is static there). A test that
    # changes system must clear the cache or it gets the first system's coefficients.
    model.main_model.backbone._cached_mole_coefficients = None
    # charge/spin left at UMAModel's defaults, which follow fairchem's per-task ones
    return model(state, compute_stress=compute_stress)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_stress_matches_fairchem_calculator(model, env, reference):
    atoms = deformed_silicon()
    _, _, stress = dynamis_stress(model, env, atoms)
    got = stress.cpu().numpy() / EV_TO_MD
    ref = reference(atoms)
    assert np.abs(ref).max() > 1e-3, "reference stress is trivially zero"
    np.testing.assert_allclose(got, ref, atol=1e-2 * np.abs(ref).max())
    np.testing.assert_allclose(got, got.T, atol=1e-7)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_forces_unchanged_by_stress_request(model, env):
    atoms = deformed_silicon()
    e0, f0 = dynamis_stress(model, env, atoms, False)
    e1, f1, _ = dynamis_stress(model, env, atoms, True)
    assert torch.allclose(e0, e1)
    assert torch.allclose(f0, f1, atol=1e-6)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_slab_in_plane_stress_matches_fairchem_and_vacuum_axis_is_zero(
    model, env, reference
):
    atoms = slab_silicon()
    _, _, stress = dynamis_stress(model, env, atoms)
    in_plane_stress_ok(stress.cpu().numpy() / EV_TO_MD, reference(atoms))


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs torchrun --nproc-per-node>=2")
@pytest.mark.parametrize("system", [deformed_silicon, slab_silicon])
def test_two_rank_stress_matches_fairchem_calculator(model, env, reference, system):
    atoms = system()
    _, _, stress = dynamis_stress(model, env, atoms)
    got = stress.cpu().numpy() / EV_TO_MD
    ref = reference(atoms)
    if atoms.pbc.all():
        assert np.abs(ref).max() > 1e-3
        np.testing.assert_allclose(got, ref, atol=1e-2 * np.abs(ref).max())
    else:
        in_plane_stress_ok(got, ref)

    # The all-reduce must leave every rank with the same tensor.
    gathered = [torch.zeros_like(stress) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stress.contiguous())
    for other in gathered:
        assert torch.equal(other, gathered[0])
