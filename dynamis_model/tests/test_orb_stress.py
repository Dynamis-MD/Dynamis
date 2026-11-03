"""Stress from ORBModel(compute_stress=True) against ORB's own conservative stress.

The reference is ``orbff.predict`` on the same weights (no Dynamis code). The
single-process tests run with plain ``pytest``; the two-rank tests need::

    torchrun --nproc-per-node=2 -m pytest dynamis_model/tests/test_orb_stress.py
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
    from orb_models.forcefield import pretrained

    from dynamis_model.nnp.orb import ORBModel

    HAS_ORB = True
except ImportError:  # pragma: no cover
    HAS_ORB = False

from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD
from dynamis_state import TorchDistEnv

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
MODEL_NAME = "orb_v3_conservative_inf_omat"
PRECISION = "float32-highest"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAS_ORB, reason="orb-models not installed"),
]

# ORB reports Voigt stress as (xx, yy, zz, yz, xz, xy)
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_full(voigt):
    full = np.zeros((3, 3))
    for value, (i, j) in zip(np.asarray(voigt, dtype=float).reshape(6), VOIGT):
        full[i, j] = full[j, i] = value
    return full


@pytest.fixture(scope="module")
def env():
    if WORLD_SIZE == 1:
        single_rank_process_group_env()
    return TorchDistEnv(is_distributed=True)  # also selects this rank's GPU


@pytest.fixture(scope="module")
def model(env):
    try:
        return ORBModel(model_name=MODEL_NAME, precision=PRECISION, device="cuda")
    except Exception as exc:  # no network / weights
        pytest.skip(f"ORB weights unavailable: {exc}")


@pytest.fixture(scope="module")
def reference(env):
    orbff, adapter = getattr(pretrained, MODEL_NAME)(
        compile=False, device="cuda", precision=PRECISION
    )
    assert orbff.has_stress, f"{MODEL_NAME} does not predict stress"

    def stress(atoms):
        # ORB's adapter rejects mixed periodicity. The slab's vacuum gap (12 A)
        # exceeds the 6 A cutoff, so full periodicity changes no interaction.
        atoms = atoms.copy()
        atoms.pbc = True
        graph = adapter.from_ase_atoms(atoms, device="cuda")
        out = orbff.predict(graph, split=False)
        return voigt_to_full(out[orbff.grad_stress_name].detach().cpu().numpy())

    return stress


def dynamis_stress(model, env, atoms, compute_stress=True):
    state = build_state(raw_from(atoms), float(model.model_radius), env)
    state.sim_config.cutoff = float(model.model_radius)
    model.initialize_buffers(state.local_partition_info, "cuda")
    return model(state, compute_stress=compute_stress)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_stress_matches_orb_predict(model, env, reference):
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
    assert torch.allclose(f0, f1, atol=1e-7)


@pytest.mark.skipif(WORLD_SIZE > 1, reason="single-process test")
def test_slab_in_plane_stress_matches_orb_and_vacuum_axis_is_zero(
    model, env, reference
):
    atoms = slab_silicon()
    _, _, stress = dynamis_stress(model, env, atoms)
    in_plane_stress_ok(stress.cpu().numpy() / EV_TO_MD, reference(atoms))


@pytest.mark.skipif(WORLD_SIZE < 2, reason="needs torchrun --nproc-per-node>=2")
@pytest.mark.parametrize("system", [deformed_silicon, slab_silicon])
def test_two_rank_stress_matches_orb_predict(model, env, reference, system):
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
