"""Convention checks for dynamis_model.strain on a periodic toy pair potential.

The toy energy sums a smooth short-range pair term over explicit periodic
images, so the strain has to flow through both the positions and the
``shift @ cell`` term, as it does in the NNP backends. Three independent
references pin the result: central finite differences of the energy under
explicit strains, the analytic pair virial, and the sign of the pressure.
"""

import itertools

import pytest
import torch
import torch.distributed as dist

from dynamis_model.strain import (
    apply_strain,
    strain_grad_from_gradients,
    stress_from_strain_grad,
    zero_strain,
)

DTYPE = torch.float64
RC = 2.5
SHIFTS = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=DTYPE)


def pair_energy(positions, cell):
    """0.5 * sum over ordered pairs and images of (rc - r)^3 for r < rc."""
    n = positions.shape[0]
    dr = positions[None, :, None, :] - positions[:, None, None, :]
    dr = dr + (SHIFTS @ cell)[None, None, :, :]
    r = torch.linalg.norm(dr, dim=-1)
    self_image = torch.eye(n, dtype=torch.bool)[:, :, None] & (
        (SHIFTS == 0).all(dim=1)[None, None, :]
    )
    phi = (RC - r).clamp(min=0.0) ** 3
    return 0.5 * phi.masked_fill(self_image, 0.0).sum()


def analytic_stress(positions, cell):
    """(1/V) * 0.5 * sum phi'(r) r_a r_b / r over ordered pairs and images."""
    n = positions.shape[0]
    dr = positions[None, :, None, :] - positions[:, None, None, :]
    dr = (dr + (SHIFTS @ cell)[None, None, :, :]).reshape(-1, 3)
    r = torch.linalg.norm(dr, dim=-1)
    keep = (r > 1e-12) & (r < RC)
    dr, r = dr[keep], r[keep]
    dphi = -3.0 * (RC - r) ** 2
    virial = 0.5 * torch.einsum("e,ea,eb->ab", dphi / r, dr, dr)
    return virial / torch.linalg.det(cell).abs()


@pytest.fixture
def system():
    torch.manual_seed(0)
    # cell edge > 2 * rc so the 27 explicit images are a complete neighbour set
    cell = torch.tensor(
        [[5.4, 0.0, 0.0], [0.4, 5.6, 0.0], [-0.3, 0.5, 5.8]], dtype=DTYPE
    )
    frac = torch.rand(12, 3, dtype=DTYPE)
    return frac @ cell, cell


def autograd_stress(positions, cell):
    strain = zero_strain("cpu", DTYPE)
    strained_pos, strained_cell = apply_strain(positions, cell, strain)
    energy = pair_energy(strained_pos, strained_cell)
    (grad,) = torch.autograd.grad(energy, [strain])
    # the toy gradient already covers the whole system: never all-reduce it
    return stress_from_strain_grad(grad, cell, reduce=False)


def test_matches_finite_difference_of_energy(system):
    positions, cell = system
    stress = autograd_stress(positions, cell)
    volume = torch.linalg.det(cell).abs()
    h = 1e-5
    for _ in range(4):
        delta = torch.randn(3, 3, dtype=DTYPE)
        delta = 0.5 * (delta + delta.T)
        plus = pair_energy(*apply_strain(positions, cell, h * delta))
        minus = pair_energy(*apply_strain(positions, cell, -h * delta))
        fd = (plus - minus) / (2 * h)
        assert torch.allclose(fd, (stress * delta).sum() * volume, rtol=1e-6, atol=1e-8)


def test_matches_analytic_pair_virial(system):
    positions, cell = system
    assert torch.allclose(
        autograd_stress(positions, cell),
        analytic_stress(positions, cell),
        rtol=1e-9,
        atol=1e-12,
    )


def test_stress_is_symmetric(system):
    stress = autograd_stress(*system)
    assert torch.allclose(stress, stress.T, atol=1e-12)


def test_repulsive_pair_potential_has_positive_pressure(system):
    stress = autograd_stress(*system)
    assert -torch.trace(stress) / 3 > 0


def test_zero_strain_leaves_forces_unchanged(system):
    positions, cell = system
    plain = positions.clone().requires_grad_(True)
    (grad_plain,) = torch.autograd.grad(pair_energy(plain, cell), [plain])

    leaf = positions.clone().requires_grad_(True)
    strain = zero_strain("cpu", DTYPE)
    strained_pos, strained_cell = apply_strain(leaf, cell, strain)
    assert torch.equal(strained_pos, positions)
    assert torch.equal(strained_cell, cell)
    (grad_strained,) = torch.autograd.grad(
        pair_energy(strained_pos, strained_cell), [leaf]
    )
    assert torch.allclose(grad_strained, grad_plain, atol=1e-14)


def test_single_process_reduce_is_a_noop(system):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        pytest.skip("a live multi-rank process group makes the default reduce real")
    _, cell = system
    grad = torch.eye(3, dtype=DTYPE)
    volume = torch.linalg.det(cell).abs()
    assert torch.allclose(stress_from_strain_grad(grad, cell), grad / volume)


def masked_stress(positions, cell, pbc):
    strain = zero_strain("cpu", DTYPE)
    strained_pos, strained_cell = apply_strain(positions, cell, strain, pbc)
    (grad,) = torch.autograd.grad(pair_energy(strained_pos, strained_cell), [strain])
    return stress_from_strain_grad(grad, cell, reduce=False)


def test_pbc_all_true_matches_unmasked(system):
    assert torch.equal(
        masked_stress(*system, (True, True, True)), autograd_stress(*system)
    )


def test_slab_mask_zeroes_vacuum_components_and_keeps_in_plane_block(system):
    full = autograd_stress(*system)
    slab = masked_stress(*system, (True, True, False))
    assert torch.count_nonzero(slab[2, :]) == 0
    assert torch.count_nonzero(slab[:, 2]) == 0
    assert torch.allclose(slab[:2, :2], full[:2, :2])


def test_no_periodic_axis_gives_zero_stress_and_untouched_inputs(system):
    positions, cell = system
    strain = zero_strain("cpu", DTYPE)
    strained_pos, strained_cell = apply_strain(
        positions, cell, strain, (False, False, False)
    )
    assert torch.equal(strained_pos, positions)
    assert torch.equal(strained_cell, cell)
    assert torch.count_nonzero(masked_stress(positions, cell, (False,) * 3)) == 0


def gradient_route_strain_grad(positions, cell, pbc):
    """dE/d eps from dE/dx and dE/dcell, as a backend that cannot hold a strain
    leaf would assemble it."""
    x = positions.clone().requires_grad_(True)
    h = cell.clone().requires_grad_(True)
    g_x, g_h = torch.autograd.grad(pair_energy(x, h), [x, h])
    return strain_grad_from_gradients(positions, g_x, cell, g_h, pbc)


@pytest.mark.parametrize("pbc", [None, (True, True, False), (False, False, False)])
def test_gradient_route_equals_strain_leaf(system, pbc):
    positions, cell = system
    strain = zero_strain("cpu", DTYPE)
    strained_pos, strained_cell = apply_strain(positions, cell, strain, pbc)
    (leaf,) = torch.autograd.grad(pair_energy(strained_pos, strained_cell), [strain])
    assert torch.allclose(
        gradient_route_strain_grad(positions, cell, pbc), leaf, rtol=1e-10, atol=1e-12
    )
