"""NPT (Langevin + C-rescale): the formulas, the bookkeeping, and the ensemble.

The ensemble test is the one that matters. For an ideal gas the NPT volume
distribution is known exactly, V ~ Gamma(N + 1, kT / P0): mean (N + 1) kT / P0 and
variance (N + 1) (kT / P0)^2. Reproducing the *variance* is what separates C-rescale
from a weak-coupling (Berendsen) barostat, which suppresses volume fluctuations.
"""

import math

import pytest
import torch

from dynamis_integrator import NPT
from dynamis_integrator.npt import cell_volume, crescale_log_strain, pressure_tensor
from dynamis_integrator.units import BAR, GPA, KB
from dynamis_state import DynamisState, MolecularInfo, SimulationConfig

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
KT = 300.0 * KB
ARGON = 39.948


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_state(n=64, volume=4000.0, pbc=(True, True, True), device="cuda", seed=0):
    torch.manual_seed(seed)
    side = volume ** (1 / 3)
    mol = MolecularInfo()
    mol.num_atoms = n
    mol.atom_positions = torch.rand(n, 3, device=device) * side
    mol.atom_velocities = torch.randn(n, 3, device=device) * (KT / ARGON) ** 0.5
    mol.atom_types = torch.full((n,), 18, dtype=torch.long, device=device)
    sim = SimulationConfig()
    sim.cutoff = 3.5
    sim.ghost_cutoff_buffer = 0.5
    sim.cell = torch.eye(3, device=device) * side
    sim.pbc = torch.tensor(list(pbc), device=device)
    state = DynamisState()
    state.local_molecular_info = mol
    state.sim_config = sim
    return state


class IdealGas:
    """No interactions: zero energy, forces and stress. Records each request."""

    def __init__(self, with_stress=True):
        self.with_stress = with_stress
        self.requests = []

    def __call__(self, state, *args, compute_stress=False, **kwargs):
        self.requests.append(compute_stress)
        pos = state.local_molecular_info.atom_positions
        out = (torch.zeros((), device=pos.device), torch.zeros_like(pos))
        if compute_stress and self.with_stress:
            return out + (torch.zeros(3, 3, device=pos.device),)
        return out


def make_npt(P0=1.0 * BAR, beta=1e-7 / BAR, **kwargs):
    defaults = dict(dt=1.0, gamma=0.02, kT=KT, pressure=P0, tau_p=1000.0, compressibility=beta)
    defaults.update(kwargs)
    return NPT(**defaults)


# ---------------------------------------------------------------------------
# formulas
# ---------------------------------------------------------------------------


def test_log_strain_deterministic_part_has_the_right_sign_and_size():
    kwargs = dict(
        target_pressure=1.0, volume=torch.tensor(1000.0, dtype=torch.float64), kT=0.0,
        compressibility=0.5, tau_p=10.0, dt=2.0, xi=torch.tensor(0.0, dtype=torch.float64),
        ndim=3,
    )
    over = crescale_log_strain(torch.tensor(3.0, dtype=torch.float64), **kwargs)
    under = crescale_log_strain(torch.tensor(-1.0, dtype=torch.float64), **kwargs)
    # x = -beta dt / (d tau_p) (P0 - P)
    assert over.item() == pytest.approx(-0.5 * 2.0 / (3 * 10.0) * (1.0 - 3.0))
    assert over > 0 > under  # over-pressure expands, under-pressure contracts
    assert crescale_log_strain(
        torch.tensor(1.0, dtype=torch.float64), **kwargs
    ).item() == 0.0


def test_log_strain_noise_scales_as_sqrt_kT_beta_dt_over_V_tau():
    def noise(**over):
        kwargs = dict(
            target_pressure=0.0, volume=torch.tensor(500.0, dtype=torch.float64),
            kT=2.0, compressibility=0.3, tau_p=7.0, dt=1.5,
            xi=torch.tensor(1.0, dtype=torch.float64), ndim=3,
        )
        kwargs.update(over)
        return crescale_log_strain(torch.tensor(0.0, dtype=torch.float64), **kwargs).item()

    base = noise()
    assert base == pytest.approx(math.sqrt(2 * 2.0 * 0.3 * 1.5 / (500.0 * 7.0)) / 3)
    assert noise(volume=torch.tensor(2000.0, dtype=torch.float64)) == pytest.approx(base / 2)
    assert noise(dt=6.0) == pytest.approx(base * 2)
    assert noise(ndim=2) == pytest.approx(base * 3 / 2)


def test_pressure_tensor_is_kinetic_over_volume_minus_stress():
    velocities = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = torch.tensor([2.0, 3.0])
    stress = torch.diag(torch.tensor([0.5, -0.25, 0.0]))
    cell = torch.eye(3) * 2.0
    pressure, volume = pressure_tensor(velocities, masses, stress, cell, reduce=False)
    assert volume.item() == pytest.approx(8.0)
    kinetic = torch.diag(torch.tensor([2.0 * 1.0, 3.0 * 4.0, 0.0], dtype=torch.float64))
    assert torch.allclose(pressure, kinetic / 8.0 - stress.double())
    # tensile stress (positive) lowers the pressure; ideal gas: P = N kT / V
    assert pressure[0, 0] < kinetic[0, 0] / 8.0


def test_cell_volume_of_a_triclinic_cell():
    cell = torch.tensor([[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.7, 5.0]])
    assert cell_volume(cell).item() == pytest.approx(60.0)


def test_constructor_rejects_nonsense():
    for bad in (dict(tau_p=0.0), dict(compressibility=-1.0), dict(max_strain_per_step=0.0)):
        with pytest.raises(ValueError):
            make_npt(**bad)
    with pytest.raises(ValueError):
        make_npt(pressure_interval=0)


def test_cell_change_bound_matches_the_clamp():
    npt = make_npt(max_strain_per_step=0.01)
    assert npt.cell_change_bound_per_step == pytest.approx(math.sqrt(3) * math.expm1(0.01))


# ---------------------------------------------------------------------------
# one coupling, on the device
# ---------------------------------------------------------------------------


def prepare(npt, state, pressure_value):
    npt._mask = npt._periodic_axes(state)
    npt._pressure_tensor = torch.eye(3, dtype=torch.float64, device="cuda") * pressure_value


@requires_cuda
def test_rescale_scales_positions_and_cell_up_and_velocities_down():
    state = make_state()
    npt = make_npt(seed=5, pressure=0.0, compressibility=1e4, tau_p=10.0)
    x0 = state.local_molecular_info.atom_positions.clone()
    v0 = state.local_molecular_info.atom_velocities.clone()
    h0 = state.sim_config.cell.clone()
    prepare(npt, state, 50.0 * BAR)

    gen = torch.Generator(device="cuda").manual_seed(5)
    xi = torch.randn((), generator=gen, device="cuda", dtype=torch.float64)
    expected = crescale_log_strain(
        torch.tensor(50.0 * BAR, dtype=torch.float64, device="cuda"), 0.0,
        cell_volume(h0), KT, 1e4, 10.0, 1.0, xi, 3,
    ).clamp(-npt.max_strain_per_step, npt.max_strain_per_step)
    mu = torch.exp(expected).item()
    assert mu != 1.0

    npt._rescale(state, 1.0)
    assert npt.last_log_strain.item() == pytest.approx(expected.item())
    torch.testing.assert_close(state.local_molecular_info.atom_positions, x0 * mu, rtol=1e-6, atol=0)
    torch.testing.assert_close(state.sim_config.cell, h0 * mu, rtol=1e-6, atol=0)
    torch.testing.assert_close(state.local_molecular_info.atom_velocities, v0 / mu, rtol=1e-6, atol=0)


@requires_cuda
def test_rescale_is_clamped_to_the_safeguard():
    state = make_state()
    npt = make_npt(seed=1, pressure=0.0, compressibility=1e12, tau_p=1.0, max_strain_per_step=1e-3)
    prepare(npt, state, 100.0 * GPA)
    npt._rescale(state, 1.0)
    assert abs(npt.last_log_strain.item()) == pytest.approx(1e-3)


@requires_cuda
def test_a_slab_couples_only_its_periodic_axes():
    state = make_state(pbc=(True, True, False))
    npt = make_npt(seed=2, pressure=0.0, compressibility=1e4, tau_p=10.0)
    x0 = state.local_molecular_info.atom_positions.clone()
    v0 = state.local_molecular_info.atom_velocities.clone()
    h0 = state.sim_config.cell.clone()
    with pytest.warns(RuntimeWarning, match="periodic axes"):
        prepare(npt, state, 50.0 * BAR)
    npt._rescale(state, 1.0)
    pos = state.local_molecular_info.atom_positions
    assert not torch.equal(pos[:, 0], x0[:, 0])
    assert torch.equal(pos[:, 2], x0[:, 2])  # the vacuum axis is untouched
    assert torch.equal(state.sim_config.cell[2, 2], h0[2, 2])
    assert torch.equal(state.local_molecular_info.atom_velocities[:, 2], v0[:, 2])


@requires_cuda
def test_a_fully_non_periodic_system_is_refused():
    state = make_state(pbc=(False, False, False))
    with pytest.raises(ValueError, match="periodic"):
        make_npt()._periodic_axes(state)


# ---------------------------------------------------------------------------
# the step's bookkeeping
# ---------------------------------------------------------------------------


@requires_cuda
def test_stress_is_requested_only_on_coupling_steps():
    state = make_state()
    model = IdealGas()
    npt = make_npt(pressure_interval=3)
    rescales = []
    original = npt._rescale
    npt._rescale = lambda *a, **k: (rescales.append(npt._count), original(*a, **k))
    for _ in range(6):
        npt.step(state, model)
    # initial evaluation, then one request per step; coupling at counts 0 and 3 consumes
    # the pressure cached at the end of the step before (counts 2 and 5 request it)
    assert model.requests == [True, False, False, True, False, False, True]
    assert rescales == [0, 3]


@requires_cuda
def test_reset_forces_a_fresh_evaluation_with_stress():
    state = make_state()
    model = IdealGas()
    npt = make_npt(pressure_interval=4)
    npt.step(state, model)
    n = len(model.requests)
    npt.reset()
    npt.step(state, model)
    assert model.requests[n] is True  # the post-reset evaluation recaches the pressure


@requires_cuda
def test_a_model_that_returns_no_stress_gets_a_clear_error():
    with pytest.raises(TypeError, match="stress"):
        make_npt().step(make_state(), IdealGas(with_stress=False))


@requires_cuda
def test_the_box_is_a_deterministic_function_of_the_seed():
    def run(seed):
        torch.manual_seed(0)
        state = make_state()
        npt = make_npt(seed=seed, tau_p=50.0, compressibility=1e4, pressure=1e-6)
        for _ in range(20):
            npt.step(state, IdealGas())
        return state.sim_config.cell.clone()

    assert torch.equal(run(3), run(3))
    assert not torch.equal(run(3), run(4))


# ---------------------------------------------------------------------------
# the ensemble
# ---------------------------------------------------------------------------


@requires_cuda
def test_ideal_gas_samples_the_isothermal_isobaric_volume_distribution():
    """V ~ Gamma(N + 1, kT / P0). Mean tolerance is several standard errors; the
    variance ratio separates C-rescale (1.0) from a Berendsen-like barostat (~0.13)."""
    n, v0, steps, seed = 64, 4000.0, 20000, 1
    p0 = (n + 1) * KT / v0
    torch.manual_seed(seed)
    state = make_state(n=n, volume=v0, seed=seed)
    npt = NPT(
        dt=1.0, gamma=0.02, kT=KT, pressure=p0, tau_p=100.0, compressibility=1.0 / p0,
        max_strain_per_step=0.05, seed=seed,
    )
    model = IdealGas()
    volumes = torch.empty(steps, dtype=torch.float64)
    for i in range(steps):
        npt.step(state, model)
        volumes[i] = cell_volume(state.sim_config.cell).item()
    v = volumes[steps // 10 :]  # drop the burn-in

    expected_mean, expected_var = (n + 1) * KT / p0, (n + 1) * (KT / p0) ** 2
    assert v.mean().item() == pytest.approx(expected_mean, rel=0.05)
    assert 0.7 < v.var().item() / expected_var < 1.4
