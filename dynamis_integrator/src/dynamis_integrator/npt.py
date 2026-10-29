"""NPT dynamics: Langevin thermostat (BBK) plus the stochastic cell-rescaling barostat.

The barostat is C-rescale (Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020)). Per
coupling it draws a log-strain for the periodic axes

    x = -beta dt / (d tau_p) (P0 - P)  +  sqrt(2 kT beta dt / (V tau_p)) xi / d,   xi ~ N(0, 1)

with beta the compressibility, tau_p the pressure time, V the cell volume, d the number
of periodic axes and P the instantaneous internal pressure, then scales positions and
cell by mu = exp(x) and velocities by 1 / mu. It samples the isothermal-isobaric
ensemble (the stochastic term restores the volume fluctuations Berendsen suppresses),
is first order (no piston mass, no oscillation) and needs no state beyond the cell.

Units are the MD system (amu, A, fs); see :mod:`dynamis_integrator.units` for ``KB``,
``GPA`` and ``BAR``. Pressure is ``P = (sum m v v^T) / V - stress``, so the model must
return its stress (``compute_stress=True``); the stress is already summed over ranks, and
the kinetic tensor is the only thing this integrator reduces.

Step (``Simulator`` calls this once per MD step)::

    half-step 1 (Langevin: half kick + full drift)
    barostat    (every ``pressure_interval`` steps; scales x, cell, v on the cached P)
    model call  (on the rescaled positions, so forces match; stress on coupling steps)
    half-step 2 (Langevin: half kick)
    cache P     (kinetic tensor of the new velocities and the stress just computed)

Every rank draws the same xi (a generator seeded identically), so the box stays
bit-identical across ranks. Nothing reads a scalar to the host on the hot path.
"""

import math
import warnings
from typing import Optional

import torch
import torch.distributed as dist

from dynamis_integrator._data_check import check_mass_shape
from dynamis_integrator.langevin import Langevin
from dynamis_integrator.masses import TORCH_MASSES
from dynamis_model import DynamisModel
from dynamis_state import DynamisState, deform_state


def cell_volume(cell: torch.Tensor) -> torch.Tensor:
    """|det| of a [3, 3] cell as a float64 scalar, by triple product: elementwise, so
    no linear-algebra call and no host synchronisation."""
    h = cell.reshape(3, 3).to(torch.float64)
    return torch.dot(h[0], torch.linalg.cross(h[1], h[2])).abs()


def pressure_tensor(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    stress: torch.Tensor,
    cell: torch.Tensor,
    reduce: bool = True,
):
    """Instantaneous pressure tensor ``(sum m v v^T) / V - stress`` and the volume.

    ``stress`` is the model's (tensile-positive, already rank-summed) stress. The local
    kinetic tensor is summed over ranks here when a multi-rank group is live; pass
    ``reduce=False`` when ``velocities`` already cover every atom.
    """
    kinetic = ((velocities * masses[:, None]).T @ velocities).to(torch.float64)
    if reduce and dist.is_available() and dist.is_initialized():
        if dist.get_world_size() > 1:
            dist.all_reduce(kinetic, op=dist.ReduceOp.SUM)
    volume = cell_volume(cell)
    return kinetic / volume - stress.to(torch.float64), volume


def crescale_log_strain(
    pressure: torch.Tensor,
    target_pressure: float,
    volume: torch.Tensor,
    kT: float,
    compressibility: float,
    tau_p: float,
    dt: float,
    xi: torch.Tensor,
    ndim: int,
) -> torch.Tensor:
    """The C-rescale log-strain ``x`` applied to each periodic axis (see module doc)."""
    drift = -compressibility * dt / (ndim * tau_p) * (target_pressure - pressure)
    noise = torch.sqrt(2.0 * kT * compressibility * dt / tau_p / volume) * xi / ndim
    return drift + noise


class NPT(Langevin):
    """Langevin dynamics with a C-rescale barostat. See the module docstring.

    Args:
        dt: Time step in fs.
        gamma: Langevin friction in 1/fs.
        kT: Thermal energy in MD units (``temperature_K * dynamis_integrator.units.KB``).
        pressure: Target pressure in MD units (e.g. ``1 * units.BAR``).
        tau_p: Barostat time constant in fs; the response time is set by this and
            ``compressibility`` together.
        compressibility: Isothermal compressibility in 1 / (MD pressure), e.g.
            ``1e-6 / units.BAR`` for a stiff solid like Si (bulk modulus ~100 GPa),
            ``4.5e-5 / units.BAR`` for water. With ``tau_p`` it sets the relaxation
            time of the volume, ``tau_p / (compressibility * bulk_modulus)``; it does
            not change the ensemble.
        pressure_interval: Couple every this many steps (the stress, which costs a
            little extra in the model's backward pass, is only computed on those steps).
        max_strain_per_step: Safeguard clamp on ``|ln mu|`` per coupling. It also bounds
            the box change the drift watchdog must look ahead over
            (``cell_change_bound_per_step``). Active only in the far tail of the noise.
        seed: Seed of the (rank-shared) generator behind the stochastic term. ``None``
            draws one seed on rank 0 and broadcasts it, so ranks always agree.

    Only fully or partly periodic systems make sense; with fewer than three periodic axes
    only those axes are coupled and the cell volume (which includes the vacuum) sets the
    noise amplitude, so prefer 3D-periodic systems.
    """

    def __init__(
        self,
        dt: float,
        gamma: float,
        kT: float,
        pressure: float,
        tau_p: float,
        compressibility: float,
        pressure_interval: int = 1,
        max_strain_per_step: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(dt=dt, gamma=gamma, kT=kT)
        for name, value in (
            ("tau_p", tau_p),
            ("compressibility", compressibility),
            ("max_strain_per_step", max_strain_per_step),
        ):
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if int(pressure_interval) < 1:
            raise ValueError("pressure_interval must be >= 1")
        self.pressure = float(pressure)
        self.tau_p = float(tau_p)
        self.compressibility = float(compressibility)
        self.pressure_interval = int(pressure_interval)
        self.max_strain_per_step = float(max_strain_per_step)
        self.seed = seed
        # ||A - I||_F of one coupling is at most sqrt(3) (exp(s) - 1) for |ln mu| <= s;
        # the drift watchdog adds half of this times the halo radius to its look-ahead.
        self.cell_change_bound_per_step = math.sqrt(3.0) * math.expm1(
            self.max_strain_per_step
        )
        self._generator: Optional[torch.Generator] = None
        self._ndim: Optional[int] = None
        self._count = 0
        self._pressure_tensor: Optional[torch.Tensor] = None
        self._volume: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self.last_log_strain: Optional[torch.Tensor] = None

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        # The cached pressure is a global quantity and stays valid across a
        # repartition, but the forces it was paired with are rebuilt from scratch.
        super().reset()
        self._pressure_tensor = None
        self._volume = None

    @property
    def instantaneous_pressure(self) -> Optional[torch.Tensor]:
        """Scalar mean of the periodic diagonal of the last cached pressure tensor, in
        MD units (a device tensor; convert with ``/ units.GPA``)."""
        if self._pressure_tensor is None:
            return None
        return self._scalar_pressure(self._pressure_tensor, self._mask)

    # -- helpers -------------------------------------------------------------

    def _periodic_axes(self, dist_state: DynamisState) -> torch.Tensor:
        pbc = dist_state.sim_config.pbc.reshape(3)
        if self._ndim is None:  # one host read, on first use
            self._ndim = int(pbc.sum().item())
            if self._ndim == 0:
                raise ValueError("NPT needs at least one periodic axis (pbc is all false)")
            if self._ndim < 3:
                warnings.warn(
                    f"NPT is coupling only the {self._ndim} periodic axes; the cell "
                    "volume, which includes the vacuum, normalises the noise and the "
                    "pressure.",
                    RuntimeWarning,
                )
        return pbc.to(torch.float64)

    def _scalar_pressure(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tensor.diagonal() * mask).sum() / self._ndim

    def _rank_shared_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None:
            seed = self.seed
            if seed is None:
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    shared = torch.tensor([seed], dtype=torch.int64, device=device)
                    dist.broadcast(shared, src=0)
                    seed = int(shared.item())
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(seed))
        return self._generator

    def _evaluate(self, dist_state, model, want_stress: bool):
        out = self._compute_force(dist_state, model, compute_stress=want_stress)
        if want_stress:
            if len(out) != 3:
                raise TypeError(
                    "NPT needs a model that returns the stress with compute_stress=True "
                    f"({type(model).__name__} returned {len(out)} values)"
                )
            return out
        return out[0], out[1], None

    def _cache_pressure(self, dist_state, masses, stress) -> None:
        local = dist_state.local_molecular_info
        self._mask = self._periodic_axes(dist_state)
        self._pressure_tensor, self._volume = pressure_tensor(
            local.atom_velocities, masses, stress, dist_state.sim_config.cell
        )

    def _rescale(self, dist_state: DynamisState, dt: float) -> None:
        """Apply one C-rescale coupling of duration ``dt`` on the cached pressure."""
        mask = self._mask
        pressure = self._scalar_pressure(self._pressure_tensor, mask)
        xi = torch.randn(
            (), generator=self._rank_shared_generator(mask.device), device=mask.device,
            dtype=torch.float64,
        )
        volume = cell_volume(dist_state.sim_config.cell)
        strain = crescale_log_strain(
            pressure, self.pressure, volume, self.kT, self.compressibility,
            self.tau_p, dt, xi, self._ndim,
        ).clamp(-self.max_strain_per_step, self.max_strain_per_step)
        scale = torch.where(mask > 0, torch.exp(strain), torch.ones_like(mask))
        deform_state(dist_state, torch.diag(scale))
        velocities = dist_state.local_molecular_info.atom_velocities
        velocities.mul_((1.0 / scale).to(velocities.dtype))
        self.last_log_strain = strain

    # -- the step ------------------------------------------------------------

    def step(self, dist_state: DynamisState, model: DynamisModel, dt=None):
        """One NPT step. Returns ``(energy, forces)`` like the other integrators; the
        pressure, volume and last log-strain are available as attributes."""
        state = dist_state.local_molecular_info
        step_size = self.dt if dt is None else dt
        masses = TORCH_MASSES.to(state.atom_positions.device)[
            state.atom_types.reshape(-1).to(torch.long)
        ]
        check_mass_shape(masses, state.atom_positions)

        if self.prev_forces is None:  # first step, or after a repartition
            _, self.prev_forces, stress = self._evaluate(dist_state, model, True)
            self._cache_pressure(dist_state, masses, stress)

        self.halfstep1_(
            state.atom_positions, state.atom_velocities, self.prev_forces, masses,
            torch.randn_like(state.atom_velocities), step_size,
        )

        if self._count % self.pressure_interval == 0:
            self._rescale(dist_state, step_size * self.pressure_interval)

        want_stress = (self._count + 1) % self.pressure_interval == 0
        energy, forces, stress = self._evaluate(dist_state, model, want_stress)

        self.halfstep2_(
            state.atom_velocities, forces, masses,
            torch.randn_like(state.atom_velocities), step_size,
        )
        if want_stress:
            self._cache_pressure(dist_state, masses, stress)

        self.prev_forces = forces
        self._count += 1
        return energy, forces
