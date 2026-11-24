"""Unit conventions shared by every Dynamis NNP backend.

Dynamis's MD stack works in atomic mass units, Angstrom and femtoseconds:
``dynamis_integrator``'s Verlet/Langevin kernels integrate ``v += f / m * dt``
with masses taken from ``dynamis_state.ATOMIC_MASSES`` (amu), positions in
Angstrom and ``dt`` in fs. Forces handed to an integrator must therefore be
amu * Angstrom / fs^2, and energies amu * Angstrom^2 / fs^2.

Public NNP checkpoints (UMA, ORB, MACE) all predict in eV and eV/Angstrom, so
every :class:`~dynamis_model.potential.DynamisModel` backend multiplies *both*
its energy and its forces by :data:`EV_PER_A_TO_AMU_A_PER_FS2` before returning
them. Keep that conversion here rather than re-spelling the literal per
backend: a backend that returns raw eV looks correct in any parity test that
compares it against an eV reference, then silently produces forces ~103.6x too
large the moment it reaches the integrator.
"""

# eV -> amu * A^2 / fs^2, equivalently eV/A -> amu * A / fs^2.
EV_PER_A_TO_AMU_A_PER_FS2 = 0.0096485332

__all__ = ["EV_PER_A_TO_AMU_A_PER_FS2"]
