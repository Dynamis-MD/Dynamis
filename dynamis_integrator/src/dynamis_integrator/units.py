"""Thermodynamic units in Dynamis's MD system (amu, Angstrom, femtosecond).

Energy is amu * A^2 / fs^2 and pressure (energy / volume) is amu / (A * fs^2), the
units in which the models return energies, forces and stress. Derived from the
eV <-> MD factor in ``dynamis_model.units`` so there is one source of truth.
"""

from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2

# Boltzmann constant in eV/K (CODATA 2018) -> MD energy per kelvin.
KB = 8.617333262e-5 * EV_PER_A_TO_AMU_A_PER_FS2

# 1 GPa = 1e9 J/m^3 = 6.241509074e-3 eV/A^3 -> MD pressure.
GPA = 6.241509074e-3 * EV_PER_A_TO_AMU_A_PER_FS2
BAR = 1e-4 * GPA

__all__ = ["BAR", "GPA", "KB"]
