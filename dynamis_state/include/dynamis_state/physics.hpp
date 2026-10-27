#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <torch/extension.h>

namespace dynamis_state {

// Boltzmann constant in amu * A^2 / (fs^2 * K)
constexpr double kBoltzmann = 8.314462618e-7;

// One femtosecond in ASE's time unit, A * sqrt(amu / eV) (ase.units.fs).
// ASE momenta are amu * sqrt(eV / amu), so (p / m) * kAseFs is in A / fs.
constexpr double kAseFs = 0.09822694788464063;

// Atomic number and mass (amu) for one element
struct Element {
    uint8_t number;
    float mass;
};

// Single source of element data for the reader, the writer and (through
// mass_table) the integrators
static const std::unordered_map<std::string, Element> periodic_table = {
    { "H", { 1, 1.008f } },    { "He", { 2, 4.0026f } },  { "Li", { 3, 6.94f } },
    { "Be", { 4, 9.0122f } },  { "B", { 5, 10.81f } },    { "C", { 6, 12.011f } },
    { "N", { 7, 14.007f } },   { "O", { 8, 15.999f } },   { "F", { 9, 18.998f } },
    { "Ne", { 10, 20.18f } },  { "Na", { 11, 22.99f } },  { "Mg", { 12, 24.305f } },
    { "Al", { 13, 26.982f } }, { "Si", { 14, 28.085f } }, { "P", { 15, 30.974f } },
    { "S", { 16, 32.06f } },   { "Cl", { 17, 35.45f } },  { "Ar", { 18, 39.948f } },
    { "K", { 19, 39.098f } },  { "Ca", { 20, 40.078f } }, { "Sc", { 21, 44.956f } },
    { "Ti", { 22, 47.867f } }, { "V", { 23, 50.942f } },  { "Cr", { 24, 51.996f } },
    { "Mn", { 25, 54.938f } }, { "Fe", { 26, 55.845f } }, { "Co", { 27, 58.933f } },
    { "Ni", { 28, 58.693f } }, { "Cu", { 29, 63.546f } }, { "Zn", { 30, 65.38f } },
    { "Ga", { 31, 69.723f } }, { "Ge", { 32, 72.63f } },  { "As", { 33, 74.922f } },
    { "Se", { 34, 78.971f } }, { "Br", { 35, 79.904f } }, { "Kr", { 36, 83.798f } },
    { "Rb", { 37, 85.468f } }, { "Sr", { 38, 87.62f } },  { "Y", { 39, 88.906f } },
    { "Zr", { 40, 91.224f } }, { "Nb", { 41, 92.906f } }, { "Mo", { 42, 95.95f } },
    { "Tc", { 43, 98.0f } },   { "Ru", { 44, 101.07f } }, { "Rh", { 45, 102.91f } },
    { "Pd", { 46, 106.42f } }, { "Ag", { 47, 107.87f } }, { "Cd", { 48, 112.41f } },
    { "In", { 49, 114.82f } }, { "Sn", { 50, 118.71f } }, { "Sb", { 51, 121.76f } },
    { "Te", { 52, 127.6f } },  { "I", { 53, 126.9f } },   { "Xe", { 54, 131.29f } },
    { "Cs", { 55, 132.91f } }, { "Ba", { 56, 137.33f } }, { "La", { 57, 138.91f } },
    { "Ce", { 58, 140.12f } }, { "Pr", { 59, 140.91f } }, { "Nd", { 60, 144.24f } },
    { "Pm", { 61, 145.0f } },  { "Sm", { 62, 150.36f } }, { "Eu", { 63, 151.96f } },
    { "Gd", { 64, 157.25f } }, { "Tb", { 65, 158.93f } }, { "Dy", { 66, 162.5f } },
    { "Ho", { 67, 164.93f } }, { "Er", { 68, 167.26f } }, { "Tm", { 69, 168.93f } },
    { "Yb", { 70, 173.05f } }, { "Lu", { 71, 174.97f } }, { "Hf", { 72, 178.49f } },
    { "Ta", { 73, 180.95f } }, { "W", { 74, 183.84f } },  { "Re", { 75, 186.21f } },
    { "Os", { 76, 190.23f } }, { "Ir", { 77, 192.22f } }, { "Pt", { 78, 195.08f } },
    { "Au", { 79, 196.97f } }, { "Hg", { 80, 200.59f } }, { "Tl", { 81, 204.38f } },
    { "Pb", { 82, 207.2f } },  { "Bi", { 83, 208.98f } }, { "Po", { 84, 209.0f } },
    { "At", { 85, 210.0f } },  { "Rn", { 86, 222.0f } },  { "Fr", { 87, 223.0f } },
    { "Ra", { 88, 226.0f } },  { "Ac", { 89, 227.0f } },  { "Th", { 90, 232.04f } },
    { "Pa", { 91, 231.04f } }, { "U", { 92, 238.03f } },  { "Np", { 93, 237.0f } },
    { "Pu", { 94, 244.0f } },  { "Am", { 95, 243.0f } },  { "Cm", { 96, 247.0f } },
    { "Bk", { 97, 247.0f } },  { "Cf", { 98, 251.0f } },  { "Es", { 99, 252.0f } },
    { "Fm", { 100, 257.0f } }, { "Md", { 101, 258.0f } }, { "No", { 102, 259.0f } },
    { "Lr", { 103, 262.0f } }, { "Rf", { 104, 267.0f } }, { "Db", { 105, 270.0f } },
    { "Sg", { 106, 271.0f } }, { "Bh", { 107, 270.0f } }, { "Hs", { 108, 277.0f } },
    { "Mt", { 109, 276.0f } }, { "Ds", { 110, 281.0f } }, { "Rg", { 111, 280.0f } },
    { "Cn", { 112, 285.0f } }, { "Nh", { 113, 284.0f } }, { "Fl", { 114, 289.0f } },
    { "Mc", { 115, 288.0f } }, { "Lv", { 116, 293.0f } }, { "Ts", { 117, 294.0f } },
    { "Og", { 118, 294.0f } }
};

inline const std::unordered_map<uint8_t, std::string> &reverse_periodic_table()
{
    static const std::unordered_map<uint8_t, std::string> table = [] {
        std::unordered_map<uint8_t, std::string> t;
        for (const auto &[label, element] : periodic_table) {
            t.emplace(element.number, label);
        }
        return t;
    }();
    return table;
}

// float32 [119] masses (amu) indexed by atomic number; index 0 is 0
torch::Tensor mass_table();

// Maxwell-Boltzmann velocities (A/fs) at temperature_K with the centre-of-mass
// momentum removed. Returns float32 [N, 3] on atom_types' device; sampling is on
// the CPU from seed alone, so every rank and device draws the same values.
// Throws std::invalid_argument on a negative temperature or an unknown atom type.
torch::Tensor sample_maxwell_boltzmann(const torch::Tensor &atom_types, double temperature_K,
                                       uint64_t seed);

} // namespace dynamis_state
