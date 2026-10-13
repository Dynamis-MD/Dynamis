#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// constants shared by reader and writer
namespace dynamis_state {

// Checks the ssize_t result of a pread/pwrite call
#define CHECK_READ(result, expected) \
    ::dynamis_state::detail::check_io_result((result), (expected), true, __FILE__, __LINE__)

#define CHECK_WRITE(result, expected) \
    ::dynamis_state::detail::check_io_result((result), (expected), false, __FILE__, __LINE__)

namespace detail {
inline void check_io_result(ssize_t result, uint64_t expected, bool is_read,
                             const char* file, int line) {
    const char* verb = is_read ? "pread" : "pwrite";
    if (result < 0) {
        throw std::runtime_error(std::string(verb) + " failed at " + file + ":" +
                                  std::to_string(line) + ": " + strerror(errno));
    }
    if (static_cast<uint64_t>(result) != expected) {
        throw std::runtime_error(std::string(is_read ? "short read" : "short write") + " at " +
                                  file + ":" + std::to_string(line) + ": expected " +
                                  std::to_string(expected) + " bytes, got " + std::to_string(result));
    }
}
} // namespace detail

// Per-atom field order
struct FieldSpec {
    const char* opt_name;        // name accepted in the Python-facing opts list
    const char* descriptor_name; // name written into the file's field_descriptors block
    uint32_t bytes_per_atom;     // per-atom byte size of this field
};

static const std::vector<FieldSpec> kFieldOrder = {
    { "indices",   "id",       sizeof(int64_t) },
    { "positions", "position", 3 * sizeof(float) },
    { "velocities","velocity", 3 * sizeof(float) },
};

// Not a C string so only compare the 8 bytes even on a corrupt file
static constexpr uint8_t kMagicBytes[8] = { 'D', 'Y', 'N', 'A', 'M', 'I', 'S', 0 };

// cell (9 x f32) + pbc (3 x u8) + padding (1)
static constexpr uint32_t kPbcCellBlockSize = 40;

// Rounds a size up to the next 8-byte boundary padding amount.
inline uint32_t padding_to_8(uint64_t n) {
    return 8 - static_cast<uint32_t>(n % 8);
}

// Fixed byte offsets into the front matter
namespace layout {
static constexpr uint32_t kMagicOffset = 0;
static constexpr uint32_t kMagicSize = 8;
static constexpr uint32_t kVersionOffset = kMagicOffset + kMagicSize;         // 8
static constexpr uint32_t kVersionSize = sizeof(uint32_t);                   // 4
static constexpr uint32_t kRunIdOffset = kVersionOffset + kVersionSize;      // 12
static constexpr uint32_t kRunIdSize = 16;
static constexpr uint32_t kAuthorOffset = kRunIdOffset + kRunIdSize;         // 28
static constexpr uint32_t kAuthorSize = 64;
static constexpr uint32_t kTotalAtomsOffset = kAuthorOffset + kAuthorSize;   // 92
static constexpr uint32_t kTotalAtomsSize = sizeof(uint64_t);                // 8
static constexpr uint32_t kTimePerStepOffset = kTotalAtomsOffset + kTotalAtomsSize; // 100
static constexpr uint32_t kTimePerStepSize = sizeof(float);                  // 4
static constexpr uint32_t kStepsPerWriteOffset = kTimePerStepOffset + kTimePerStepSize; // 104
static constexpr uint32_t kStepsPerWriteSize = sizeof(uint32_t);             // 4
static constexpr uint32_t kBytesPerAtomOffset = kStepsPerWriteOffset + kStepsPerWriteSize; // 108
static constexpr uint32_t kBytesPerAtomSize = sizeof(uint32_t);              // 4
static constexpr uint32_t kFieldCountOffset = kBytesPerAtomOffset + kBytesPerAtomSize; // 112
static constexpr uint32_t kFieldCountSize = sizeof(uint32_t);                // 4
static constexpr uint32_t kFieldDescriptorsOffset = kFieldCountOffset + kFieldCountSize; // 116
static constexpr uint32_t kFieldDescriptorSize = 32;
} // namespace layout

// Atomic number and mass (amu) for one element. Masses mirror
// dynamis_state/masses.py, which is what the integrators use.
struct Element {
    uint8_t number;
    float mass;
};

// TODO figure out a better way to do this
// only doing up to Lead as idk if we need any of the unstable elements
static const std::unordered_map<std::string, Element> periodic_table = {
    {"H",  { 1,    1.008f}},
    {"He", { 2,   4.0026f}},
    {"Li", { 3,     6.94f}},
    {"Be", { 4,   9.0122f}},
    {"B",  { 5,    10.81f}},
    {"C",  { 6,   12.011f}},
    {"N",  { 7,   14.007f}},
    {"O",  { 8,   15.999f}},
    {"F",  { 9,   18.998f}},
    {"Ne", {10,    20.18f}},
    {"Na", {11,    22.99f}},
    {"Mg", {12,   24.305f}},
    {"Al", {13,   26.982f}},
    {"Si", {14,   28.085f}},
    {"P",  {15,   30.974f}},
    {"S",  {16,    32.06f}},
    {"Cl", {17,    35.45f}},
    {"Ar", {18,   39.948f}},
    {"K",  {19,   39.098f}},
    {"Ca", {20,   40.078f}},
    {"Sc", {21,   44.956f}},
    {"Ti", {22,   47.867f}},
    {"V",  {23,   50.942f}},
    {"Cr", {24,   51.996f}},
    {"Mn", {25,   54.938f}},
    {"Fe", {26,   55.845f}},
    {"Co", {27,   58.933f}},
    {"Ni", {28,   58.693f}},
    {"Cu", {29,   63.546f}},
    {"Zn", {30,    65.38f}},
    {"Ga", {31,   69.723f}},
    {"Ge", {32,    72.63f}},
    {"As", {33,   74.922f}},
    {"Se", {34,   78.971f}},
    {"Br", {35,   79.904f}},
    {"Kr", {36,   83.798f}},
    {"Rb", {37,   85.468f}},
    {"Sr", {38,    87.62f}},
    {"Y",  {39,   88.906f}},
    {"Zr", {40,   91.224f}},
    {"Nb", {41,   92.906f}},
    {"Mo", {42,    95.95f}},
    {"Tc", {43,     98.0f}},
    {"Ru", {44,   101.07f}},
    {"Rh", {45,   102.91f}},
    {"Pd", {46,   106.42f}},
    {"Ag", {47,   107.87f}},
    {"Cd", {48,   112.41f}},
    {"In", {49,   114.82f}},
    {"Sn", {50,   118.71f}},
    {"Sb", {51,   121.76f}},
    {"Te", {52,    127.6f}},
    {"I",  {53,    126.9f}},
    {"Xe", {54,   131.29f}},
    {"Cs", {55,   132.91f}},
    {"Ba", {56,   137.33f}},
    {"La", {57,   138.91f}},
    {"Ce", {58,   140.12f}},
    {"Pr", {59,   140.91f}},
    {"Nd", {60,   144.24f}},
    {"Pm", {61,    145.0f}},
    {"Sm", {62,   150.36f}},
    {"Eu", {63,   151.96f}},
    {"Gd", {64,   157.25f}},
    {"Tb", {65,   158.93f}},
    {"Dy", {66,    162.5f}},
    {"Ho", {67,   164.93f}},
    {"Er", {68,   167.26f}},
    {"Tm", {69,   168.93f}},
    {"Yb", {70,   173.05f}},
    {"Lu", {71,   174.97f}},
    {"Hf", {72,   178.49f}},
    {"Ta", {73,   180.95f}},
    {"W",  {74,   183.84f}},
    {"Re", {75,   186.21f}},
    {"Os", {76,   190.23f}},
    {"Ir", {77,   192.22f}},
    {"Pt", {78,   195.08f}},
    {"Au", {79,   196.97f}},
    {"Hg", {80,   200.59f}},
    {"Tl", {81,   204.38f}},
    {"Pb", {82,    207.2f}}
};

inline const std::unordered_map<uint8_t, std::string>& reverse_periodic_table() {
    static const std::unordered_map<uint8_t, std::string> table = [] {
        std::unordered_map<uint8_t, std::string> t;
        for (const auto& [label, element] : periodic_table) {
            t.emplace(element.number, label);
        }
        return t;
    }();
    return table;
}

} // namespace dynamis_state
