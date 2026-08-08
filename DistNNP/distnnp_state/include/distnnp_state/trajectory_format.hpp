#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// constants shared by reader and writer
namespace distnnp_state {

// Checks the ssize_t result of a pread/pwrite call
#define CHECK_READ(result, expected) \
    ::distnnp_state::detail::check_io_result((result), (expected), true, __FILE__, __LINE__)

#define CHECK_WRITE(result, expected) \
    ::distnnp_state::detail::check_io_result((result), (expected), false, __FILE__, __LINE__)

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
static constexpr uint8_t kMagicBytes[8] = { 'D', 'I', 'S', 'T', 'N', 'N', 'P', 0 };

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

// TODO figure out a better way to do this
// only doing up to Lead as idk if we need any of the unstable elements
static const std::unordered_map<std::string, uint8_t> periodic_table = {
    {"H", 1},
    {"He", 2},
    {"Li", 3},
    {"Be", 4},
    {"B", 5},
    {"C", 6},
    {"N", 7},
    {"O", 8},
    {"F", 9},
    {"Ne", 10},
    {"Na", 11},
    {"Mg", 12},
    {"Al", 13},
    {"Si", 14},
    {"P", 15},
    {"S", 16},
    {"Cl", 17},
    {"Ar", 18},
    {"K", 19},
    {"Ca", 20},
    {"Sc", 21},
    {"Ti", 22},
    {"V", 23},
    {"Cr", 24},
    {"Mn", 25},
    {"Fe", 26},
    {"Co", 27},
    {"Ni", 28},
    {"Cu", 29},
    {"Zn", 30},
    {"Ga", 31},
    {"Ge", 32},
    {"As", 33},
    {"Se", 34},
    {"Br", 35},
    {"Kr", 36},
    {"Rb", 37},
    {"Sr", 38},
    {"Y", 39},
    {"Zr", 40},
    {"Nb", 41},
    {"Mo", 42},
    {"Tc", 43},
    {"Ru", 44},
    {"Rh", 45},
    {"Pd", 46},
    {"Ag", 47},
    {"Cd", 48},
    {"In", 49},
    {"Sn", 50},
    {"Sb", 51},
    {"Te", 52},
    {"I", 53},
    {"Xe", 54},
    {"Cs", 55},
    {"Ba", 56},
    {"La", 57},
    {"Ce", 58},
    {"Pr", 59},
    {"Nd", 60},
    {"Pm", 61},
    {"Sm", 62},
    {"Eu", 63},
    {"Gd", 64},
    {"Tb", 65},
    {"Dy", 66},
    {"Ho", 67},
    {"Er", 68},
    {"Tm", 69},
    {"Yb", 70},
    {"Lu", 71},
    {"Hf", 72},
    {"Ta", 73},
    {"W", 74},
    {"Re", 75},
    {"Os", 76},
    {"Ir", 77},
    {"Pt", 78},
    {"Au", 79},
    {"Hg", 80},
    {"Ti", 81},
    {"Pb", 82}
};

inline const std::unordered_map<uint8_t, std::string>& reverse_periodic_table() {
    static const std::unordered_map<uint8_t, std::string> table = [] {
        std::unordered_map<uint8_t, std::string> t;
        for (const auto& [label, number] : periodic_table) {
            t.emplace(static_cast<uint8_t>(number), label);
        }
        return t;
    }();
    return table;
}

} // namespace distnnp_state
