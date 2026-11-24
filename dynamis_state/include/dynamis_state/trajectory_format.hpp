#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// constants shared by reader and writer
namespace dynamis_state {

// Checks the ssize_t result of a pread/pwrite call
#define CHECK_READ(result, expected)                                                               \
    ::dynamis_state::detail::check_io_result((result), (expected), true, __FILE__, __LINE__)

#define CHECK_WRITE(result, expected)                                                              \
    ::dynamis_state::detail::check_io_result((result), (expected), false, __FILE__, __LINE__)

namespace detail {
inline void check_io_result(ssize_t result, uint64_t expected, bool is_read, const char *file,
                            int line)
{
    const char *verb = is_read ? "pread" : "pwrite";
    if (result < 0) {
        throw std::runtime_error(std::string(verb) + " failed at " + file + ":" +
                                 std::to_string(line) + ": " + strerror(errno));
    }
    if (static_cast<uint64_t>(result) != expected) {
        throw std::runtime_error(std::string(is_read ? "short read" : "short write") + " at " +
                                 file + ":" + std::to_string(line) + ": expected " +
                                 std::to_string(expected) + " bytes, got " +
                                 std::to_string(result));
    }
}
} // namespace detail

// Per-atom field order
struct FieldSpec {
    const char *opt_name; // name accepted in the Python-facing opts list
    const char *descriptor_name; // name written into the file's field_descriptors block
    uint32_t bytes_per_atom; // per-atom byte size of this field
};

static const std::vector<FieldSpec> kFieldOrder = {
    { "indices", "id", sizeof(int64_t) },
    { "positions", "position", 3 * sizeof(float) },
    { "velocities", "velocity", 3 * sizeof(float) },
};

// Not a C string so only compare the 8 bytes even on a corrupt file
static constexpr uint8_t kMagicBytes[8] = { 'D', 'Y', 'N', 'A', 'M', 'I', 'S', 0 };

// cell (9 x f32) + pbc (3 x u8) + padding (1)
static constexpr uint32_t kPbcCellBlockSize = 40;

// Rounds a size up to the next 8-byte boundary padding amount.
inline uint32_t padding_to_8(uint64_t n)
{
    return 8 - static_cast<uint32_t>(n % 8);
}

// Fixed byte offsets into the front matter
namespace layout {
static constexpr uint32_t kMagicOffset = 0;
static constexpr uint32_t kMagicSize = 8;
static constexpr uint32_t kVersionOffset = kMagicOffset + kMagicSize; // 8
static constexpr uint32_t kVersionSize = sizeof(uint32_t); // 4
static constexpr uint32_t kRunIdOffset = kVersionOffset + kVersionSize; // 12
static constexpr uint32_t kRunIdSize = 16;
static constexpr uint32_t kAuthorOffset = kRunIdOffset + kRunIdSize; // 28
static constexpr uint32_t kAuthorSize = 64;
static constexpr uint32_t kTotalAtomsOffset = kAuthorOffset + kAuthorSize; // 92
static constexpr uint32_t kTotalAtomsSize = sizeof(uint64_t); // 8
static constexpr uint32_t kTimePerStepOffset = kTotalAtomsOffset + kTotalAtomsSize; // 100
static constexpr uint32_t kTimePerStepSize = sizeof(float); // 4
static constexpr uint32_t kStepsPerWriteOffset = kTimePerStepOffset + kTimePerStepSize; // 104
static constexpr uint32_t kStepsPerWriteSize = sizeof(uint32_t); // 4
static constexpr uint32_t kBytesPerAtomOffset = kStepsPerWriteOffset + kStepsPerWriteSize; // 108
static constexpr uint32_t kBytesPerAtomSize = sizeof(uint32_t); // 4
static constexpr uint32_t kFieldCountOffset = kBytesPerAtomOffset + kBytesPerAtomSize; // 112
static constexpr uint32_t kFieldCountSize = sizeof(uint32_t); // 4
static constexpr uint32_t kFieldDescriptorsOffset = kFieldCountOffset + kFieldCountSize; // 116
static constexpr uint32_t kFieldDescriptorSize = 32;
} // namespace layout

} // namespace dynamis_state
