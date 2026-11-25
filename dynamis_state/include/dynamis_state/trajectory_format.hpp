#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

// constants shared by reader and writer
namespace dynamis_state {

// Max bytes per individual pread/pwrite syscall. Linux caps a single call at
// 0x7ffff000 (~2 GiB - 4 KiB); chunking at 1 GiB stays well below that limit
// while also handling EINTR and partial transfers cleanly.
static constexpr uint64_t kMaxIoChunkBytes = static_cast<uint64_t>(1) << 30; // 1 GiB

#define PREAD_ALL(fd, buf, count, offset)                                                          \
    ::dynamis_state::detail::pread_all((fd), (buf), (count), (offset), __FILE__, __LINE__)

#define PWRITE_ALL(fd, buf, count, offset)                                                         \
    ::dynamis_state::detail::pwrite_all((fd), (buf), (count), (offset), __FILE__, __LINE__)

namespace detail {
inline void pread_all(int fd, void *buf, uint64_t count, uint64_t offset, const char *file,
                      int line)
{
    auto *dst = static_cast<uint8_t *>(buf);
    uint64_t done = 0;
    while (done < count) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(count - done, kMaxIoChunkBytes));
        ssize_t n = ::pread(fd, dst + done, chunk, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::string("pread failed at ") + file + ":" +
                                     std::to_string(line) + ": " + strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error(std::string("short read at ") + file + ":" +
                                     std::to_string(line) + ": expected " + std::to_string(count) +
                                     " bytes, got " + std::to_string(done));
        }
        done += static_cast<uint64_t>(n);
    }
}

inline void pwrite_all(int fd, const void *buf, uint64_t count, uint64_t offset, const char *file,
                       int line)
{
    const auto *src = static_cast<const uint8_t *>(buf);
    uint64_t done = 0;
    while (done < count) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(count - done, kMaxIoChunkBytes));
        ssize_t n = ::pwrite(fd, src + done, chunk, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(std::string("pwrite failed at ") + file + ":" +
                                     std::to_string(line) + ": " + strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error(std::string("short write at ") + file + ":" +
                                     std::to_string(line) + ": expected " + std::to_string(count) +
                                     " bytes, got " + std::to_string(done));
        }
        done += static_cast<uint64_t>(n);
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
