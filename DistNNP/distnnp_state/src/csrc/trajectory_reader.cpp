#include "distnnp_state/trajectory_reader.hpp"
#include "distnnp_state/trajectory_format.hpp"
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace distnnp_state {

namespace {

// Resolves a possibly-negative frame_index against a known frame count,
// shared by both the xyz and distnnp read paths.
int64_t resolve_frame_index(int64_t frame_index, uint64_t frame_count, const std::string& path) {
    int64_t count = static_cast<int64_t>(frame_count);
    int64_t resolved = frame_index < 0 ? count + frame_index : frame_index;
    if (resolved < 0 || resolved >= count) {
        throw std::runtime_error(
            "frame_index " + std::to_string(frame_index) + " out of range for '" + path +
            "' (found " + std::to_string(count) + (count == 1 ? " frame)" : " frames)"));
    }
    return resolved;
}

// Validates the caller-supplied opts list, throwing on anything unusable, and
// reports whether velocities were requested.
bool validate_xyz_opts(const std::vector<std::string>& opts) {
    if (opts.empty()) return true; // default: everything present
    bool has_velocities = false;
    bool has_positions = false;
    for (const auto& o : opts) {
        if (o == "velocities") has_velocities = true;
        else if (o == "positions") has_positions = true;
        else if (o == "indices") continue; // xyz never writes an id column
        else throw std::runtime_error("translate_distnnp_to_xyz: unknown option '" + o + "'");
    }
    if (!has_positions) {
        throw std::runtime_error("translate_distnnp_to_xyz: 'positions' cannot be excluded from options");
    }
    return has_velocities;
}

void write_xyz_frame(std::ostream& out, const MolecularInfo& state, torch::Tensor cell,
                      torch::Tensor pbc, bool has_velocities) {
    const auto& rev = reverse_periodic_table();

    auto types = state.atom_types.accessor<int64_t, 1>();
    auto positions = state.atom_positions.accessor<float, 2>();

    torch::Tensor cell_cpu = cell.to(torch::kCPU).contiguous().toType(torch::kFloat32);
    torch::Tensor pbc_cpu = pbc.to(torch::kCPU).contiguous().toType(torch::kUInt8);
    auto cell_acc = cell_cpu.accessor<float, 2>();
    auto pbc_acc = pbc_cpu.accessor<uint8_t, 1>();

    auto write_floats = [&](float x, float y, float z) {
        out << std::to_string(x) << '\t' << std::to_string(y) << '\t' << std::to_string(z);
    };

    out << std::to_string(state.num_atoms) << '\n';

    out << "Lattice=\"";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out << cell_acc[i][j];
            if (!(i == 2 && j == 2)) out << ' ';
        }
    }
    out << "\" ";
    // vel, not momenta: MolecularInfo stores velocities (A/fs), which is also
    // what the integrators consume, so the column is labelled for what it is
    // rather than as p = m*v.
    out << "Properties=species:S:1:pos:R:3" << (has_velocities ? ":vel:R:3" : "") << " ";
    out << "pbc=\"" << (pbc_acc[0] ? "T" : "F") << ' ' << (pbc_acc[1] ? "T" : "F") << ' ' << (pbc_acc[2] ? "T" : "F") << "\"\n";

    if (has_velocities) {
        auto velocities = state.atom_velocities.accessor<float, 2>();
        for (int64_t i = 0; i < static_cast<int64_t>(state.num_atoms); i++) {
            out << rev.at(static_cast<uint8_t>(types[i])) << '\t';
            write_floats(positions[i][0], positions[i][1], positions[i][2]);
            out << '\t';
            write_floats(velocities[i][0], velocities[i][1], velocities[i][2]);
            out << '\n';
        }
    } else {
        for (int64_t i = 0; i < static_cast<int64_t>(state.num_atoms); i++) {
            out << rev.at(static_cast<uint8_t>(types[i])) << '\t';
            write_floats(positions[i][0], positions[i][1], positions[i][2]);
            out << '\n';
        }
    }
}

// ─── .distnnp reading ──────────────────────────────────────────────────────

struct DistnnpHeader {
    uint64_t num_atoms{ 0 };
    uint32_t bytes_per_atom{ 0 };
    std::vector<std::string> field_names;
    uint64_t front_matter_size{ 0 };
    torch::Tensor atom_types; // [num_atoms] int64, parsed once, shared across frames
    uint64_t frame_size{ 0 };
    uint64_t frame_count{ 0 };
};

DistnnpHeader read_distnnp_header(int fd, const std::string& path) {
    ssize_t n;

    uint8_t magic[8];
    n = pread(fd, magic, sizeof(magic), layout::kMagicOffset);
    CHECK_READ(n, sizeof(magic));
    if (memcmp(magic, kMagicBytes, sizeof(magic)) != 0) {
        throw std::runtime_error("'" + path + "' is not a valid distnnp file (bad magic)");
    }

    uint32_t version;
    n = pread(fd, &version, sizeof(version), layout::kVersionOffset);
    CHECK_READ(n, sizeof(version));
    if (version != 1) {
        throw std::runtime_error("'" + path + "': unsupported distnnp version " + std::to_string(version) +
                                  " (only version 1 is supported)");
    }

    DistnnpHeader h;

    n = pread(fd, &h.num_atoms, sizeof(h.num_atoms), layout::kTotalAtomsOffset);
    CHECK_READ(n, sizeof(h.num_atoms));

    n = pread(fd, &h.bytes_per_atom, sizeof(h.bytes_per_atom), layout::kBytesPerAtomOffset);
    CHECK_READ(n, sizeof(h.bytes_per_atom));

    uint32_t field_count;
    n = pread(fd, &field_count, sizeof(field_count), layout::kFieldCountOffset);
    CHECK_READ(n, sizeof(field_count));

    // Field count can never be more than 4 (pbc_cell + id + position + velocity)
    if (field_count > 4) {
        throw std::runtime_error("implausible field_count (" + std::to_string(field_count) + ") in '" + path + "'");
    }

    h.field_names.resize(field_count);
    for (uint32_t i = 0; i < field_count; i++) {
        char name_buf[layout::kFieldDescriptorSize] = { 0 };
        n = pread(fd, name_buf, sizeof(name_buf), layout::kFieldDescriptorsOffset + i * layout::kFieldDescriptorSize);
        CHECK_READ(n, sizeof(name_buf));
        h.field_names[i] = std::string(name_buf, strnlen(name_buf, sizeof(name_buf)));
    }

    n = pread(fd, &h.front_matter_size, sizeof(h.front_matter_size),
              layout::kFieldDescriptorsOffset + static_cast<uint64_t>(field_count) * layout::kFieldDescriptorSize);
    CHECK_READ(n, sizeof(h.front_matter_size));

    // atom types sit right before the front_matter_size boundary, padded to
    // an 8-byte multiple
    uint32_t atom_type_padding = padding_to_8(h.num_atoms);
    uint64_t atom_types_offset = h.front_matter_size - h.num_atoms - atom_type_padding;

    std::vector<uint8_t> types(h.num_atoms);
    n = pread(fd, types.data(), types.size(), atom_types_offset);
    CHECK_READ(n, static_cast<ssize_t>(types.size()));
    h.atom_types = torch::from_blob(types.data(), { static_cast<int64_t>(h.num_atoms) }, torch::kUInt8).clone().to(torch::kInt64);

    h.frame_size = kPbcCellBlockSize + static_cast<uint64_t>(h.bytes_per_atom) * h.num_atoms;
    if (h.frame_size <= kPbcCellBlockSize) {
        throw std::runtime_error("'" + path + "': malformed header (empty frame size)");
    }

    uint64_t file_size = static_cast<uint64_t>(lseek(fd, 0, SEEK_END));
    if (file_size < h.front_matter_size || (file_size - h.front_matter_size) % h.frame_size != 0) {
        throw std::runtime_error("'" + path + "': file size is inconsistent with its own header (truncated or corrupt)");
    }
    h.frame_count = (file_size - h.front_matter_size) / h.frame_size;
    if (h.frame_count == 0) {
        throw std::runtime_error("'" + path + "' contains no frames");
    }

    return h;
}

std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> read_distnnp_frame(int fd, const DistnnpHeader& h, const std::string& path, uint64_t frame_index) {
    ssize_t n;

    MolecularInfo atoms;
    atoms.num_atoms = h.num_atoms;
    atoms.atom_types = h.atom_types;

    uint64_t loc = h.front_matter_size + frame_index * h.frame_size;

    float cell_buf[9];
    n = pread(fd, cell_buf, sizeof(cell_buf), loc);
    CHECK_READ(n, sizeof(cell_buf));
    torch::Tensor cell = torch::from_blob(cell_buf, { 3, 3 }, torch::kFloat32).clone();

    uint8_t pbc_buf[3];
    n = pread(fd, pbc_buf, sizeof(pbc_buf), loc + sizeof(cell_buf));
    CHECK_READ(n, sizeof(pbc_buf));
    torch::Tensor pbc = torch::from_blob(pbc_buf, { 3 }, torch::kUInt8).clone();

    uint64_t cursor = loc + kPbcCellBlockSize;

    std::unordered_map<std::string, uint64_t> field_offsets;
    for (const auto& name : h.field_names) {
        if (name == "pbc_cell") continue; // already read above; not part of the per-atom block
        field_offsets[name] = cursor;
        if (name == "id") cursor += h.num_atoms * sizeof(int64_t);
        else if (name == "position") cursor += h.num_atoms * 3 * sizeof(float);
        else if (name == "velocity") cursor += h.num_atoms * 3 * sizeof(float);
        else throw std::runtime_error("unknown field descriptor '" + name + "' in '" + path + "'");
    }

    // ids: default to sequential order if not present
    if (field_offsets.count("id")) {
        uint32_t block_size = h.num_atoms * sizeof(int64_t);
        std::vector<int64_t> ids(h.num_atoms);
        n = pread(fd, ids.data(), block_size, field_offsets.at("id"));
        CHECK_READ(n, block_size);
        atoms.atom_vertex_numbers = torch::from_blob(ids.data(), { static_cast<int64_t>(h.num_atoms) }, torch::kInt64).clone();
    } else {
        atoms.atom_vertex_numbers = torch::arange(0, h.num_atoms, 1).to(torch::kInt64);
    }

    // positions: mandatory
    if (!field_offsets.count("position")) {
        throw std::runtime_error("'position' field missing from '" + path + "', cannot load");
    }
    {
        uint32_t block_size = h.num_atoms * 3 * sizeof(float);
        std::vector<float> attribs(h.num_atoms * 3);
        n = pread(fd, attribs.data(), block_size, field_offsets.at("position"));
        CHECK_READ(n, block_size);
        atoms.atom_positions = torch::from_blob(attribs.data(), { static_cast<int64_t>(h.num_atoms), 3 }).clone();
    }

    // velocities: default to zero if not present
    if (field_offsets.count("velocity")) {
        uint32_t block_size = h.num_atoms * 3 * sizeof(float);
        std::vector<float> attribs(h.num_atoms * 3);
        n = pread(fd, attribs.data(), block_size, field_offsets.at("velocity"));
        CHECK_READ(n, block_size);
        atoms.atom_velocities = torch::from_blob(attribs.data(), { static_cast<int64_t>(h.num_atoms), 3 }).clone();
    } else {
        atoms.atom_velocities = torch::zeros({ static_cast<int64_t>(h.num_atoms), 3 });
    }

    return { atoms, cell, pbc };
}

std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> load_distnnp_file(const std::string& path, int64_t frame_index) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("failed to open '" + path + "': " + strerror(errno));

    try {
        DistnnpHeader h = read_distnnp_header(fd, path);
        int64_t resolved = resolve_frame_index(frame_index, h.frame_count, path);
        auto result = read_distnnp_frame(fd, h, path, static_cast<uint64_t>(resolved));
        close(fd);
        return result;
    } catch (...) {
        close(fd);
        throw;
    }
}

// ─── .xyz reading ──────────────────────────────────────────────────────────

std::vector<std::string> tokenize_xyz_comment(const std::string& line) {
    // Tokenize on whitespace, but treat a "..." span (which may contain
    // embedded spaces, e.g. Lattice="15.0 0.0 0.0 ...") as a single token.
    std::vector<std::string> tokens;
    bool in_quotes = false;
    std::string current_token;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c)) && !in_quotes) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    if (!current_token.empty()) tokens.push_back(current_token);
    return tokens;
}

struct XyzFrameLocator {
    std::vector<std::streampos> offsets; // position of the natoms line, one per frame
    uint64_t num_atoms{ 0 };
};

// First pass: walks every frame block, recording where each starts and
// validating the constant-atom-count invariant, without parsing atom lines.
XyzFrameLocator scan_xyz_frames(std::ifstream& xyz_file, const std::string& path) {
    XyzFrameLocator locator;
    std::string line;

    while (true) {
        std::streampos block_start = xyz_file.tellg();
        if (!std::getline(xyz_file, line)) break; // clean EOF between blocks

        int natoms;
        try {
            natoms = std::stoi(line);
        } catch (const std::exception&) {
            throw std::runtime_error("'" + path + "': malformed atom-count line '" + line + "'");
        }
        if (natoms < 0) {
            throw std::runtime_error("'" + path + "': negative atom count '" + line + "'");
        }

        if (locator.offsets.empty()) {
            locator.num_atoms = static_cast<uint64_t>(natoms);
        } else if (static_cast<uint64_t>(natoms) != locator.num_atoms) {
            throw std::runtime_error(
                "'" + path + "': frame " + std::to_string(locator.offsets.size()) + " has " +
                std::to_string(natoms) + " atoms, expected " + std::to_string(locator.num_atoms) +
                " (atom count must be constant across frames)");
        }
        locator.offsets.push_back(block_start);

        if (!std::getline(xyz_file, line)) {
            throw std::runtime_error("'" + path + "': unexpected end of file (missing comment line)");
        }
        for (int i = 0; i < natoms; i++) {
            if (!std::getline(xyz_file, line)) {
                throw std::runtime_error("'" + path + "': unexpected end of file while scanning frame " +
                                          std::to_string(locator.offsets.size() - 1));
            }
        }
    }

    if (locator.offsets.empty()) {
        throw std::runtime_error("'" + path + "' contains no frames");
    }
    return locator;
}

// Second pass: xyz_file must already be seeked to the start of the target
// block (its natoms line). Parses that one block fully.
std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> parse_xyz_block(std::ifstream& xyz_file, const std::string& path, uint64_t block_index) {
    std::string line;

    std::getline(xyz_file, line);
    int num_atoms = std::stoi(line); // already validated during the scan pass

    std::getline(xyz_file, line);
    std::vector<std::string> tokens = tokenize_xyz_comment(line);

    int velocity_index = -1;
    bool velocity_is_momenta = false; // momenta columns hold p = m*v, not v
    uint32_t current_index = 0;
    std::vector<float> cell_values;
    std::vector<uint8_t> pbc_values;

    for (std::string token : tokens) {
        if (token.find("Properties") == 0) {
            size_t property_start = token.find('=');
            if (property_start == std::string::npos) {
                throw std::runtime_error("'" + path + "': malformed Properties= in frame " + std::to_string(block_index));
            }
            property_start += 1; // past the =
            token = token.substr(property_start, token.size() - property_start);
            std::string name, ty, count;
            std::stringstream opt(token);
            while (std::getline(opt, name, ':')) {
                std::getline(opt, ty, ':');
                std::getline(opt, count, ':');

                if (name == "vel" || name == "velo" || name == "velocities") {
                    velocity_index = current_index;
                    velocity_is_momenta = false;
                } else if (name == "momenta") {
                    velocity_index = current_index;
                    velocity_is_momenta = true;
                }

                current_index += std::stoi(count);
            }
        } else if (token.find("Lattice=") == 0) {
            std::stringstream ls(token.substr(strlen("Lattice=")));
            float v;
            while (ls >> v) cell_values.push_back(v);
            if (cell_values.size() != 9) cell_values.clear(); // malformed; fall back to default below
        } else if (token.find("pbc=") == 0) {
            std::stringstream ps(token.substr(strlen("pbc=")));
            std::string flag;
            while (ps >> flag) {
                pbc_values.push_back((flag == "T" || flag == "True" || flag == "1") ? 1 : 0);
            }
            if (pbc_values.size() != 3) pbc_values.clear(); // malformed; fall back to default below
        }
    }

    // row-major [3, 3] cell, [3] pbc flags; default to no box/no periodicity
    // when the comment line doesn't carry Lattice=/pbc= (e.g. plain xyz)
    torch::Tensor cell = (cell_values.size() == 9)
        ? torch::from_blob(cell_values.data(), { 3, 3 }, torch::kFloat32).clone()
        : torch::zeros({ 3, 3 }, torch::kFloat32);
    torch::Tensor pbc = (pbc_values.size() == 3)
        ? torch::from_blob(pbc_values.data(), { 3 }, torch::kUInt8).clone()
        : torch::zeros({ 3 }, torch::kUInt8);

    std::vector<uint8_t> types;
    std::vector<float> positions;
    std::vector<float> velocities;
    for (int i = 0; i < num_atoms; i++) {
        std::getline(xyz_file, line); // already validated to exist during the scan pass
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::stringstream ss(line);
        std::string element, xstr, ystr, zstr;
        if (!(ss >> element >> xstr >> ystr >> zstr)) {
            throw std::runtime_error("'" + path + "': malformed atom line in frame " + std::to_string(block_index) +
                                      ": '" + line + "'");
        }

        auto it = periodic_table.find(element);
        if (it == periodic_table.end()) {
            throw std::runtime_error("'" + path + "': unknown element symbol '" + element + "' in frame " +
                                      std::to_string(block_index));
        }
        types.emplace_back(it->second.number);
        try {
            positions.emplace_back(std::stof(xstr));
            positions.emplace_back(std::stof(ystr));
            positions.emplace_back(std::stof(zstr));
        } catch (const std::exception&) {
            throw std::runtime_error("'" + path + "': malformed position in frame " + std::to_string(block_index) +
                                      ": '" + line + "'");
        }

        if (velocity_index != -1) {
            int idx = velocity_index - 4; // already read in 4 things (species + xyz)
            std::string skip;
            while (idx > 0) {
                ss >> skip;
                idx--;
            }
            if (!(ss >> xstr >> ystr >> zstr)) {
                throw std::runtime_error("'" + path + "': malformed velocity in frame " + std::to_string(block_index) +
                                          ": '" + line + "'");
            }
            // MolecularInfo stores velocities, so a momenta column has to be
            // divided back through by this atom's mass (v = p / m).
            float scale = velocity_is_momenta ? 1.0f / it->second.mass : 1.0f;
            velocities.emplace_back(std::stof(xstr) * scale);
            velocities.emplace_back(std::stof(ystr) * scale);
            velocities.emplace_back(std::stof(zstr) * scale);
        }
    }

    MolecularInfo atoms;
    atoms.num_atoms = num_atoms;
    atoms.atom_vertex_numbers = torch::arange(0, num_atoms, 1).to(torch::kInt64);
    atoms.atom_types = torch::from_blob(types.data(), { num_atoms }, torch::kUInt8).clone().toType(torch::kInt64);
    atoms.atom_positions = torch::from_blob(positions.data(), { num_atoms, 3 }).clone();
    atoms.atom_velocities = (velocity_index == -1)
        ? torch::zeros({ num_atoms, 3 })
        : torch::from_blob(velocities.data(), { num_atoms, 3 }).clone();

    return { atoms, cell, pbc };
}

std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> load_xyz_file(const std::string& path, int64_t frame_index) {
    std::ifstream xyz_file(path);
    if (!xyz_file.is_open()) {
        throw std::runtime_error("failed to open '" + path + "'");
    }

    XyzFrameLocator locator = scan_xyz_frames(xyz_file, path);
    int64_t resolved = resolve_frame_index(frame_index, locator.offsets.size(), path);

    xyz_file.clear(); // drop eof/fail bits set by the scan pass before seeking back
    xyz_file.seekg(locator.offsets[static_cast<size_t>(resolved)]);

    return parse_xyz_block(xyz_file, path, static_cast<uint64_t>(resolved));
}

} // namespace

std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> read_molecular_info(const std::string& path, int64_t frame_index) {
    auto len = path.size();
    if (len >= 4 && path.compare(len - 4, 4, ".xyz") == 0) {
        return load_xyz_file(path, frame_index);
    }
    if (len >= 8 && path.compare(len - 8, 8, ".distnnp") == 0) {
        return load_distnnp_file(path, frame_index);
    }
    throw std::runtime_error("'" + path + "': unsupported file type (expected .xyz or .distnnp)");
}

void translate_distnnp_to_xyz(const std::string& distnnp_path, const std::string& xyz_path,
                               std::vector<std::string> opts) {
    bool has_velocities = validate_xyz_opts(opts);

    int fd = open(distnnp_path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("failed to open '" + distnnp_path + "': " + strerror(errno));

    std::ofstream out(xyz_path);
    if (!out.is_open()) {
        close(fd);
        throw std::runtime_error("failed to open '" + xyz_path + "' for writing");
    }

    try {
        DistnnpHeader h = read_distnnp_header(fd, distnnp_path);
        for (uint64_t i = 0; i < h.frame_count; i++) {
            auto [state, cell, pbc] = read_distnnp_frame(fd, h, distnnp_path, i);
            write_xyz_frame(out, state, cell, pbc, has_velocities);
        }
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
}

} // namespace distnnp_state
