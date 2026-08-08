#include "distnnp_state/trajectory_writer.hpp"
#include "distnnp_state/timing.hpp"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace distnnp_state {

TrajectoryWriterImpl::TrajectoryWriterImpl(
        uint32_t global_rank,
        py::bytes run_id,
        const std::string& author,
        const std::string& output_dir,
        const std::string& output_file,
        const MolecularInfo& molecular_info,
        torch::Tensor cell,
        torch::Tensor pbc,
        float time_per_step,
        uint32_t steps_per_write,
        std::vector<std::string> opts
)
    : rank(global_rank)
    , output_dir(output_dir)
    , output_file(output_file)
    , run_id(run_id)
    , author(author)
{
    // Ensure output directory exists.
    struct stat st;
    if (stat(output_dir.c_str(), &st) != 0) {
        if (mkdir(output_dir.c_str(), 0755) != 0) {
            throw std::runtime_error("failed to create output directory '" + output_dir + "': " + strerror(errno));
        }
    } else if (!S_ISDIR(st.st_mode)) {
        throw std::runtime_error("output path exists and is not a directory: '" + output_dir + "'");
    }

    const std::string file_path = output_dir + "/" + output_file;
    fd = (global_rank == 0)
        ? open(file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644)
        : open(file_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        throw std::runtime_error("failed to open '" + file_path + "': " + strerror(errno));
    }

    parse_opts(opts);
    write_front_matter(time_per_step, steps_per_write, molecular_info, rank);

    current_cell = cell.to(torch::kCPU).contiguous().toType(torch::kFloat32);
    current_pbc = pbc.to(torch::kCPU).contiguous().toType(torch::kUInt8);
}

TrajectoryWriterImpl::~TrajectoryWriterImpl() {
    // only flush and close if not already done
    if (!flushed) {
        auto res = close(fd);
        if (res != 0) throw std::runtime_error("close failed: " + std::string(strerror(errno)));
    }
}
void TrajectoryWriterImpl::flush() {
    if (!flushed) {
        // Where::Host: closing an fd is host work, and on a networked filesystem
        // (Delta's /u is NFS) it is not necessarily cheap.
        Timer t;
        t.start("traj_flush", TimerClass::Other, Where::Host);
        auto res = close(fd);
        t.stop();
        if (res != 0) throw std::runtime_error("close failed: " + std::string(strerror(errno)));
        flushed = true;
    }
}

void TrajectoryWriterImpl::parse_opts(const std::vector<std::string>& opts) {
    if (opts.empty()) {
        has_indices = true;
        has_positions = true;
        has_velocities = true;
        return; // default everything exists
    }
    has_indices = false;
    has_positions = false;
    has_velocities = false;
    for (const auto& o : opts) {
        if (o == "indices") has_indices = true;
        else if (o == "positions") has_positions = true;
        else if (o == "velocities") has_velocities = true;
        else throw std::runtime_error("TrajectoryWriter: unknown option '" + o + "'");
    }
    if (!has_positions) {
        throw std::runtime_error("TrajectoryWriter: 'positions' cannot be excluded from options");
    }
}

void TrajectoryWriterImpl::write_front_matter(float time_per_step, uint32_t steps_per_write, const MolecularInfo& state, uint32_t rank) {
    // figure out sizes of header and each rank size
    // resize file to the correct size

    // write the front_matter

    // determine which fields are active, in canonical order
    std::vector<const FieldSpec*> active_fields;
    for (const auto& f : kFieldOrder) {
        std::string opt(f.opt_name);
        if (opt == "indices" && !has_indices) continue;
        if (opt == "positions" && !has_positions) continue;
        if (opt == "velocities" && !has_velocities) continue;
        active_fields.push_back(&f);
    }
    // +1 for the pbc_cell block, which is always written as a field
    // descriptor even though it isn't a per-atom field (see submit())
    const uint32_t field_count = static_cast<uint32_t>(active_fields.size()) + 1;

    uint64_t front_matter_size = layout::kFieldDescriptorsOffset
                               + field_count * layout::kFieldDescriptorSize
                               + sizeof(uint64_t); // front_matter_size itself

    // should pad the front_matter_size so this block is on a 8 byte boundary
    uint32_t front_matter_padding = padding_to_8(front_matter_size);
    front_matter_size += front_matter_padding;

    // then the atom types start at a 8 byte boundary and then
    // I will insert padding at the end to make sure they stay on the 8 byte boundary

    front_matter_size += state.num_atoms; // 8 bit unsigned ints for each atom
    uint32_t atom_type_padding = padding_to_8(state.num_atoms);
    front_matter_size += atom_type_padding;

    // all ranks need to know front matter size, bytes per atom, and total atoms

    // This is only for version 1
    uint32_t bytes_per_atom = 0;
    for (const auto* f : active_fields)
        bytes_per_atom += f->bytes_per_atom;

    this->front_matter_size = front_matter_size;
    this->bytes_per_atom = bytes_per_atom;
    this->total_atoms = state.num_atoms;

    // all other ranks only need to know the above 3 values so can return here
    if (rank != 0) return;


    // create a buffer
    std::vector<uint8_t> buf(front_matter_size);

    uint32_t offset = 0;

    // magic
    memcpy(buf.data() + offset, kMagicBytes, sizeof(kMagicBytes));
    offset += sizeof(kMagicBytes);

    // version
    const uint32_t version = 1;
    memcpy(buf.data() + offset, &version, sizeof(version));
    offset += sizeof(version);

    // run_id
    std::string r = run_id;
    memcpy(buf.data() + offset, r.data(), r.size());
    memset(buf.data() + offset + r.size(), 0, layout::kRunIdSize - r.size());
    offset += layout::kRunIdSize;

    // author
    memcpy(buf.data() + offset, author.data(), author.size());
    memset(buf.data() + offset + author.size(), 0, layout::kAuthorSize - author.size());
    offset += layout::kAuthorSize;

    // total_atoms
    memcpy(buf.data() + offset, &state.num_atoms, sizeof(state.num_atoms));
    offset += sizeof(state.num_atoms);

    // time_per_step
    memcpy(buf.data() + offset, &time_per_step, sizeof(time_per_step));
    offset += sizeof(time_per_step);

    // steps_per_write
    memcpy(buf.data() + offset, &steps_per_write, sizeof(steps_per_write));
    offset += sizeof(steps_per_write);

    // bytes_per_atom, id, position, velocity
    memcpy(buf.data() + offset, &bytes_per_atom, sizeof(bytes_per_atom));
    offset += sizeof(bytes_per_atom);

    // field_count
    memcpy(buf.data() + offset, &field_count, sizeof(field_count));
    offset += sizeof(field_count);

    // fields
    // pbc_cell — the per-frame cell/pbc block (not a per-atom field, but
    // listed here for self-description; its layout is fixed, see submit())
    {
        const char* pbc_cell_str = "pbc_cell";
        uint32_t pbc_cell_size = strlen(pbc_cell_str);
        memcpy(buf.data() + offset, pbc_cell_str, pbc_cell_size);
        memset(buf.data() + offset + pbc_cell_size, 0, layout::kFieldDescriptorSize - pbc_cell_size);
        offset += layout::kFieldDescriptorSize;
    }
    for (const auto* f : active_fields) {
        std::string name(f->descriptor_name);
        memcpy(buf.data() + offset, name.data(), name.size());
        memset(buf.data() + offset + name.size(), 0, layout::kFieldDescriptorSize - name.size());
        offset += layout::kFieldDescriptorSize;
    }

    // front_matter_size
    memcpy(buf.data() + offset, &front_matter_size, sizeof(front_matter_size));
    offset += sizeof(front_matter_size);

    // padding
    memset(buf.data() + offset, 0, front_matter_padding);
    offset += front_matter_padding;

    // tensor of atomic numbers
    // TODO this could be large and its own separate write to the file to make things easier probably
    auto t = state.atom_types.contiguous().toType(torch::kUInt8);

    size_t tensor_bytes = t.numel() * t.element_size();
    memcpy(buf.data() + offset, t.data_ptr(), tensor_bytes);
    offset += tensor_bytes;

    // padding
    memset(buf.data() + offset, 0, atom_type_padding);
    offset += atom_type_padding;

    // write to the file
    ssize_t written = pwrite(fd, buf.data(), front_matter_size, 0);
    CHECK_WRITE(written, buf.size());
}

void TrajectoryWriterImpl::submit(uint32_t step, MolecularInfo state, c10::optional<torch::Tensor> cell, c10::optional<torch::Tensor> pbc) {

    // Where::Host, not Device: this launches no asynchronous CUDA work. The
    // .to(kCPU) copies below synchronize internally, so a host clock measures the
    // real cost — device-to-host transfer plus pwrite — rather than launch overhead.
    Timer t;
    t.start("traj_write", TimerClass::Other, Where::Host);

    // to make this easier, i think im going to write each tensor one after the other
    // so it will be
    // pbc_cell | 9 x float32 (cell) + 3 x uint8 (pbc) + 1 byte padding, written once per frame by rank 0
    // vertex_ids | uint8 x frame_size
    // atom_positions | 3 x float32 x frame_size
    // atom_velocities | 3 x float32 x frame_size

    // cell/pbc are optional; if not provided, keep writing whatever was last
    // known (seeded by the constructor, or a previous submit() call)
    if (cell.has_value())
        current_cell = cell->to(torch::kCPU).contiguous().toType(torch::kFloat32);
    if (pbc.has_value())
        current_pbc = pbc->to(torch::kCPU).contiguous().toType(torch::kUInt8);

    if (front_matter_size == 0)
        throw std::runtime_error("submit() called before front matter was written");

    uint32_t offset = front_matter_size + (step * (kPbcCellBlockSize + total_atoms * bytes_per_atom));

    if (rank == 0) {
        std::vector<uint8_t> pbc_cell_buf(kPbcCellBlockSize, 0);
        torch::Tensor cell_cpu = current_cell.contiguous();
        memcpy(pbc_cell_buf.data(), cell_cpu.data_ptr(), 9 * sizeof(float));
        torch::Tensor pbc_cpu = current_pbc.contiguous();
        memcpy(pbc_cell_buf.data() + 9 * sizeof(float), pbc_cpu.data_ptr(), 3 * sizeof(uint8_t));

        ssize_t written = pwrite(fd, pbc_cell_buf.data(), kPbcCellBlockSize, offset);
        CHECK_WRITE(written, kPbcCellBlockSize);
    }

    uint32_t atom_block_offset = offset + kPbcCellBlockSize;

    uint64_t atom_count = 0;
    for (uint32_t i = 0; i < rank; i++) {
        atom_count += rank_atom_counts.at(i);
    }

    auto write_tensor = [&](torch::Tensor t, uint32_t off) {
        torch::Tensor cpu_tensor = t.to(torch::kCPU).contiguous();
        size_t tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size();
        ssize_t written = pwrite(fd, cpu_tensor.data_ptr(), tensor_bytes, off);
        CHECK_WRITE(written, tensor_bytes);
    };

    // doing sizeof here to be more explicit about the calculations; atom data
    // starts after the pbc_cell block, at atom_block_offset
    uint32_t cursor = atom_block_offset;
    if (has_indices) {
        uint32_t field_offset = cursor + (atom_count * sizeof(int64_t));
        write_tensor(state.atom_vertex_numbers, field_offset);
        cursor += total_atoms * sizeof(int64_t);
    }
    if (has_positions) {
        uint32_t field_offset = cursor + (atom_count * 3 * sizeof(float));
        write_tensor(state.atom_positions, field_offset);
        cursor += total_atoms * 3 * sizeof(float);
    }
    if (has_velocities) {
        uint32_t field_offset = cursor + (atom_count * 3 * sizeof(float));
        write_tensor(state.atom_velocities, field_offset);
        cursor += total_atoms * 3 * sizeof(float);
    }

    t.stop();
}

} // namespace distnnp_state
