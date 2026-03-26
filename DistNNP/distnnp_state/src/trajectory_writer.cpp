#include "distnnp_state/trajectory_writer.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>

namespace distnnp_state {

TrajectoryWriter::TrajectoryWriter(uint32_t global_rank, py::bytes run_id, const std::string& author, std::string output_dir)
    : rank(global_rank)
    // TODO validate both of these
    , run_id(run_id)
    , author(author)
{
    // currently just overwriting, TODO increment number over time
    fd = open((output_dir + "/trajectory.xyzbin").c_str(), O_RDWR | O_CREAT, 0644);

    if (fd < 0) {
        std::cerr << "Failed to open " << output_dir << "/trajectory.xyzbin" << std::endl;
        exit(1);
    }
}

MolecularInfo TrajectoryWriter::initialize(std::string xyz_file_name, float time_per_step, float steps_per_write) {
    auto state = load_file(xyz_file_name);

    if (rank != 0) return state;

    write_front_matter(time_per_step, steps_per_write, state);

    return state;
}

void TrajectoryWriter::write_front_matter(float time_per_step, uint32_t steps_per_write, MolecularInfo state) {
    // figure out sizes of header and each rank size
    // resize file to the correct size

    // write the front_matter

    // 220 bytes, TODO need to pad for alignment out to 256 bytes
    uint64_t front_matter_size = 8  // magic
                               + 4  // version
                               + 16 // run_id 
                               + 64 // author
                               + 8  // total_atoms
                               + 4  // time_per_step
                               + 4  // steps_per_write
                               + 4  // bytes_per_atom
                               + 4  // field_count
                               + 96 // 3x32 byte fields
                               + 8  // front_matter_size
                               + 36; // padding to align with 256 bytes for this part of the front_matter

    // 32 bit ints for easier alignment
    // TODO figure out padding for this
    front_matter_size += state.num_atoms * 4; // 32 bit unsigned ints for each atom

    // create a buffer
    std::vector<uint8_t> buf(front_matter_size);

    uint32_t offset = 0;

    // magic
    const char* magic = "DISTNNP\x00";
    std::cerr << "copying " << magic << std::endl;
    memcpy(buf.data() + offset, magic, 8);
    offset += 8;

    // version
    const uint32_t version = 1;
    std::cerr << "copying " << version << std::endl;
    memcpy(buf.data() + offset, &version, sizeof(version));
    offset += sizeof(version);

    // run_id
    std::string r = run_id;
    std::cerr << "copying " << r << std::endl;
    memcpy(buf.data() + offset, r.data(), 16);
    offset += 16;

    // author
    std::cerr << "copying " << author << std::endl;
    memcpy(buf.data() + offset, author.data(), author.size());
    memset(buf.data() + offset + author.size(), 0, 64 - author.size());
    offset += 64;

    // total_atoms
    std::cerr << "copying " << state.num_atoms << std::endl;
    memcpy(buf.data() + offset, &state.num_atoms, sizeof(state.num_atoms));
    offset += sizeof(state.num_atoms);

    // time_per_step
    std::cerr << "copying " << time_per_step << std::endl;
    memcpy(buf.data() + offset, &time_per_step, sizeof(time_per_step));
    offset += sizeof(time_per_step);

    // steps_per_write 
    std::cerr << "copying " << steps_per_write << std::endl;
    memcpy(buf.data() + offset, &steps_per_write, sizeof(steps_per_write));
    offset += sizeof(time_per_step);

    // bytes_per_atom, id, position, velocity
    const uint32_t bytes_per_atom = 8 + (3 * 4) + (3 * 4);
    std::cerr << "copying " << bytes_per_atom << std::endl;
    memcpy(buf.data() + offset, &bytes_per_atom, sizeof(bytes_per_atom));
    offset += sizeof(bytes_per_atom);

    // field_count
    const uint32_t field_count = 3;
    std::cerr << "copying " << field_count << std::endl;
    memcpy(buf.data() + offset, &field_count, sizeof(field_count));
    offset += sizeof(field_count);

    // fields

    // id
    const char* id_str = "id\0";
    uint32_t id_size = strlen(id_str);
    memcpy(buf.data() + offset, id_str, id_size);
    memset(buf.data() + offset + id_size, 0, 32 - id_size);
    offset += 32;

    // position 
    const char* pos_str = "position\0";
    uint32_t pos_size = strlen(pos_str);
    memcpy(buf.data() + offset, pos_str, pos_size);
    memset(buf.data() + offset + pos_size, 0, 32 - pos_size);
    offset += 32;

    // velocity 
    const char* vel_str = "velocity\0";
    uint32_t vel_size = strlen(vel_str);
    memcpy(buf.data() + offset, vel_str, vel_size);
    memset(buf.data() + offset + vel_size, 0, 32 - vel_size);
    offset += 32;

    // front_matter_size
    memcpy(buf.data() + offset, &front_matter_size, sizeof(front_matter_size));
    offset += front_matter_size;

    // padding
    uint8_t padding = 36;
    memset(buf.data() + offset, 0, padding);
    offset += padding;

    // tensor of atomic numbers
    // TODO this could be large and its own separate write to the file to make things easier probably
    auto t = state.atom_types.contiguous();

    size_t tensor_bytes = t.numel() * t.element_size();
    memcpy(buf.data() + offset, t.data_ptr(), tensor_bytes);
    offset += tensor_bytes;

    // write to the file
    ssize_t written = write(fd, buf.data(), buf.size());
    if (written < 0) {
        throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (static_cast<size_t>(written) != buf.size()) {
        throw std::runtime_error("partial write");
    }
}
void TrajectoryWriter::write_frame(int fd, uint32_t frame_idx, torch::Tensor indices, torch::Tensor positions, torch::Tensor velocities) {

    // to make this easier, i think im going to write each tensor one after the other
    // so it will be
    // vertex_ids | uint8 x frame_size
    // atom_positions | 3 x float32 x frame_size
    // atom_velocities | 3 x float32 x frame_size

    if (writes = 0) {
        // first write, don't have these values saved
        // need to read in total_atoms, bytes_per_atom, front_matter_size

        // TODO need to guarentee that rank 0 is done writing

        auto n = pread(fd, &total_atoms, sizeof(total_atoms), 92); // TODO check offset
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(total_atoms)) throw std::runtime_error("unexpected end of file");

        n = pread(fd, &bytes_per_atom, sizeof(bytes_per_atom), 100); // TODO check offset
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(bytes_per_atom)) throw std::runtime_error("unexpected end of file");

        n = pread(fd, &front_matter_size, sizeof(front_matter_size), 212); // TODO check offset
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(front_matter_size)) throw std::runtime_error("unexpected end of file");
    }

    uint32_t init_offset = front_matter_size + (writes * total_atoms * bytes_per_atom);
    uint32_t in_frame_offset = 0;
    for (int i = 0; i < rank; i++) {
        in_frame_offset += rank_atom_counts.at(i) * bytes_per_atom; // this will error if the vector is setup wrong
    }

    // TODO check this math
    uint32_t offset = init_offset + in_frame_offset;


    auto write_tensor = [&](torch::Tensor t, uint32_t off) -> uint32_t {
        size_t tensor_bytes = t.numel() * t.element_size();
        ssize_t written = pwrite(fd, t.data_ptr(), tensor_bytes, off);

        // TODO might have to make this a loop if cant write the entire tensor at once
        if (written < 0) throw std::runtime_error("pwrite failed: " + std::string(strerror(errno)));
        if (static_cast<size_t>(written) != tensor_bytes) throw std::runtime_error("partial write");

        off += tensor_bytes;
        return off;
    };

    offset = write_tensor(indices.contiguous(), offset); // TODO make sure on CPU
    offset = write_tensor(positions.contiguous(), offset); // TODO make sure on CPU
    offset = write_tensor(velocities.contiguous(), offset); // TODO make sure on CPU

    // to calculate frame offsets better
    writes += 1;
}

MolecularInfo TrajectoryWriter::load_file(std::string xyz_file_name) {
    MolecularInfo atoms;

    // open the file
    std::ifstream xyz_file(xyz_file_name);
    if (!xyz_file.is_open()) {
        std::cerr << xyz_file_name << " could not be opened" << std::endl;
        return atoms;
    }
    std::string line;

    // read in the atom numbers
    std::getline(xyz_file, line);
    uint32_t num_atoms = std::stoi(line);

    int velocity_index = -1;

    std::getline(xyz_file, line);
    std::stringstream options(line);
    uint32_t current_index = 0;
    while (options >> line) {
        // Extra options if there is some
        // Although we only care about Properties, and in that species, pos, and vel
        if (line.find("Properties") != 0) continue;

        int property_start = line.find('=');
        if (property_start == std::string::npos) {
            std::cerr << "Properties is set up wrong" << std::endl;
            return atoms;
        }
        property_start += 1; // to go past the =
        line = line.substr(property_start, line.size() - property_start);
        std::string name, ty, count;
        std::stringstream opt(line);
        while (std::getline(opt, name, ':')) {
            std::getline(opt, ty, ':');
            std::getline(opt, count, ':');
            std::cerr << name << " : " << ty << " : " << count << std::endl;

            // only care about velocity
            if (name == "vel") {
                velocity_index = current_index;
            }

            current_index += std::stoi(count);
        }
    }

    // read in all of the positions
    std::vector<float> positions;
    std::vector<float> velocities;
    while (std::getline(xyz_file, line)) {
        // add this atom to info
        std::stringstream ss(line);
        std::string element, xstr, ystr, zstr;
        if(!(ss >> element >> xstr >> ystr >> zstr)) {
            std::cerr << "Line failed to parse" << std::endl;
            return atoms;
        }
        positions.emplace_back(std::stof(xstr));
        positions.emplace_back(std::stof(ystr));
        positions.emplace_back(std::stof(zstr));

        if (velocity_index != -1) {
            int idx = velocity_index - 4; // already read in 4 things
            while (idx > 0) {
                ss >> element;
                idx--;
            }
            // read in velocity
            if (!(ss >> xstr >> ystr >> zstr)) {
                std::cerr << "Velocities failed to parse" << std::endl;
                return atoms;
            }
            velocities.emplace_back(std::stof(xstr));
            velocities.emplace_back(std::stof(ystr));
            velocities.emplace_back(std::stof(zstr));
        }
    }

    // tensor setup
    atoms.num_atoms = num_atoms;
    atoms.atom_indices = torch::arange(0, num_atoms, 1);
    atoms.atom_positions = torch::from_blob(positions.data(), {num_atoms, 3}).clone();

    if (velocity_index == -1)
        atoms.atom_velocities = torch::zeros({num_atoms, 3});
    else
        atoms.atom_velocities = torch::from_blob(velocities.data(), {num_atoms, 3}).clone();

    return atoms;
}
} // namespace distnnp_state
