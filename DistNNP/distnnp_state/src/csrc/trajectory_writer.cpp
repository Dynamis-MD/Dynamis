#include "distnnp_state/trajectory_writer.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

namespace distnnp_state {

TrajectoryWriter::TrajectoryWriter(uint32_t global_rank, py::bytes run_id, const std::string& author, std::string output_dir)
    : rank(global_rank)
    , output_dir(output_dir)
    // TODO validate both of these
    , run_id(run_id)
    , author(author)
{
    // currently just overwriting, TODO increment number over time
    if (global_rank == 0)
        fd = open((output_dir + "/trajectory.distnnp").c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    else
        fd = open((output_dir + "/trajectory.distnnp").c_str(), O_RDWR | O_CREAT, 0644);

    if (fd < 0) {
        std::cerr << "Failed to open " << output_dir << "/trajectory.distnnp" << std::endl;
        exit(1);
    }
}
TrajectoryWriter::~TrajectoryWriter() {
    auto res = close(fd);
    if (res != 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
}

MolecularInfo TrajectoryWriter::initialize(std::string input_file_name, float time_per_step, uint32_t steps_per_write) {

    MolecularInfo state;
    auto len = input_file_name.size();
    if (len >= 4 && input_file_name.compare(len - 4, 4, ".xyz") == 0) {
        state = load_xyz_file(input_file_name);
    } else if (len >= 8 && input_file_name.compare(len - 8, 8, ".distnnp") == 0) {
        state = load_distnnp_file(input_file_name);
    } else {
        std::cerr << input_file_name << " file type is not supported" << std::endl;
        return state;
    }

    if (rank == 0)
        write_front_matter(time_per_step, steps_per_write, state);

    return state;
}

void TrajectoryWriter::write_front_matter(float time_per_step, uint32_t steps_per_write, MolecularInfo state) {
    // figure out sizes of header and each rank size
    // resize file to the correct size

    // write the front_matter

    // 220 bytes
    uint64_t front_matter_size = sizeof(uint8_t) * 8  // magic
                               + sizeof(uint32_t)  // version
                               + sizeof(uint8_t) * 16 // run_id
                               + sizeof(uint8_t) * 64 // author
                               + sizeof(uint64_t)  // total_atoms
                               + sizeof(float)  // time_per_step
                               + sizeof(uint32_t)  // steps_per_write
                               + sizeof(uint32_t)  // bytes_per_atom
                               + sizeof(uint32_t)  // field_count
                               + 3 * sizeof(uint8_t) * 32 // 3x32 byte fields
                               + sizeof(uint64_t); // front_matter_size

    // should pad the front_matter_size so this block is on a 8 byte boundary
    // this should pad out to 224 bytes
    uint32_t front_matter_padding = 8 - (front_matter_size % 8);
    front_matter_size += front_matter_padding;

    // then the atom types start at a 8 byte boundary and then
    // I will insert padding at the end to make sure they stay on the 8 byte boundary

    front_matter_size += state.num_atoms; // 8 bit unsigned ints for each atom
    uint32_t atom_type_padding = 8 - (state.num_atoms % 8);
    front_matter_size += atom_type_padding;

    // create a buffer
    std::vector<uint8_t> buf(front_matter_size);

    uint32_t offset = 0;

    // magic
    const char* magic = "DISTNNP\x00";
    memcpy(buf.data() + offset, magic, 8);
    offset += 8;

    // version
    const uint32_t version = 1;
    memcpy(buf.data() + offset, &version, sizeof(version));
    offset += sizeof(version);

    // run_id
    std::string r = run_id;
    memcpy(buf.data() + offset, r.data(), r.size());
    memset(buf.data() + offset + r.size(), 0, 16 - r.size());
    offset += 16;

    // author
    memcpy(buf.data() + offset, author.data(), author.size());
    memset(buf.data() + offset + author.size(), 0, 64 - author.size());
    offset += 64;

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
    const uint32_t bytes_per_atom = 8 + (3 * 4) + (3 * 4);
    memcpy(buf.data() + offset, &bytes_per_atom, sizeof(bytes_per_atom));
    offset += sizeof(bytes_per_atom);

    // field_count
    const uint32_t field_count = 3;
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
    offset += sizeof(front_matter_size);

    // padding
    memset(buf.data() + offset, 0, front_matter_padding);
    offset += front_matter_padding;

    // tensor of atomic numbers
    // TODO this could be large and its own separate write to the file to make things easier probably
    auto t = state.atom_types.contiguous();

    size_t tensor_bytes = t.numel() * t.element_size();
    memcpy(buf.data() + offset, t.data_ptr(), tensor_bytes);
    offset += tensor_bytes;

    // padding
    memset(buf.data() + offset, 0, atom_type_padding);
    offset += atom_type_padding;

    // write to the file
    ssize_t written = pwrite(fd, buf.data(), front_matter_size, 0);
    if (written < 0) {
        throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (static_cast<size_t>(written) != buf.size()) {
        throw std::runtime_error("partial write");
    }
}
void TrajectoryWriter::write_frame(uint32_t frame_idx, torch::Tensor indices, torch::Tensor positions, torch::Tensor velocities) {

    // to make this easier, i think im going to write each tensor one after the other
    // so it will be
    // vertex_ids | uint8 x frame_size
    // atom_positions | 3 x float32 x frame_size
    // atom_velocities | 3 x float32 x frame_size

    if (writes == 0) {
        // first write, don't have these values saved
        // need to read in total_atoms, bytes_per_atom, front_matter_size

        // TODO need to guarentee that rank 0 is done writing

        auto n = pread(fd, &total_atoms, sizeof(total_atoms), 92);
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(total_atoms)) throw std::runtime_error("unexpected end of file");

        n = pread(fd, &bytes_per_atom, sizeof(bytes_per_atom), 108);
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(bytes_per_atom)) throw std::runtime_error("unexpected end of file");

        n = pread(fd, &front_matter_size, sizeof(front_matter_size), 212);
        if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
        if (n != sizeof(front_matter_size)) throw std::runtime_error("unexpected end of file");
    }

    uint32_t offset = front_matter_size + (writes * total_atoms * bytes_per_atom);

    if (offset < 220)
        throw std::runtime_error("Illegal offset");


    uint64_t atom_count = 0;
    for (int i = 0; i < rank; i++) {
        atom_count += rank_atom_counts.at(i);
    }

    // doing sizeof here to be more explicit about the calculations
    uint32_t indices_offset = offset + (atom_count * sizeof(uint64_t));
    offset += total_atoms * sizeof(uint64_t);
    uint32_t positions_offset = offset + (atom_count * 3 * sizeof(float));
    offset += total_atoms * 3 * sizeof(float);
    uint32_t velocities_offset = offset + (atom_count * 3 * sizeof(float));



    auto write_tensor = [&](torch::Tensor t, uint32_t off) {
        size_t tensor_bytes = t.numel() * t.element_size();
        ssize_t written = pwrite(fd, t.data_ptr(), tensor_bytes, off);

        // TODO might have to make this a loop if cant write the entire tensor at once
        if (written < 0) throw std::runtime_error("pwrite failed: " + std::string(strerror(errno)));
        if (static_cast<size_t>(written) != tensor_bytes) throw std::runtime_error("partial write");
    };

    write_tensor(indices.contiguous(), indices_offset); // TODO make sure on CPU
    write_tensor(positions.contiguous(), positions_offset); // TODO make sure on CPU
    write_tensor(velocities.contiguous(), velocities_offset); // TODO make sure on CPU

    // to calculate frame offsets better
    writes += 1;
}

MolecularInfo TrajectoryWriter::load_distnnp_file(std::string distnnp_file_name) {
    // reads the last frame in from the file and returns it as the current state
    MolecularInfo atoms;

    int dist_fd = open(distnnp_file_name.c_str(), O_RDWR, 0644);

    if (dist_fd < 0) {
        std::cerr << "Failed to open " << distnnp_file_name << std::endl;
        exit(1);
    }

    char magic[8];
    auto n = pread(dist_fd, magic, 8, 0);
    if (strcmp(magic, "DISTNNP\0") != 0) {
        // TODO handle endian mismatch
        std::cerr << distnnp_file_name << " is not a valid distnnp file" << std::endl;
        return atoms;
    }
    uint32_t version;
    n = pread(dist_fd, &version, sizeof(version), 8);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != sizeof(version)) throw std::runtime_error("unexpected end of file");

    if (version != 1) {
        std::cerr << version << " not supported. Version 1 is only supported" << std::endl;
        return atoms;
    }

    uint32_t bytes_per_atom;
    uint64_t front_matter_size;

    n = pread(dist_fd, &atoms.num_atoms, sizeof(atoms.num_atoms), 92);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != sizeof(atoms.num_atoms)) throw std::runtime_error("unexpected end of file");

    n = pread(dist_fd, &bytes_per_atom, sizeof(bytes_per_atom), 108);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != sizeof(bytes_per_atom)) throw std::runtime_error("unexpected end of file");

    n = pread(dist_fd, &front_matter_size, sizeof(front_matter_size), 212);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != sizeof(front_matter_size)) throw std::runtime_error("unexpected end of file");

    // 224 bytes is the offset of the first data and then its atom types
    std::vector<uint8_t> types(atoms.num_atoms);
    n = pread(dist_fd, types.data(), types.size(), 224);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != types.size()) throw std::runtime_error("unexpected end of file");
    atoms.atom_types = torch::from_blob(types.data(), {atoms.num_atoms}, torch::kUInt8).clone();

    uint64_t frame_size = bytes_per_atom * atoms.num_atoms;

    uint32_t size = lseek(dist_fd, 0, SEEK_END);
    uint32_t loc = size - frame_size; // start of frame

    // ids
    
    uint32_t block_size = atoms.num_atoms * sizeof(uint64_t);
    // offset is the start of the frame
    std::vector<uint64_t> ids(atoms.num_atoms);
    n = pread(dist_fd, ids.data(), block_size, loc);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != block_size) throw std::runtime_error("unexpected end of file");
    atoms.atom_vertex_numbers = torch::from_blob(ids.data(), {atoms.num_atoms}, torch::kUInt64).clone();

    loc += block_size; // new loc is at the end of this chunk of data

    // positions
    // resize to 3 floats per atom
    block_size = atoms.num_atoms * 3 * sizeof(float);
    std::vector<float> attribs(atoms.num_atoms * 3);
    n = pread(dist_fd, attribs.data(), block_size, loc);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != block_size) throw std::runtime_error("unexpected end of file");
    atoms.atom_positions = torch::from_blob(attribs.data(), {atoms.num_atoms, 3}).clone();

    loc += block_size; // new loc is at the end of this chunk of data

    // velocities
    n = pread(dist_fd, attribs.data(), block_size, loc);
    if (n < 0) throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    if (n != block_size) throw std::runtime_error("unexpected end of file");
    atoms.atom_velocities = torch::from_blob(attribs.data(), {atoms.num_atoms, 3}).clone();

    return atoms;
}

MolecularInfo TrajectoryWriter::load_xyz_file(std::string xyz_file_name) {
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
    int num_atoms = std::stoi(line);

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
            if (name == "vel" || name == "momenta") {
                velocity_index = current_index;
            }

            current_index += std::stoi(count);
        }
    }

    // read in all of the positions
    std::vector<uint8_t> types;
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
        types.emplace_back(periodic_table.at(element));
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
    atoms.atom_vertex_numbers = torch::arange(0, num_atoms, 1).to(torch::kUInt64);
    atoms.atom_types = torch::from_blob(types.data(), {num_atoms}, torch::kUInt8).clone();
    atoms.atom_positions = torch::from_blob(positions.data(), {num_atoms, 3}).clone();

    if (velocity_index == -1)
        atoms.atom_velocities = torch::zeros({num_atoms, 3});
    else
        atoms.atom_velocities = torch::from_blob(velocities.data(), {num_atoms, 3}).clone();

    return atoms;
}
void TrajectoryWriter::write_to_xyz(MolecularInfo state, std::string file_name) {

    std::unordered_map<uint8_t, std::string> reversed_table;

    for (const auto& [label, number] : periodic_table) {
        reversed_table.insert({number, label});
    }

    // accessors for the state info
    auto types = state.atom_types.accessor<uint8_t, 1>();
    auto positions = state.atom_positions.accessor<float, 2>();
    auto velocities = state.atom_velocities.accessor<float, 2>();

    // loop through and write to the xyz file
    std::ofstream out(output_dir + "/" + file_name);

    auto write_floats = [&](float x, float y, float z) {
        out << std::to_string(x)
            << '\t' << std::to_string(y)
            << '\t' << std::to_string(z);
    };

    out << std::to_string(state.num_atoms) << '\n';
    out << "Properties=species:S:1:pos:R:3:momenta:R:3\n";
    for (int i = 0; i < state.num_atoms; i++) {
        out << reversed_table[types[i]] << '\t';
        write_floats(positions[i][0], positions[i][1], positions[i][2]);
        out << '\t';
        write_floats(velocities[i][0], velocities[i][1], velocities[i][2]);
        out << '\n';
    }
    out.close();
}

} // namespace distnnp_state
