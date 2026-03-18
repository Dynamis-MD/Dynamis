#include "distnnp_state/md_io.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace distnnp_state {

MolecularInfo load_state(std::string xyz_file_name) {
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
    atoms.atom_vertex_numbers = torch::arange(0, num_atoms, 1);
    atoms.atom_types = torch::from_blob(types.data(), {num_atoms}, torch::kByte).clone();
    atoms.atom_positions = torch::from_blob(positions.data(), {num_atoms, 3}).clone();

    if (velocity_index == -1)
        atoms.atom_velocities = torch::zeros({num_atoms, 3});
    else
        atoms.atom_velocities = torch::from_blob(velocities.data(), {num_atoms, 3}).clone();

    return atoms;
}

} // namespace distnnp_state
