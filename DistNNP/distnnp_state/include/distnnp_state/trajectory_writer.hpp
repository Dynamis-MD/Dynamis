#pragma once
#include "distnnp_state.hpp"

namespace distnnp_state {

class TrajectoryWriter {
public:
    MolecularInfo load_state(std::string xyz_file_name);
    void write_front_matter(uint64_t total_atoms, char* run_id, char* author);
    void write_frame(int fd, uint32_t frame_idx, MolecularInfo atom_info);

private:
};

}
