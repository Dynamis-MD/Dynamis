#pragma once
#include "distnnp_state.hpp"
#include <optional>

namespace distnnp_state {

class TrajectoryWriter {
public:
    explicit TrajectoryWriter(uint32_t global_rank, uint32_t world_size, std::string xyz_file_name, char* run_id, char* author);
    MolecularInfo load_state(std::string xyz_file_name);
    void write_front_matter(uint32_t world_size, char* run_id, char* author);
    void write_frame(int fd, uint32_t frame_idx, MolecularInfo atom_info);

private:
    MolecularInfo current_state;
    int fd;
};

}
