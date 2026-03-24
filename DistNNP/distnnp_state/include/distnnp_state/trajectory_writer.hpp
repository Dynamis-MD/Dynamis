#pragma once
#include "distnnp_state.hpp"
#include <cstdint>
#include <vector>

namespace distnnp_state {

class TrajectoryWriter {
public:
    explicit TrajectoryWriter(uint32_t global_rank, py::bytes run_id, const std::string& author, std::string output_dir = "./trajectory");

    // TODO eventually this can be initialize with the custom binary format as a file
    MolecularInfo initialize(std::string xyz_file_name, float time_per_step, float steps_per_write);

    void write_frame(int fd, uint32_t frame_idx, torch::Tensor indices, torch::Tensor positions, torch::Tensor velocities);

    void update_atom_counts(std::vector<uint32_t> counts) {
        rank_atom_counts = std::move(counts);
    }

private:
    MolecularInfo load_file(std::string xyz_file_name);
    void write_front_matter(float time_per_step, float steps_per_write, MolecularInfo state);

    py::bytes run_id;
    std::string author;

    uint32_t writes{ 0 };

    uint32_t total_atoms;
    uint32_t bytes_per_atom;
    uint32_t front_matter_size;

    int fd; // fd of the output file
    uint32_t rank; // rank of the current process
    std::vector<uint32_t> rank_atom_counts; // atom offsets
};

}
