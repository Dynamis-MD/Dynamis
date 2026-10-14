#pragma once
#include "trajectory_format.hpp"
#include "dynamis_state.hpp"
#include <c10/util/Optional.h>
#include <cstdint>
#include <string>
#include <vector>

namespace dynamis_state {

class TrajectoryWriterImpl
{
public:
    TrajectoryWriterImpl(uint32_t global_rank, py::bytes run_id, const std::string &author,
                         const std::string &output_dir, const std::string &output_file,
                         const MolecularInfo &molecular_info, torch::Tensor cell, torch::Tensor pbc,
                         float time_per_step, uint32_t steps_per_write,
                         std::vector<std::string> opts = {});
    ~TrajectoryWriterImpl();

    // cell/pbc are carried forward from last seen value if not provided
    void submit(uint32_t step, MolecularInfo state,
                c10::optional<torch::Tensor> cell = c10::nullopt,
                c10::optional<torch::Tensor> pbc = c10::nullopt);
    void flush();

    torch::Tensor get_cell() const { return current_cell; }
    torch::Tensor get_pbc() const { return current_pbc; }

    void update_atom_counts(torch::Tensor counts)
    {
        // Must save to a named variable before calling data_ptr — the temporary
        // from .to(kCPU).contiguous() would be destroyed at the semicolon,
        // leaving a dangling pointer into freed storage.
        torch::Tensor cpu_counts = counts.to(torch::kCPU).to(torch::kLong).contiguous();
        int64_t *data = cpu_counts.data_ptr<int64_t>();
        rank_atom_counts.clear();
        rank_atom_counts.reserve(cpu_counts.numel());
        for (int64_t i = 0; i < cpu_counts.numel(); i++) {
            rank_atom_counts.push_back(static_cast<uint32_t>(data[i]));
        }
    }

private:
    void write_front_matter(float time_per_step, uint32_t steps_per_write,
                            const MolecularInfo &state, uint32_t rank);
    void parse_opts(const std::vector<std::string> &opts);

    // only set to true if the file has been closed
    bool flushed{ false };

    py::bytes run_id;
    std::string author;

    uint64_t total_atoms{ 0 };
    uint32_t bytes_per_atom{ 0 };
    uint32_t front_matter_size{ 0 };

    int fd; // fd of the output file
    uint32_t rank; // rank of the current process
    std::vector<uint32_t> rank_atom_counts; // atom offsets
    std::string output_dir;
    std::string output_file;

    bool has_indices{ false };
    bool has_positions{ false };
    bool has_velocities{ false };

    // last known simulation cell / pbc flags, written into every submitted
    // frame's pbc_cell block; updated whenever submit() is given new values
    torch::Tensor current_cell; // [3, 3] float32
    torch::Tensor current_pbc; // [3] uint8 (0 or 1)
};

} // namespace dynamis_state
