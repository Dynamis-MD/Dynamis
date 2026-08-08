#pragma once
#include "distnnp_state.hpp"
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace distnnp_state {

class TrajectoryWriter {
public:
    explicit TrajectoryWriter(uint32_t global_rank, py::bytes run_id, const std::string& author, const std::string& output_dir = "./trajectory", const std::string& output_file = "trajectory.distnnp");
    ~TrajectoryWriter();

    // TODO eventually this can be initialize with the custom binary format as a file
    MolecularInfo initialize(const std::string& input_file_name, const float time_per_step, const uint32_t steps_per_write);

    // cell/pbc are optional and carried forward from the last value seen
    // (either from initialize() or a previous submit()) if not provided,
    // so a static-box simulation only needs to pass them once.
    void submit(uint32_t step, MolecularInfo state, c10::optional<torch::Tensor> cell = c10::nullopt, c10::optional<torch::Tensor> pbc = c10::nullopt);
    void flush();

    torch::Tensor get_cell() const { return current_cell; }
    torch::Tensor get_pbc() const { return current_pbc; }

    void update_atom_counts(torch::Tensor counts) {
        // Must save to a named variable before calling data_ptr — the temporary
        // from .to(kCPU).contiguous() would be destroyed at the semicolon,
        // leaving a dangling pointer into freed storage.
        torch::Tensor cpu_counts = counts.to(torch::kCPU).to(torch::kLong).contiguous();
        int64_t* data = cpu_counts.data_ptr<int64_t>();
        rank_atom_counts.clear();
        rank_atom_counts.reserve(cpu_counts.numel());
        for (int64_t i = 0; i < cpu_counts.numel(); i++) {
            rank_atom_counts.push_back(static_cast<uint32_t>(data[i]));
        }
    }

    void write_to_xyz(const MolecularInfo& state, const std::string& file_name = "output.xyz");

private:
    std::string output_file;
    MolecularInfo load_xyz_file(const std::string& xyz_file_name);
    MolecularInfo load_distnnp_file(const std::string& xyz_file_name);
    void write_front_matter(float time_per_step, uint32_t steps_per_write, MolecularInfo state, uint32_t rank);

    // only set to true if the file has been closed
    bool flushed{ false };

    py::bytes run_id;
    std::string author;

    uint64_t total_atoms;
    uint32_t bytes_per_atom;
    uint32_t front_matter_size;

    int fd; // fd of the output file
    uint32_t rank; // rank of the current process
    std::vector<uint32_t> rank_atom_counts; // atom offsets
    std::string output_dir;

    // last known simulation cell / pbc flags, written into every submitted
    // frame's pbc_cell block; updated whenever submit() is given new values
    torch::Tensor current_cell; // [3, 3] float32
    torch::Tensor current_pbc;  // [3] uint8 (0 or 1)
};


// TODO figure out a better way to do this
// only doing up to Lead as idk if we need any of the unstable elements
static const std::unordered_map<std::string, uint8_t> periodic_table = {
    {"H", 1},
    {"He", 2},
    {"Li", 3},
    {"Be", 4},
    {"B", 5},
    {"C", 6},
    {"N", 7},
    {"O", 8},
    {"F", 9},
    {"Ne", 10},
    {"Na", 11},
    {"Mg", 12},
    {"Al", 13},
    {"Si", 14},
    {"P", 15},
    {"S", 16},
    {"Cl", 17},
    {"Ar", 18},
    {"K", 19},
    {"Ca", 20},
    {"Sc", 21},
    {"Ti", 22},
    {"V", 23},
    {"Cr", 24},
    {"Mn", 25},
    {"Fe", 26},
    {"Co", 27},
    {"Ni", 28},
    {"Cu", 29},
    {"Zn", 30},
    {"Ga", 31},
    {"Ge", 32},
    {"As", 33},
    {"Se", 34},
    {"Br", 35},
    {"Kr", 36},
    {"Rb", 37},
    {"Sr", 38},
    {"Y", 39},
    {"Zr", 40},
    {"Nb", 41},
    {"Mo", 42},
    {"Tc", 43},
    {"Ru", 44},
    {"Rh", 45},
    {"Pd", 46},
    {"Ag", 47},
    {"Cd", 48},
    {"In", 49},
    {"Sn", 50},
    {"Sb", 51},
    {"Te", 52},
    {"I", 53},
    {"Xe", 54},
    {"Cs", 55},
    {"Ba", 56},
    {"La", 57},
    {"Ce", 58},
    {"Pr", 59},
    {"Nd", 60},
    {"Pm", 61},
    {"Sm", 62},
    {"Eu", 63},
    {"Gd", 64},
    {"Tb", 65},
    {"Dy", 66},
    {"Ho", 67},
    {"Er", 68},
    {"Tm", 69},
    {"Yb", 70},
    {"Lu", 71},
    {"Hf", 72},
    {"Ta", 73},
    {"W", 74},
    {"Re", 75},
    {"Os", 76},
    {"Ir", 77},
    {"Pt", 78},
    {"Au", 79},
    {"Hg", 80},
    {"Ti", 81},
    {"Pb", 82}
};
}
