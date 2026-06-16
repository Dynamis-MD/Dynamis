#pragma once
#include "distnnp_state.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace distnnp_state {

class TrajectoryWriter {
public:
    explicit TrajectoryWriter(uint32_t global_rank, py::bytes run_id, const std::string& author, std::string output_dir = "./trajectory");
    ~TrajectoryWriter();

    // TODO eventually this can be initialize with the custom binary format as a file
    MolecularInfo initialize(const std::string& input_file_name, const float time_per_step, const uint32_t steps_per_write);

    void submit(uint32_t step, MolecularInfo state);
    void flush();

    void update_atom_counts(torch::Tensor counts) {
        uint32_t* data_ptr = counts.to(torch::kCPU).contiguous().data_ptr<uint32_t>();
        rank_atom_counts = std::vector<uint32_t>(data_ptr, data_ptr + counts.numel());
    }

    void write_to_xyz(const MolecularInfo& state, const std::string& file_name = "output.xyz");

private:
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
