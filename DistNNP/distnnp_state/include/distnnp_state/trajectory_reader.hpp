#pragma once
#include "distnnp_state.hpp"
#include <string>
#include <tuple>
#include <vector>

namespace distnnp_state {

// Read of one frame from a .xyz/.distnnp file, frame_index of -1 is the last frame
// throws a std::runtime_error on any failure
std::tuple<MolecularInfo, torch::Tensor, torch::Tensor> read_molecular_info(const std::string& path, int64_t frame_index = -1);

// Translates all frames of a .distnnp file to a multi-frame .xyz file
// opts determine what fields are written, empty writes everythin
void translate_distnnp_to_xyz(const std::string& distnnp_path, const std::string& xyz_path, std::vector<std::string> opts = {});

} // namespace distnnp_state
