#pragma once
#include "distnnp_state.hpp"

namespace distnnp_state {

// TODO this function will probably end up in a class for all io
MolecularInfo load_state(std::string xyz_file_name);

}
