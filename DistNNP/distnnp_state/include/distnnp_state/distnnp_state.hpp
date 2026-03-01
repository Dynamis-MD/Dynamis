#pragma once
#include <torch/extension.h>

namespace distnnp_state
{
    struct DistNNPState
    {
      // Distributed environment information
      unsigned long global_rank;
      unsigned long world_size;
      unsigned long gpus_per_node;
      unsigned long num_nodes;
      unsigned long local_rank;

      // Molecular system information
      unsigned long num_atoms;
      torch::Tensor global_atom_list; // Tensor of shape [num_atoms] containing atomic numbers or type indices
      torch::Tensor global_atom_positions;  // Tensor of shape [num_atoms, 3] containing atomic positions
      torch::Tensor global_atom_velocities; // Tensor of shape [num_atoms, 3] containing atomic velocities

      // Partitioned data information


    };
} // namespace distnnp_state