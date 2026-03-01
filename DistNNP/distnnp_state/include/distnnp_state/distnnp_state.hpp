#pragma once
#include <torch/extension.h>

namespace distnnp_state
{
    struct MolecularInfo
    {
        unsigned long num_atoms; // Number of atoms in this partition
        torch::Tensor atom_indices; // Tensor of shape [num_atoms] containing global atom indices for this partition
        torch::Tensor atom_positions; // Tensor of shape [num_atoms, 3] containing local atomic positions for this partition
        torch::Tensor atom_velocities; // Tensor of shape [num_atoms, 3] containing local atomic velocities for this partition
    };

    struct DistEnvironment
    {
        unsigned long global_rank;
        unsigned long world_size;
        unsigned long gpus_per_node;
        unsigned long num_nodes;
        unsigned long local_rank;
    };

    struct PartitionMessagePassingInfo
    {
        torch::Tensor partition_indices; // Tensor of shape [num_local_atoms] containing global atom indices for this partition
        torch::Tensor local_atom_positions; // Tensor of shape [num_local_atoms, 3] containing local atomic positions for this partition
        torch::Tensor local_atom_velocities; // Tensor of shape [num_local_atoms, 3] containing local atomic velocities for this partition
        torch::tensor global_send_recv_matrix; // Tensor of shape [world_size, world_size] containing the number of atoms to send from each rank to every other rank
        torch::tensor halo_map;
    };
    
    struct DistNNPState
    {
      // Distributed environment information
      DistEnvironment dist_env;

      // Molecular system information
      
      MolecularInfo global_molecular_info; // Contains global atom information for the entire system
      
      // Partitioned data information
      MolecularInfo local_partition_info; // Contains local atom information for this partition

    };
} // namespace distnnp_state
