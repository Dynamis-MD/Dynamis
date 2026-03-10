#pragma once
#include <torch/extension.h>

namespace distnnp_state
{
    struct MolecularInfo
    {
        unsigned long num_atoms;          // Number of atoms in this partition
        torch::Tensor atom_vertex_number; // Tensor of shape [num_atoms] containing global atom indices
        torch::Tensor atom_types;         // Tensor of shape [num_atoms] containing global atom types
        torch::Tensor atom_positions;     // Tensor of shape [num_atoms, 3] containing atomic positions
        torch::Tensor atom_velocities;    // Tensor of shape [num_atoms, 3] containing atomic velocities
    };

    struct DistEnvironment
    {
        unsigned long global_rank;
        unsigned long world_size;
        unsigned long gpus_per_node;
        unsigned long num_nodes;
        unsigned long local_rank;
    };

    struct GeometryConfig
    {
        double cutoff;              // Neighbor cutoff radius
        double ghost_cutoff_buffer; // Extra buffer for ghost zone (default 0.5)
        torch::Tensor cell;         // [3, 3] cell matrix
        torch::Tensor pbc;          // [3] periodic boundary conditions (int, 0 or 1)
    };

    struct PartitionMessagePassingInfo
    {
        // --- Partition assignment ---
        torch::Tensor partition_id; // [num_atoms] rank assignment for every atom

        // --- Partition results ---
        torch::Tensor partition_indices;       // [num_local] global atom indices for this rank
        torch::Tensor global_send_recv_matrix; // [world_size, world_size] — only row[rank] is filled locally; needs all-reduce for full view

        // --- Halo communication maps (CSR, push-only) ---
        torch::Tensor send_offsets;       // [world_size + 1] CSR row pointers
        torch::Tensor send_local_indices; // [total_send] local atom indices to push to neighbors
        torch::Tensor recv_offsets;       // [world_size + 1] CSR row pointers
        torch::Tensor recv_ghost_indices; // [total_recv] local ghost slot indices

        // --- Interior mask ---
        torch::Tensor interior_mask; // [num_local] bool, true = all neighbors are local
    };

    struct DistNNPState
    {
        // Distributed environment information
        DistEnvironment dist_env;

        // Structure geometric configuration
        GeometryConfig geometry_config;

        // Molecular system information
        // Contains global atom information for the entire system
        // Stored on the CPU for coherency, assumed to be negligible in size
        MolecularInfo global_molecular_info; 

        // Partitioned Local Molecular information
        MolecularInfo local_molecular_info; // Used for computation

        // Partitioned data information
        PartitionMessagePassingInfo local_partition_info; // Contains local atom information for this partition
    };
} // namespace distnnp_state
