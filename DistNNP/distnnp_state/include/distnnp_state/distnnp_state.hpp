#pragma once
#include <string>
#include <torch/extension.h>

namespace distnnp_state {
struct MolecularInfo {
  unsigned long num_atoms;           // Number of atoms in this partition
  torch::Tensor atom_vertex_numbers; // Tensor of shape [num_atoms] containing
                                     // atom indices
  torch::Tensor atom_types; // Tensor of shape [num_atoms] containing atom types
  torch::Tensor atom_positions;  // Tensor of shape [num_atoms, 3] containing
                                 // atomic positions
  torch::Tensor atom_velocities; // Tensor of shape [num_atoms, 3] containing
                                 // atomic velocities
  torch::Tensor edge_counts;     // [num_atoms], optional per-atom edge cost for
                             // weighted partitioning. Undefined = unweighted.
};

struct DistEnvironment {
  unsigned long global_rank;
  unsigned long world_size;
  unsigned long gpus_per_node;
  unsigned long num_nodes;
  unsigned long local_rank;
};

struct SimulationConfig {
  double cutoff;              // Neighbor cutoff radius
  double ghost_cutoff_buffer; // Extra buffer for ghost zone (default 0.5)
  torch::Tensor cell;         // [3, 3] simulation cell matrix
  torch::Tensor pbc;          // [3] periodic boundary conditions (int, 0 or 1)
};

struct PartitionMessagePassingInfo {
  // --- Partition assignment ---
  torch::Tensor partition_id; // [num_atoms] rank assignment for every atom

  // --- Partition results ---
  torch::Tensor
      partition_indices; // [num_local] global atom indices for this rank
  torch::Tensor
      global_send_recv_matrix; // [world_size, world_size] — only row[rank] is
                               // filled locally; needs all-reduce for full view

  // --- Halo communication maps (CSR, push-only) ---
  torch::Tensor send_offsets;       // [world_size + 1] CSR row pointers
  torch::Tensor send_local_indices; // [total_send] local atom indices to push
                                    // to neighbors
  torch::Tensor recv_offsets;       // [world_size + 1] CSR row pointers
  torch::Tensor recv_ghost_indices; // [total_recv] local ghost slot indices

  // --- Interior mask ---
  torch::Tensor
      interior_mask; // [num_local] bool, true = all neighbors are local

  // --- Cached graphs (returned for downstream NNP model use) ---
  torch::Tensor edge_index; // [2, E] graph at cutoff radius (from auto-weighted
                            // partition). Undefined if not built.
  torch::Tensor
      edge_index_halo; // [2, E'] graph at cutoff + ghost_buffer radius (from
                       // build_halo_maps). Undefined if not built.

  torch::Tensor atom_counts; // [world_size] atom counts for each rank, needed
                             // for trajectory writing
};

struct DistNNPState {
  // Distributed environment information
  DistEnvironment dist_env;

  // Simulation configuration
  SimulationConfig sim_config;

  // Molecular system information
  MolecularInfo global_molecular_info; // Contains global atom information for
                                       // the entire system

  MolecularInfo local_molecular_info; // Contains local atom information for
                                      // this partition

  // Partitioned data information
  PartitionMessagePassingInfo
      local_partition_info; // Contains local atom communication information for
                            // this partition
};

} // namespace distnnp_state
