#pragma once
#include "distnnp_state/distnnp_state.hpp"

namespace distnnp_state {

class AbstractPartitioner {
public:
  virtual ~AbstractPartitioner() = default;

  // Subclass implements: fills partition_id, partition_indices,
  // local_atom_positions, local_atom_velocities, global_send_recv_matrix.
  virtual PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                                const DistEnvironment &env) = 0;

  // Shared: builds CSR halo maps using info.partition_id + CUDA
  // radius_graph_pbc. Fills send_offsets, send_local_indices, recv_offsets,
  // recv_ghost_indices, interior_mask, and updates global_send_recv_matrix with
  // halo counts.
  void build_halo_maps(PartitionMessagePassingInfo &info,
                       const MolecularInfo &global_state,
                       const DistEnvironment &env,
                       const SimulationConfig &config);
};

class SimpleSpatialPartitioner : public AbstractPartitioner {
  // Identity partitioner: contiguous split of global atoms across ranks.
  // Rank r gets atoms [r*N/W, (r+1)*N/W).
public:
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;
};

class KDTreePartitioner : public AbstractPartitioner {
  // Uses a KD-tree to spatially partition the atoms. More efficient
  // for large systems with non-uniform atom distributions.
  //
  // Three partition modes:
  //   1. Unweighted: edge_counts undefined, no SimulationConfig
  //   2. User-weighted: edge_counts set on MolecularInfo
  //   3. Auto-weighted: edge_counts undefined + SimulationConfig provided
  //      → builds graph at cutoff, computes edge_counts, stores edge_index
public:
  // Modes 1 & 2: unweighted or user-provided weights
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;

  // Mode 3: auto-compute edge weights via radius_graph_pbc (CUDA only).
  // Builds graph at config.cutoff, counts edges per atom, partitions by cost.
  // Stores the graph in info.edge_index for downstream use.
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env,
                                        const SimulationConfig &config);
};

enum class SFCCurveType : int { Morton = 0, Hilbert = 1 };

class SFCPartitioner : public AbstractPartitioner {
  // Uses a space-filling curve (Morton Z-curve or Hilbert curve) to partition
  // the atoms. Good locality and load balancing for certain systems.
  //
  // Three partition modes (same as KDTreePartitioner):
  //   1. Unweighted: edge_counts undefined, no SimulationConfig
  //   2. User-weighted: edge_counts set on MolecularInfo
  //   3. Auto-weighted: edge_counts undefined + SimulationConfig provided
  //      → builds graph at cutoff, computes edge_counts, stores edge_index
  SFCCurveType curve_type_;

public:
  explicit SFCPartitioner(SFCCurveType ct = SFCCurveType::Morton)
      : curve_type_(ct) {}

  // Modes 1 & 2: unweighted or user-provided weights
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;

  // Mode 3: auto-compute edge weights via radius_graph_pbc (CUDA only).
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env,
                                        const SimulationConfig &config);
};

} // namespace distnnp_state
