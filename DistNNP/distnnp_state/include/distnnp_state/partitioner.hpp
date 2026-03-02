#pragma once
#include "distnnp_state/distnnp_state.hpp"

class AbstractPartitioner
{
public:
  virtual ~AbstractPartitioner() = default;
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) = 0;
  
};

class SimpleSpatialPartitioner : public AbstractPartitioner
{
  /*

  */
public:
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;
};

class KDTreePartitioner : public AbstractPartitioner
{
  /*
    This partitioner uses a KD-tree to spatially partition the atoms. It should
    be more efficient for large systems with non-uniform atom distributions.
  */
public:
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;
};


class SFCPartitioner : public AbstractPartitioner
{
  /*
    This partitioner uses a space-filling curve (e.g. Hilbert curve) to partition
    the atoms. It can provide good locality and load balancing for certain systems.
  */
public:
  PartitionMessagePassingInfo partition(const MolecularInfo &state,
                                        const DistEnvironment &env) override;
};