#pragma once
#include "distnnp_state/distnnp_state.hpp"

class AbstractPartitioner
{
public:
  virtual ~AbstractPartitioner() = default;
  PartitionMessagePassingInfo partition(const DistNNPState &state,
                                        const DistEnvironment &env) = 0;
};