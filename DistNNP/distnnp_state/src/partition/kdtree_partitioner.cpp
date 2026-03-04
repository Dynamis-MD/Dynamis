// kdtree_partitioner.cpp — KD-tree recursive bisection partitioner.
//
// Three modes:
//   1. Unweighted:     edge_counts undefined, no SimulationConfig
//   2. User-weighted:  edge_counts set on MolecularInfo
//   3. Auto-weighted:  edge_counts undefined + SimulationConfig provided
//      → builds graph at cutoff, computes edge_counts, stores edge_index

#include "distnnp_state/partitioner.hpp"
#include "partition/kdtree_cuda_.h"
#include "radius_pbc/radius_pbc_.h"

namespace distnnp_state
{

static constexpr int64_t PARTITION_MAX_NUM_NEIGHBORS = 1000;

// ---------------------------------------------------------------------------
// Recursive bisection: split atoms along the longest-extent axis.
// Weighted mode splits so each half has ~equal cumulative edge cost.
// ---------------------------------------------------------------------------
static void kdtree_bisect(
    torch::Tensor idx,                 // [M] global atom indices for this subset
    torch::Tensor &labels,             // [N] output partition labels (mutated in-place)
    const torch::Tensor &coords,       // [N, 3] all atom positions
    const torch::Tensor &weights,      // [N] edge counts (may be undefined)
    int64_t parts,
    int64_t base_label)
{
    if (parts <= 1 || idx.numel() == 0) {
        labels.index_put_({idx}, base_label);
        return;
    }

    auto sub_coords = coords.index_select(0, idx);  // [M, 3]

    // Find longest axis
    auto extent = std::get<0>(sub_coords.max(0)) - std::get<0>(sub_coords.min(0));
    int64_t axis = extent.argmax().item<int64_t>();

    // Sort along that axis
    auto order = std::get<1>(sub_coords.select(1, axis).sort(0));
    auto sorted_idx = idx.index_select(0, order);
    int64_t M = sorted_idx.size(0);

    int64_t left_parts = parts / 2;
    int64_t right_parts = parts - left_parts;

    int64_t mid;
    if (!weights.defined()) {
        mid = M / 2;
    } else {
        auto sorted_w = weights.index_select(0, sorted_idx).to(torch::kFloat);
        auto cumsum = sorted_w.cumsum(0);
        float target = cumsum[-1].item<float>() / 2.0f;
        auto ge_mask = cumsum.ge(target);
        mid = ge_mask.nonzero().select(1, 0)[0].item<int64_t>();
        mid = std::max(int64_t(1), std::min(mid, M - 1));
    }

    kdtree_bisect(sorted_idx.slice(0, 0, mid), labels, coords, weights, left_parts, base_label);
    kdtree_bisect(sorted_idx.slice(0, mid, M), labels, coords, weights, right_parts, base_label + left_parts);
}

// ---------------------------------------------------------------------------
// Common helper: bisect + extract local atoms for rank R.
// ---------------------------------------------------------------------------
static PartitionMessagePassingInfo do_partition(
    const MolecularInfo &state,
    const DistEnvironment &env,
    const torch::Tensor &weights)   // may be undefined
{
    PartitionMessagePassingInfo info;
    auto device = state.atom_positions.device();
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);

    int64_t N = static_cast<int64_t>(state.num_atoms);
    int64_t W = static_cast<int64_t>(env.world_size);
    int64_t R = static_cast<int64_t>(env.global_rank);

    torch::Tensor labels;
    if (device.is_cuda()) {
        labels = kdtree_bisect_cuda(
            state.atom_positions.to(torch::kFloat), weights, W);
    } else {
        labels = torch::zeros({N}, long_opts);
        auto all_idx = torch::arange(N, long_opts);
        kdtree_bisect(all_idx, labels, state.atom_positions, weights, W, 0);
    }

    info.partition_id = labels;

    // Extract local atoms for rank R (non-contiguous partition)
    auto my_mask = labels.eq(R);
    auto my_indices = torch::nonzero(my_mask).squeeze(-1);
    int64_t num_local = my_indices.size(0);

    info.partition_indices     = state.atom_indices.index_select(0, my_indices);
    info.local_atom_positions  = state.atom_positions.index_select(0, my_indices);
    info.local_atom_velocities = state.atom_velocities.index_select(0, my_indices);

    info.global_send_recv_matrix = torch::zeros({W, W}, long_opts);
    info.global_send_recv_matrix.index_put_({R, R}, num_local);

    return info;
}

// ---------------------------------------------------------------------------
// Mode 1 & 2: unweighted or user-provided edge_counts
// ---------------------------------------------------------------------------
PartitionMessagePassingInfo KDTreePartitioner::partition(
    const MolecularInfo &state, const DistEnvironment &env)
{
    return do_partition(state, env, state.edge_counts);
}

// ---------------------------------------------------------------------------
// Mode 3: auto-compute edge weights via radius_graph_pbc (CUDA required).
// Builds graph at config.cutoff, counts edges per atom (dst-based),
// then partitions by cumulative edge cost.
// Stores graph in info.edge_index for downstream NNP model use.
// ---------------------------------------------------------------------------
PartitionMessagePassingInfo KDTreePartitioner::partition(
    const MolecularInfo &state,
    const DistEnvironment &env,
    const SimulationConfig &config)
{
    auto pos = state.atom_positions;
    auto device = pos.device();
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    int64_t N = static_cast<int64_t>(state.num_atoms);

    float cutoff = static_cast<float>(config.cutoff);
    auto cell = config.cell.to(device).to(torch::kFloat);
    auto pbc  = config.pbc.to(device);

    // Build graph at cutoff radius
    auto graph_result = radius_graph_pbc(
        pos.to(torch::kFloat), cell, pbc, cutoff,
        PARTITION_MAX_NUM_NEIGHBORS, /*enforce_max_strictly=*/false);
    auto edge_idx = std::get<0>(graph_result);  // [2, E]

    // Count edges per atom (dst-based, matching baseline: compute is on dst atoms)
    auto dst = edge_idx[1];  // [E]
    auto edge_counts = torch::zeros({N}, long_opts);
    edge_counts.scatter_add_(0, dst, torch::ones({dst.size(0)}, long_opts));

    // Partition using computed weights
    auto info = do_partition(state, env, edge_counts);

    // Store graph for downstream use
    info.edge_index = edge_idx;

    return info;
}

} // namespace distnnp_state
