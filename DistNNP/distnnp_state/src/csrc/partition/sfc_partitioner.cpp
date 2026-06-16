// sfc_partitioner.cpp — SFC (Space-Filling Curve) partitioner.
//
// Three modes (matching KDTreePartitioner):
//   1. Unweighted:     edge_counts undefined, no SimulationConfig
//   2. User-weighted:  edge_counts set on MolecularInfo
//   3. Auto-weighted:  edge_counts undefined + SimulationConfig provided
//      → builds graph at cutoff, computes edge_counts, stores edge_index

#include "distnnp_state/partitioner.hpp"
#include "distnnp_state/sfc_cuda.hpp"
#include "distnnp_state/radius_pbc.hpp"

namespace distnnp_state
{

static constexpr int64_t PARTITION_MAX_NUM_NEIGHBORS = 1000;

// ---------------------------------------------------------------------------
// CPU fallback: Morton key computation (21-bit, bit-interleave loop)
// ---------------------------------------------------------------------------
static torch::Tensor compute_morton_keys_cpu(
    const torch::Tensor &pos,    // [N, 3]
    const torch::Tensor &pos_min,
    const torch::Tensor &pos_range)
{
    int64_t N = pos.size(0);
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(pos.device());
    auto coords_normalized = (pos - pos_min) / pos_range;

    const int64_t grid_max = (1 << 21) - 1;
    auto coords_int = (coords_normalized * static_cast<double>(grid_max)).to(torch::kLong);
    auto x = coords_int.select(1, 0);
    auto y = coords_int.select(1, 1);
    auto z = coords_int.select(1, 2);

    auto result = torch::zeros({N}, long_opts);
    for (int i = 0; i < 21; i++) {
        auto x_bit = x.bitwise_right_shift(i).bitwise_and(1);
        auto y_bit = y.bitwise_right_shift(i).bitwise_and(1);
        auto z_bit = z.bitwise_right_shift(i).bitwise_and(1);
        result.bitwise_or_(x_bit.bitwise_left_shift(3 * i));
        result.bitwise_or_(y_bit.bitwise_left_shift(3 * i + 1));
        result.bitwise_or_(z_bit.bitwise_left_shift(3 * i + 2));
    }
    return result;
}

// ---------------------------------------------------------------------------
// CPU fallback: Hilbert key computation (order=7, matching baseline)
// ---------------------------------------------------------------------------
static torch::Tensor compute_hilbert_keys_cpu(
    const torch::Tensor &pos,    // [N, 3]
    const torch::Tensor &pos_min,
    const torch::Tensor &pos_range)
{
    int64_t N = pos.size(0);
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(pos.device());
    auto coords_normalized = (pos - pos_min) / pos_range;

    const int order = 7;
    const int64_t grid_max = (1 << order) - 1;
    auto coords_int = (coords_normalized * static_cast<double>(grid_max)).to(torch::kLong).clamp(0, grid_max);
    auto x = coords_int.select(1, 0);
    auto y = coords_int.select(1, 1);
    auto z = coords_int.select(1, 2);

    auto result = torch::zeros({N}, long_opts);
    for (int i = 0; i < order; i++) {
        auto x_bit = x.bitwise_right_shift(i).bitwise_and(1);
        auto y_bit = y.bitwise_right_shift(i).bitwise_and(1);
        auto z_bit = z.bitwise_right_shift(i).bitwise_and(1);

        torch::Tensor bx, by, bz;
        if (i == 0) {
            bx = x_bit; by = y_bit; bz = z_bit;
        } else {
            auto x_mask = x_bit.to(torch::kBool);
            auto y_mask = y_bit.to(torch::kBool);
            auto z_mask = z_bit.to(torch::kBool);

            bx = torch::where(x_mask, z_bit, x_bit);
            by = torch::where(x_mask, y_bit,
                    torch::where(y_mask, z_bit, y_bit));
            bz = torch::where(x_mask, y_bit,
                    torch::where(y_mask, x_bit,
                        torch::where(z_mask, y_bit, z_bit)));
        }
        result.bitwise_or_(bx.bitwise_left_shift(3 * i)
                           .bitwise_or(by.bitwise_left_shift(3 * i + 1))
                           .bitwise_or(bz.bitwise_left_shift(3 * i + 2)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Common helper: SFC partition → extract local atoms for rank R.
// ---------------------------------------------------------------------------
static PartitionMessagePassingInfo do_partition(
    const MolecularInfo &state,
    const DistEnvironment &env,
    const torch::Tensor &weights,   // may be undefined
    SFCCurveType curve_type)
{
    PartitionMessagePassingInfo info;
    auto device = state.atom_positions.device();
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);

    int64_t N = static_cast<int64_t>(state.num_atoms);
    int64_t W = static_cast<int64_t>(env.world_size);
    int64_t R = static_cast<int64_t>(env.global_rank);
    int64_t ct = static_cast<int64_t>(curve_type);

    torch::Tensor labels;
    if (device.is_cuda()) {
        labels = sfc_partition_cuda(
            state.atom_positions.to(torch::kFloat), weights, W, ct);
    } else {
        // CPU fallback: compute keys → sort → assign
        auto pos = state.atom_positions.to(torch::kFloat);
        auto pos_min = std::get<0>(pos.min(0));
        auto pos_range = std::get<0>(pos.max(0)) - pos_min;
        pos_range = torch::where(pos_range > 0, pos_range, torch::ones_like(pos_range));

        torch::Tensor sfc_keys;
        if (curve_type == SFCCurveType::Morton) {
            sfc_keys = compute_morton_keys_cpu(pos, pos_min, pos_range);
        } else {
            sfc_keys = compute_hilbert_keys_cpu(pos, pos_min, pos_range);
        }

        auto perm = std::get<1>(at::sort(sfc_keys));
        labels = torch::empty({N}, long_opts);

        if (weights.defined()) {
            auto sorted_w = weights.to(torch::kFloat).index_select(0, perm);
            auto cumsum_w = sorted_w.cumsum(0);
            float total = cumsum_w[-1].item<float>();

            for (int64_t i = 0; i < N; i++) {
                float cs = cumsum_w[i].item<float>();
                int64_t p = static_cast<int64_t>(cs * static_cast<float>(W) / total);
                p = std::max(p, int64_t(0));
                p = std::min(p, W - 1);
                labels[perm[i].item<int64_t>()] = p;
            }
        } else {
            int64_t base = N / W;
            int64_t rem  = N % W;
            int64_t threshold = rem * (base + 1);

            for (int64_t i = 0; i < N; i++) {
                int64_t p;
                if (i < threshold) {
                    p = i / (base + 1);
                } else {
                    p = rem + (i - threshold) / base;
                }
                p = std::min(p, W - 1);
                labels[perm[i].item<int64_t>()] = p;
            }
        }
    }

    info.partition_id = labels;

    // Extract local atoms for rank R
    auto my_mask = labels.eq(R);
    auto my_indices = torch::nonzero(my_mask).squeeze(-1);
    int64_t num_local = my_indices.size(0);

    info.atom_counts           = torch::bincount(labels, /*weights=*/{}, /*minlength=*/env.world_size);
    info.partition_indices     = state.atom_vertex_numbers.index_select(0, my_indices);

    info.global_send_recv_matrix = torch::zeros({W, W}, long_opts);
    info.global_send_recv_matrix.index_put_({R, R}, num_local);

    return info;
}

// ---------------------------------------------------------------------------
// Mode 1 & 2: unweighted or user-provided edge_counts
// ---------------------------------------------------------------------------
PartitionMessagePassingInfo SFCPartitioner::partition(
    const MolecularInfo &state, const DistEnvironment &env)
{
    return do_partition(state, env, state.edge_counts, curve_type_);
}

// ---------------------------------------------------------------------------
// Mode 3: auto-compute edge weights via radius_graph_pbc (CUDA required).
// ---------------------------------------------------------------------------
PartitionMessagePassingInfo SFCPartitioner::partition(
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

    // Count edges per atom (dst-based)
    auto dst = edge_idx[1];  // [E]
    auto edge_counts = torch::zeros({N}, long_opts);
    edge_counts.scatter_add_(0, dst, torch::ones({dst.size(0)}, long_opts));

    // Partition using computed weights
    auto info = do_partition(state, env, edge_counts, curve_type_);

    // Store graph for downstream use
    info.edge_index = edge_idx;

    return info;
}

} // namespace distnnp_state
