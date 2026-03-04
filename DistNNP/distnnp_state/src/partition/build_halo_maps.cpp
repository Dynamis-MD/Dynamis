// build_halo_maps.cpp — Shared halo map builder using CUDA radius_graph_pbc + ATen ops.
//
// Algorithm (mirrors baseline halo_exchange.py:build_halo_maps):
//   1. Build full radius graph over all atoms (CUDA cell-list)
//   2. Build needed_by[atom, rank] matrix via 1-hop edge propagation
//   3. Extract send/recv maps, convert to CSR
//   4. Compute interior mask (atoms whose ALL neighbors are local)
//
// All tensor ops are ATen — they run on whatever device the input is on (CUDA or CPU).
// radius_graph_pbc requires CUDA input tensors.

#include "distnnp_state/partitioner.hpp"
#include "radius_pbc/radius_pbc_.h"

namespace distnnp_state
{

static constexpr int64_t HALO_MAX_NUM_NEIGHBORS = 1000;

void AbstractPartitioner::build_halo_maps(
    PartitionMessagePassingInfo &info,
    const MolecularInfo &global_state,
    const DistEnvironment &env,
    const SimulationConfig &config)
{
    auto pos = global_state.atom_positions;
    auto device = pos.device();
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);

    long N = static_cast<long>(global_state.num_atoms);
    long W = static_cast<long>(env.world_size);
    long R = static_cast<long>(env.global_rank);

    float ghost_cutoff = static_cast<float>(config.cutoff + config.ghost_cutoff_buffer);
    auto cell = config.cell.to(device).to(torch::kFloat);
    auto pbc = config.pbc.to(device);

    // --- Step 1: Build full neighbor graph (CUDA cell-list) ---
    auto graph_result = radius_graph_pbc(
        pos.to(torch::kFloat), cell, pbc, ghost_cutoff,
        HALO_MAX_NUM_NEIGHBORS, /*enforce_max_strictly=*/false);
    auto edge_index = std::get<0>(graph_result);  // [2, E]
    auto src = edge_index[0];  // [E]
    auto dst = edge_index[1];  // [E]

    auto partition_id = info.partition_id;

    // --- Step 2: needed_by matrix [N, W] ---
    auto needed_by = torch::zeros({N, W}, bool_opts);

    // Each atom is needed by its own partition
    needed_by.scatter_(1, partition_id.unsqueeze(1), true);

    // 1-hop propagation: for edge (src, dst), dst's owner needs src
    auto owner_dst = partition_id.index_select(0, dst);  // [E]
    needed_by.index_put_({src, owner_dst}, torch::tensor(true, bool_opts));

    // --- Step 3: Send map (CSR) ---
    auto my_mask = partition_id.eq(R);
    auto my_indices = torch::nonzero(my_mask).squeeze(-1);  // [num_local]
    long num_local = my_indices.size(0);

    auto my_needed = needed_by.index_select(0, my_indices);  // [num_local, W]
    // Zero out self-rank column (don't send to self)
    auto my_needed_send = my_needed.clone();
    my_needed_send.select(1, R).fill_(false);

    // All (local_idx, target_rank) pairs where we need to send.
    // NOTE: send_pairs[:,0] are row indices into my_needed_send, i.e. positions
    // 0..num_local-1.  For contiguous partitioners (SimpleSpatial) these equal
    // local array indices.  For non-contiguous partitioners (KDTree/SFC), callers
    // must map through my_indices to get global atom IDs.
    auto send_pairs = torch::nonzero(my_needed_send);  // [total_send, 2]
    long total_send = send_pairs.size(0);

    torch::Tensor send_local_indices_t, send_offsets_t;
    if (total_send > 0) {
        auto send_ranks = send_pairs.select(1, 1).contiguous().to(torch::kLong);
        auto send_atom_idx = send_pairs.select(1, 0).contiguous().to(torch::kLong);

        // Sort by target rank for CSR ordering
        auto sort_order = torch::argsort(send_ranks, /*stable=*/true);
        send_local_indices_t = send_atom_idx.index_select(0, sort_order);
        auto sorted_send_ranks = send_ranks.index_select(0, sort_order);

        // CSR offsets via bincount
        auto send_counts = torch::zeros({W}, long_opts);
        send_counts.scatter_add_(0, sorted_send_ranks,
                                 torch::ones({total_send}, long_opts));
        send_offsets_t = torch::zeros({W + 1}, long_opts);
        send_offsets_t.slice(0, 1, W + 1).copy_(torch::cumsum(send_counts, 0));
    } else {
        send_local_indices_t = torch::empty({0}, long_opts);
        send_offsets_t = torch::zeros({W + 1}, long_opts);
    }

    // --- Step 4: Recv map (CSR) ---
    // Atoms I need from other ranks: needed_by[:, my_rank] AND not my atom
    auto i_need = needed_by.select(1, R).clone();
    i_need.masked_fill_(my_mask, false);

    auto recv_global = torch::nonzero(i_need).squeeze(-1);  // [total_recv]
    long total_recv = recv_global.size(0);

    torch::Tensor recv_ghost_indices_t, recv_offsets_t;
    if (total_recv > 0) {
        auto recv_src_ranks = partition_id.index_select(0, recv_global);
        auto recv_sort = torch::argsort(recv_src_ranks, /*stable=*/true);
        auto sorted_recv_ranks = recv_src_ranks.index_select(0, recv_sort);

        // Ghost slot indices: ghosts are appended after local atoms in the
        // concatenated [local | ghost] array, so slots start at num_local.
        // Ordered by source rank to match recv_offsets CSR.
        recv_ghost_indices_t = torch::arange(num_local, num_local + total_recv, long_opts);

        // CSR offsets
        auto recv_counts = torch::zeros({W}, long_opts);
        recv_counts.scatter_add_(0, sorted_recv_ranks,
                                 torch::ones({total_recv}, long_opts));
        recv_offsets_t = torch::zeros({W + 1}, long_opts);
        recv_offsets_t.slice(0, 1, W + 1).copy_(torch::cumsum(recv_counts, 0));
    } else {
        recv_ghost_indices_t = torch::empty({0}, long_opts);
        recv_offsets_t = torch::zeros({W + 1}, long_opts);
    }

    // --- Step 5: Interior mask ---
    auto dst_is_mine = partition_id.index_select(0, dst).eq(R);
    auto src_of_my_edges = src.masked_select(dst_is_mine);
    auto dst_of_my_edges = dst.masked_select(dst_is_mine);
    auto src_is_ghost = partition_id.index_select(0, src_of_my_edges).ne(R);

    auto has_ghost_neighbor = torch::zeros({N}, bool_opts);
    auto boundary_atoms = dst_of_my_edges.masked_select(src_is_ghost);
    if (boundary_atoms.numel() > 0) {
        has_ghost_neighbor.scatter_(0, boundary_atoms, true);
    }
    auto interior_mask = has_ghost_neighbor.index_select(0, my_indices).logical_not();

    // --- Step 6: Fill info ---
    info.send_offsets = send_offsets_t;
    info.send_local_indices = send_local_indices_t;
    info.recv_offsets = recv_offsets_t;
    info.recv_ghost_indices = recv_ghost_indices_t;
    info.interior_mask = interior_mask;

    // Update send_recv_matrix with halo counts (vectorized, no CPU sync)
    auto send_counts_per_rank = send_offsets_t.slice(0, 1, W + 1) - send_offsets_t.slice(0, 0, W);
    info.global_send_recv_matrix.select(0, R).copy_(send_counts_per_rank);
    // Restore diagonal (local atom count, not halo count)
    info.global_send_recv_matrix.index_put_({R, R}, num_local);
}

} // namespace distnnp_state
