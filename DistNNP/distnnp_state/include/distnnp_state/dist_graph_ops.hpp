#pragma once

#include <ATen/ATen.h>
#include <tuple>

// Distributed build_graph helper ops (v5), implemented in CUDA and exposed to
// Python. These replace the ATen-heavy hot-path steps of DistGraphBuilder
// Mode A: edge filtering (step 8) and halo-info extraction (step 7).

// ===== CUDA implementations (dist_graph_ops_kernels.cu) =====

// Filter edges built at r+skin down to the true cutoff r. For each edge
// (src -> dst) the integer PBC cell_shift is applied to src; the edge is kept
// iff ||pos[src] + shift @ cell - pos[dst]||^2 <= r^2. Order-preserving.
//
// Returns (edges_kept [2, E'], cell_shifts_kept [E', 3]).
std::tuple<at::Tensor, at::Tensor> filter_edges_to_radius_cuda_impl(
    at::Tensor edges,       // [2, E] int64
    at::Tensor cell_shifts, // [E, 3] float32 (shift applied to src)
    at::Tensor pos,         // [N, 3] float32 (PBC-wrapped)
    at::Tensor cell,        // [3, 3] float32 row-major (lattice vectors)
    double radius);

// Extract halo send/recv CSR + interior mask from per-rank edges (dst is
// guaranteed to be in my_outer) using geometric symmetry, matching
// main::build_halo_maps conventions: both CSRs are grouped by peer rank and,
// within each peer segment, sorted by global atom id (so sender/receiver align
// with no NCCL exchange of plans). v5 step 7.
//
// Returns (all CUDA):
//   send_offsets        [W+1]        int64
//   send_local_indices  [total_send] int64  (positions into my_outer_atoms)
//   recv_offsets        [W+1]        int64
//   recv_ghost_indices  [total_recv] int64  (arange(num_local,
//   num_local+total_recv)) interior_mask       [num_local]  bool   (my atoms
//   with no remote-src edge) recv_atom_global    [total_recv] int64  (halo recv
//   global ids, peer-then-id order)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
extract_halo_info_cuda_impl(
    at::Tensor edges,          // [2, E] int64 (dst in my_outer)
    at::Tensor pid_outer,      // [N] int64 partition id
    at::Tensor my_outer_atoms, // [num_local] int64 global ids
    int64_t my_rank, int64_t world_size, int64_t N);

// ===== C++ entry points with input validation (dist_graph_ops.cpp) =====

std::tuple<at::Tensor, at::Tensor>
filter_edges_to_radius(at::Tensor edges, at::Tensor cell_shifts, at::Tensor pos,
                       at::Tensor cell, double radius);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
extract_halo_info(at::Tensor edges, at::Tensor pid_outer,
                  at::Tensor my_outer_atoms, int64_t my_rank,
                  int64_t world_size, int64_t N);
