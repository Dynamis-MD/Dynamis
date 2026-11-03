#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <tuple>

// ===== max_num_neighbors (truncation) semantics =====
//
// max_num_neighbors = k = 0 → no truncation: every pair within the cutoff is an edge.
// k > 0 → per target atom (edge_index[1]), keep only its nearest neighbours:
//   enforce_max_strictly = true  : the k nearest (Orb). Exact distance ties at the k-th
//                                  neighbour are all kept.
//   enforce_max_strictly = false : every neighbour with d^2 <= d^2_(k+1) + 0.01 Å^2, where
//                                  d^2_(k+1) is the (k+1)-th smallest (fairchem / UMA).
// Targets with at most k neighbours keep all of them.
//
// Internally the rule is a per-target threshold tau on the squared distance: an edge is
// kept iff d^2 <= tau (tau = +inf when nothing is cut). radius_graph_pbc_select returns tau;
// radius_graph_pbc_ex accepts it back, so a rank can build edges for ghost targets with
// the tau their owner computed from the full neighbourhood.

// Which part of the pipeline radius_graph_pbc_cuda_impl_ex runs.
enum class RadiusGraphStage {
    Full, // edges + per-target counts (+ tau when truncating)
    CountsOnly, // per-target (kept) counts only; no edge buffer, no edge-count sync
    SelectOnly, // per-target tau + kept counts only; no edge buffer, no edge-count sync
};

// ===== CUDA implementations (defined in radius_graph_kernels.cu) =====

// Legacy 3-tuple impl. All atoms are both sources and targets.
// Forwards to _ex with target_indices=nullopt; back-compat for existing callers.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl(at::Tensor pos, at::Tensor cell, at::Tensor pbc, float radius,
                           int64_t max_num_neighbors, bool enforce_max_strictly);

// Extended impl. If `target_indices` is provided, only those atoms are used as targets
// (sources still come from full `pos` + PBC tiling). Edge endpoints in `edge_index` use
// global atom indices regardless. Per-target outputs are in target_indices order (or
// atom-id order if nullopt).
//
// `thresholds` [N_target] float32: caller-supplied tau (skips the per-target selection;
// max_num_neighbors / enforce_max_strictly are then unused). nullopt → computed when
// max_num_neighbors > 0.
//
// Returns (edge_index, cell_shifts, num_neighbors_image, neighbor_counts, thresholds);
// thresholds is undefined when nothing is truncated. CountsOnly / SelectOnly skip the edge
// buffer / prefix-sum / B4 write kernel and return empty placeholders for the first three.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl_ex(at::Tensor pos, at::Tensor cell, at::Tensor pbc, float radius,
                              int64_t max_num_neighbors, bool enforce_max_strictly,
                              c10::optional<at::Tensor> target_indices,
                              c10::optional<at::Tensor> thresholds = c10::nullopt,
                              RadiusGraphStage stage = RadiusGraphStage::Full);

// ===== C++ entry points with input validation (defined in radius_graph.cpp) =====

std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc(at::Tensor pos, // [N, 3] float32 CUDA
                 at::Tensor cell, // [3, 3] float32
                 at::Tensor pbc, // [3] bool
                 float radius, int64_t max_num_neighbors, bool enforce_max_strictly);

// Extended entry point: accepts optional `target_indices` (int64, [N_target]) and
// returns an extra `neighbor_counts[N_target]` (int32) for per-target (kept) degree.
//
// Use cases:
//   - target_indices=nullopt → equivalent to radius_graph_pbc(), plus neighbor_counts return
//   - target_indices=subset  → only those atoms get edges built; useful for distributed
//                              graph build where each rank only computes its own atoms
//   - thresholds=tau         → second pass of a distributed truncated build: keep d^2 <= tau
//                              per target (tau from radius_graph_pbc_select, own or received)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_ex(at::Tensor pos, at::Tensor cell, at::Tensor pbc, float radius,
                    int64_t max_num_neighbors, bool enforce_max_strictly,
                    c10::optional<at::Tensor> target_indices,
                    c10::optional<at::Tensor> thresholds = c10::nullopt);

// First pass of a truncated build: per-target threshold tau [N_target] float32 and kept
// degree [N_target] int32 (target_indices order). Requires max_num_neighbors > 0. No
// edge buffer and no edge-count sync (the cell-list stages still sync for their sizes).
std::tuple<at::Tensor, at::Tensor>
radius_graph_pbc_select(at::Tensor pos, at::Tensor cell, at::Tensor pbc, float radius,
                        int64_t max_num_neighbors, bool enforce_max_strictly,
                        c10::optional<at::Tensor> target_indices);

// Count-only variant: returns just per-target (kept) degrees [N_target] int32.
// Skips edge buffer allocation, prefix-sum, and B4 write kernel. Used by
// distributed Mode A refresh step where only degrees are needed for SFC
// weighted partition; edges are built later by Mode B at radius r.
at::Tensor radius_graph_pbc_counts_ex(at::Tensor pos, at::Tensor cell, at::Tensor pbc, float radius,
                                      c10::optional<at::Tensor> target_indices,
                                      int64_t max_num_neighbors = 0,
                                      bool enforce_max_strictly = false);
