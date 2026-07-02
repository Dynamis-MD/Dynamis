#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <tuple>

// ===== CUDA implementations (defined in radius_graph_kernels.cu) =====

// Legacy 3-tuple impl. All atoms are both sources and targets.
// Forwards to _ex with target_indices=nullopt; back-compat for existing
// callers.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl(at::Tensor pos, at::Tensor cell, at::Tensor pbc,
                           float radius, int64_t max_num_neighbors,
                           bool enforce_max_strictly);

// Extended 4-tuple impl. If `target_indices` is provided, only those atoms
// are used as targets (sources still come from full `pos` + PBC tiling).
// Edge endpoints in `edge_index` use global atom indices regardless.
// `neighbor_counts[N_target]` is in target_indices order (or atom-id order if
// nullopt).
//
// `counts_only`: if true, skip edge buffer allocation / prefix-sum / B4 write
//                kernel and return only neighbor_counts (edge_index,
//                cell_shifts, num_neighbors_image will be empty placeholders).
//                ~50% faster when caller only needs per-target degrees (used by
//                Mode A).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_cuda_impl_ex(at::Tensor pos, at::Tensor cell, at::Tensor pbc,
                              float radius, int64_t max_num_neighbors,
                              bool enforce_max_strictly,
                              c10::optional<at::Tensor> target_indices,
                              bool counts_only = false);

// ===== C++ entry points with input validation (defined in radius_graph.cpp)
// =====

std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc(at::Tensor pos,  // [N, 3] float32 CUDA
                 at::Tensor cell, // [3, 3] float32
                 at::Tensor pbc,  // [3] bool
                 float radius, int64_t max_num_neighbors,
                 bool enforce_max_strictly);

// Extended entry point: accepts optional `target_indices` (int64, [N_target])
// and returns an extra `neighbor_counts[N_target]` (int32) for per-target
// degree.
//
// Use cases:
//   - target_indices=nullopt → equivalent to radius_graph_pbc(), plus
//   neighbor_counts return
//   - target_indices=subset  → only those atoms get edges built; useful for
//   distributed
//                              graph build where each rank only computes its
//                              own atoms
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc_ex(at::Tensor pos, at::Tensor cell, at::Tensor pbc,
                    float radius, int64_t max_num_neighbors,
                    bool enforce_max_strictly,
                    c10::optional<at::Tensor> target_indices);

// Count-only variant: returns just per-target neighbor degrees [N_target]
// int32. Skips edge buffer allocation, prefix-sum, and B4 write kernel. Used by
// distributed Mode A refresh step where only degrees are needed for SFC
// weighted partition; edges are built later by Mode B at radius r.
at::Tensor radius_graph_pbc_counts_ex(at::Tensor pos, at::Tensor cell,
                                      at::Tensor pbc, float radius,
                                      c10::optional<at::Tensor> target_indices);
