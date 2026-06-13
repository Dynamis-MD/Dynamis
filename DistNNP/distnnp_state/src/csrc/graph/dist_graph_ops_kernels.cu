// dist_graph_ops_kernels.cu — CUDA hot-path ops for DistGraphBuilder Mode A.

#include "distnnp_state/dist_graph_ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int BLOCK = 256;

// One fused pass per edge: gather src/dst, apply integer PBC shift to src via
// shift @ cell, compute squared distance, compare to r^2. Replaces the ATen
// chain (index_select + matmul + sub + square + sum + le) with a single kernel
// and no [E,3] temporaries.
__global__ void edge_keep_mask_kernel(
    const int64_t* __restrict__ src,    // [E]
    const int64_t* __restrict__ dst,    // [E]
    const float* __restrict__ shifts,   // [E, 3]
    const float* __restrict__ pos,      // [N, 3]
    const float* __restrict__ cell,     // [3, 3] row-major
    float r2,
    int64_t E,
    bool* __restrict__ keep)            // [E] out
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    const int64_t s = src[e];
    const int64_t d = dst[e];

    float sx = pos[3 * s + 0], sy = pos[3 * s + 1], sz = pos[3 * s + 2];
    const float dx = pos[3 * d + 0], dy = pos[3 * d + 1], dz = pos[3 * d + 2];
    const float h0 = shifts[3 * e + 0], h1 = shifts[3 * e + 1], h2 = shifts[3 * e + 2];

    // src image = pos[s] + shift @ cell ; cell row-major: cell[i*3 + j]
    sx += h0 * cell[0] + h1 * cell[3] + h2 * cell[6];
    sy += h0 * cell[1] + h1 * cell[4] + h2 * cell[7];
    sz += h0 * cell[2] + h1 * cell[5] + h2 * cell[8];

    const float ax = sx - dx, ay = sy - dy, az = sz - dz;
    keep[e] = (ax * ax + ay * ay + az * az) <= r2;
}

// Fused gather + boundary test: peer[e] = pid[src[e]]; keep iff peer != my_rank.
__global__ void boundary_peer_kernel(
    const int64_t* __restrict__ src,   // [E]
    const int64_t* __restrict__ pid,   // [N]
    int64_t my_rank,
    int64_t E,
    int64_t* __restrict__ peer_out,    // [E]
    bool* __restrict__ keep_out)       // [E]
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int64_t p = pid[src[e]];
    peer_out[e] = p;
    keep_out[e] = (p != my_rank);
}
} // namespace

std::tuple<at::Tensor, at::Tensor>
filter_edges_to_radius_cuda_impl(
    at::Tensor edges,
    at::Tensor cell_shifts,
    at::Tensor pos,
    at::Tensor cell,
    double radius)
{
    at::cuda::CUDAGuard device_guard(pos.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t E = edges.size(1);
    if (E == 0) {
        return std::make_tuple(edges, cell_shifts);
    }

    // Row views of the [2, E] contiguous edge index are contiguous [E] slices.
    auto src = edges.select(0, 0).contiguous();
    auto dst = edges.select(0, 1).contiguous();
    auto shifts_f = cell_shifts.to(at::kFloat).contiguous();  // compute in f32
    auto keep = at::empty({E}, edges.options().dtype(at::kBool));

    const float r2 = static_cast<float>(radius * radius);
    const int64_t grid = (E + BLOCK - 1) / BLOCK;
    edge_keep_mask_kernel<<<grid, BLOCK, 0, stream>>>(
        src.data_ptr<int64_t>(),
        dst.data_ptr<int64_t>(),
        shifts_f.data_ptr<float>(),
        pos.data_ptr<float>(),
        cell.data_ptr<float>(),
        r2, E,
        keep.data_ptr<bool>());

    // Compaction. nonzero() returns ascending indices -> order-preserving,
    // matching the original boolean-mask filter exactly. Compact the ORIGINAL
    // cell_shifts so dtype is preserved for downstream consumers.
    auto idx = keep.nonzero().squeeze(1);
    auto edges_kept = edges.index_select(1, idx).contiguous();
    auto shifts_kept = cell_shifts.index_select(0, idx).contiguous();
    return std::make_tuple(edges_kept, shifts_kept);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
extract_halo_info_cuda_impl(
    at::Tensor edges,
    at::Tensor pid_outer,
    at::Tensor my_outer_atoms,
    int64_t my_rank,
    int64_t world_size,
    int64_t N)
{
    at::cuda::CUDAGuard device_guard(edges.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t E = edges.size(1);
    const int64_t num_local = my_outer_atoms.size(0);
    const auto long_opts = at::TensorOptions().dtype(at::kLong).device(edges.device());
    const auto bool_opts = at::TensorOptions().dtype(at::kBool).device(edges.device());

    auto src = edges.select(0, 0).contiguous();   // [E] int64
    auto dst = edges.select(0, 1).contiguous();   // [E] int64 (in my_outer)

    // ---- boundary edges: src owned by a remote rank ----
    at::Tensor b_src, b_dst, b_peer;
    if (E > 0) {
        auto peer = at::empty({E}, long_opts);
        auto keep = at::empty({E}, bool_opts);
        const int64_t grid = (E + BLOCK - 1) / BLOCK;
        boundary_peer_kernel<<<grid, BLOCK, 0, stream>>>(
            src.data_ptr<int64_t>(), pid_outer.data_ptr<int64_t>(),
            my_rank, E, peer.data_ptr<int64_t>(), keep.data_ptr<bool>());
        auto b_idx = keep.nonzero().squeeze(1);
        b_src = src.index_select(0, b_idx);
        b_dst = dst.index_select(0, b_idx);
        b_peer = peer.index_select(0, b_idx);
    } else {
        b_src = at::empty({0}, long_opts);
        b_dst = at::empty({0}, long_opts);
        b_peer = at::empty({0}, long_opts);
    }

    // ---- global -> local lookup for my_outer ----
    auto g2l = at::full({N}, -1, long_opts);
    g2l.index_copy_(0, my_outer_atoms, at::arange(num_local, long_opts));

    // ---- RECV CSR: unique (peer, b_src), grouped by peer, sorted by atom id ----
    auto recv_key = b_peer.mul(N).add(b_src);
    auto recv_unique = std::get<0>(at::_unique2(recv_key, /*sorted=*/true,
                                                /*return_inverse=*/false,
                                                /*return_counts=*/false));
    auto recv_peer = recv_unique.floor_divide(N);
    auto recv_atom = recv_unique.remainder(N);
    const int64_t total_recv = recv_atom.size(0);
    auto recv_counts = at::zeros({world_size}, long_opts);
    if (total_recv > 0) {
        recv_counts.scatter_add_(0, recv_peer, at::ones({total_recv}, long_opts));
    }
    auto recv_offsets = at::zeros({world_size + 1}, long_opts);
    recv_offsets.narrow(0, 1, world_size).copy_(recv_counts.cumsum(0));
    auto recv_ghost = at::arange(num_local, num_local + total_recv, long_opts);

    // ---- SEND CSR: unique (peer, b_dst), grouped by peer, sorted by atom id ----
    auto send_key = b_peer.mul(N).add(b_dst);
    auto send_unique = std::get<0>(at::_unique2(send_key, /*sorted=*/true,
                                                /*return_inverse=*/false,
                                                /*return_counts=*/false));
    auto send_peer = send_unique.floor_divide(N);
    auto send_atom_global = send_unique.remainder(N);
    auto send_local_indices = g2l.index_select(0, send_atom_global);
    const int64_t total_send = send_atom_global.size(0);
    auto send_counts = at::zeros({world_size}, long_opts);
    if (total_send > 0) {
        send_counts.scatter_add_(0, send_peer, at::ones({total_send}, long_opts));
    }
    auto send_offsets = at::zeros({world_size + 1}, long_opts);
    send_offsets.narrow(0, 1, world_size).copy_(send_counts.cumsum(0));

    // ---- interior mask: my atoms with no remote-src edge ----
    auto has_remote = at::zeros({N}, bool_opts);
    if (b_dst.size(0) > 0) {
        has_remote.index_fill_(0, b_dst, true);
    }
    auto interior_mask = has_remote.index_select(0, my_outer_atoms).logical_not();

    return std::make_tuple(send_offsets, send_local_indices, recv_offsets,
                           recv_ghost, interior_mask, recv_atom);
}
