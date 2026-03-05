// ipc_exchange_kernels.cu — CUDA kernels for one-sided push feature exchange.
//
// Single-kernel design: one kernel launch pushes to ALL peers simultaneously.
// Each thread handles one (atom, feature_element) pair. A binary search over
// send_offsets determines the target peer rank.

#include "distnnp_state/exchanger.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace distnnp_state
{

// Binary search: find the CSR segment that contains `idx`.
// Returns r such that offsets[r] <= idx < offsets[r+1].
__device__ __forceinline__ int64_t
binary_search_segment(const int64_t *offsets, int64_t num_segments, int64_t idx)
{
    int64_t lo = 0, hi = num_segments - 1;
    while (lo < hi)
    {
        int64_t mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// Forward push kernel: gather from local x_buffer, write to peers' ghost regions.
//
// Grid: 1D over total_send * feature_dim.
// Each thread computes:
//   1. Which atom in the send list (atom_idx = linear / feature_dim)
//   2. Which feature element (feat_idx = linear % feature_dim)
//   3. Binary search send_offsets to find target peer rank
//   4. Read from x[send_local_indices[atom_idx], feat_idx]
//   5. Write to target's remote_ptr[(remote_offset + local_atom_in_segment) * feature_dim + feat_idx]
__global__ void gather_and_push_kernel(
    const float *__restrict__ x,               // [N_local + N_ghost, F]
    const int64_t *__restrict__ send_offsets,   // [W+1] CSR
    const int64_t *__restrict__ send_indices,   // [total_send]
    const PushTarget *__restrict__ targets,     // [W]
    int64_t total_send,
    int64_t feature_dim,
    int64_t world_size)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_elements = total_send * feature_dim;

    for (int64_t i = tid; i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t atom_idx = i / feature_dim;
        int64_t feat_idx = i % feature_dim;

        // Find which peer this atom belongs to via binary search on send_offsets
        int64_t peer = binary_search_segment(send_offsets, world_size, atom_idx);

        const PushTarget &tgt = targets[peer];

        // Local atom in this peer's segment
        int64_t local_in_segment = atom_idx - tgt.send_start;

        // Gather from local x
        int64_t src_atom = send_indices[atom_idx];
        float val = x[src_atom * feature_dim + feat_idx];

        // Write to remote ghost region
        int64_t dst_offset = (tgt.remote_offset + local_in_segment) * feature_dim + feat_idx;
        tgt.remote_ptr[dst_offset] = val;
    }

    // Make NVLink writes visible to other GPU contexts (required before IPC event record)
    __threadfence_system();
}

// Backward push kernel: gather gradients from ghost region, write to peers' grad_recv_buffer.
//
// In backward, each rank gathers gradients at its own ghost positions (contiguous)
// and pushes them to the owning rank's grad_recv_buffer.
//
// The recv_offsets CSR on this rank tells us where each peer's ghost atoms are:
//   ghost atoms from peer r: x[num_local + recv_offsets[r] : num_local + recv_offsets[r+1]]
//
// We push these to peer r's grad_recv_buffer at the offset where peer r
// expects to receive gradients from us.
__global__ void gather_ghost_grad_and_push_kernel(
    const float *__restrict__ grad,            // [N_local + N_ghost, F] full gradient
    const int64_t *__restrict__ recv_offsets,   // [W+1] CSR (this rank's)
    const PushTarget *__restrict__ targets,     // [W] backward push targets
    int64_t num_local,
    int64_t total_recv,
    int64_t feature_dim,
    int64_t world_size)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_elements = total_recv * feature_dim;

    for (int64_t i = tid; i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t ghost_idx = i / feature_dim;  // index within ghost region [0, total_recv)
        int64_t feat_idx = i % feature_dim;

        // Find which peer owns this ghost atom
        int64_t peer = binary_search_segment(recv_offsets, world_size, ghost_idx);

        const PushTarget &tgt = targets[peer];

        // Position within this peer's segment
        int64_t local_in_segment = ghost_idx - recv_offsets[peer];

        // Read gradient from ghost region (contiguous)
        float val = grad[(num_local + ghost_idx) * feature_dim + feat_idx];

        // Write to peer's grad_recv_buffer
        int64_t dst_offset = (tgt.remote_offset + local_in_segment) * feature_dim + feat_idx;
        tgt.remote_ptr[dst_offset] = val;
    }

    __threadfence_system();
}

// Gather-to-staging kernel: gather from scattered send_local_indices into
// a contiguous local buffer. Writes are coalesced (sequential output).
// Used as stage 1 of the CE path.
__global__ void gather_to_staging_kernel(
    const float *__restrict__ x,               // [N_local + N_ghost, F]
    const int64_t *__restrict__ send_indices,   // [total_send]
    float *__restrict__ staging,                // [total_send, F] contiguous output
    int64_t total_send,
    int64_t feature_dim)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_elements = total_send * feature_dim;

    for (int64_t i = tid; i < total_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t atom = i / feature_dim;
        int64_t feat = i % feature_dim;
        int64_t src_atom = send_indices[atom];
        staging[atom * feature_dim + feat] = x[src_atom * feature_dim + feat];
    }
}

// Launch helpers

void launch_gather_and_push(
    torch::Tensor x_buffer,
    torch::Tensor send_offsets,
    torch::Tensor send_local_indices,
    torch::Tensor fwd_targets,
    int64_t total_send,
    int64_t feature_dim,
    int64_t world_size,
    cudaStream_t stream)
{
    if (total_send == 0)
        return;

    int64_t total_elements = total_send * feature_dim;
    int threads = 256;
    int blocks = std::min(static_cast<int64_t>((total_elements + threads - 1) / threads),
                          static_cast<int64_t>(65535));

    gather_and_push_kernel<<<blocks, threads, 0, stream>>>(
        x_buffer.data_ptr<float>(),
        send_offsets.data_ptr<int64_t>(),
        send_local_indices.data_ptr<int64_t>(),
        reinterpret_cast<const PushTarget *>(fwd_targets.data_ptr()),
        total_send,
        feature_dim,
        world_size);
}

void launch_gather_ghost_grad_and_push(
    torch::Tensor grad,
    torch::Tensor recv_offsets,
    torch::Tensor bwd_targets,
    int64_t num_local,
    int64_t total_recv,
    int64_t feature_dim,
    int64_t world_size,
    cudaStream_t stream)
{
    if (total_recv == 0)
        return;

    int64_t total_elements = total_recv * feature_dim;
    int threads = 256;
    int blocks = std::min(static_cast<int64_t>((total_elements + threads - 1) / threads),
                          static_cast<int64_t>(65535));

    gather_ghost_grad_and_push_kernel<<<blocks, threads, 0, stream>>>(
        grad.data_ptr<float>(),
        recv_offsets.data_ptr<int64_t>(),
        reinterpret_cast<const PushTarget *>(bwd_targets.data_ptr()),
        num_local,
        total_recv,
        feature_dim,
        world_size);
}

void launch_gather_to_staging(
    torch::Tensor x_buffer,
    torch::Tensor send_local_indices,
    torch::Tensor staging,
    int64_t total_send,
    int64_t feature_dim,
    cudaStream_t stream)
{
    if (total_send == 0)
        return;

    int64_t total_elements = total_send * feature_dim;
    int threads = 256;
    int blocks = std::min(static_cast<int64_t>((total_elements + threads - 1) / threads),
                          static_cast<int64_t>(65535));

    gather_to_staging_kernel<<<blocks, threads, 0, stream>>>(
        x_buffer.data_ptr<float>(),
        send_local_indices.data_ptr<int64_t>(),
        staging.data_ptr<float>(),
        total_send,
        feature_dim);
}

} // namespace distnnp_state
