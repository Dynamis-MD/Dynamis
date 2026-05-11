#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <vector>
#include "distnnp_state/distnnp_state.hpp"

namespace distnnp_model
{

// Pre-computed push target for a single peer rank.
// Stored in device memory so the kernel can access it without host interaction.
struct PushTarget
{
    float *remote_ptr;      // IPC-opened pointer to peer's buffer
    int64_t remote_offset;  // atom offset in peer's buffer where our data starts
    int64_t send_start;     // start index in send_local_indices for this peer
    int64_t send_count;     // number of atoms to push to this peer
};

// One-sided push feature exchanger over NVLink (CUDA IPC).
//
// Lifecycle:
//   1. setup()         — allocate x_buffer, export/exchange IPC handles,
//                        open remote pointers, compute remote offsets
//   2. push_forward()  — gather local features, write to peers' ghost regions
//   3. push_backward() — gather ghost gradients, write to peers' grad buffers
//   4. teardown()      — close IPC handles
//
// x_buffer layout: [N_local | ghost_from_R0 | ghost_from_R1 | ... ]
//   The ghost region is contiguous per source rank (CSR ordering from build_halo_maps).
//   Each sender writes one contiguous block into the receiver's ghost region.

class IPCExchanger
{
public:
    IPCExchanger() = default;
    ~IPCExchanger();

    // Allocate a tensor via raw cudaMalloc (bypasses PyTorch caching allocator).
    // Required for CUDA IPC: cudaIpcGetMemHandle needs the base pointer
    // of a cudaMalloc allocation.
    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index);

    // Export CUDA IPC handle for a buffer allocated via allocate_ipc_buffer().
    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer);

    // Open remote IPC handles and compute push targets.
    void setup(
        torch::Tensor x_buffer,
        torch::Tensor grad_recv_buffer,
        const std::vector<std::vector<uint8_t>> &all_x_handles,
        const std::vector<std::vector<uint8_t>> &all_grad_handles,
        torch::Tensor send_offsets,
        torch::Tensor send_local_indices,
        torch::Tensor recv_offsets,
        torch::Tensor all_recv_offsets,   // [W, W+1]
        torch::Tensor all_num_local,      // [W]
        int64_t num_local,
        int64_t feature_dim,
        int64_t rank,
        int64_t world_size);

    // Forward push: gather features from x_buffer at send_local_indices,
    // write directly into peers' x_buffer ghost regions.
    void push_forward(torch::Tensor x_buffer, cudaStream_t stream = 0,
                      int64_t ce_threshold_bytes = 0);

    // Backward push: gather gradients from ghost region,
    // write into peers' grad_recv_buffer.
    void push_backward(torch::Tensor grad_at_ghost, cudaStream_t stream = 0);

    // --- IPC push synchronization (generation counter) ---
    std::vector<uint8_t> create_and_export_sync_handle();
    void setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles);
    void record_push_complete(cudaStream_t stream = 0);
    void wait_for_pushes(cudaStream_t stream = 0);

    void teardown();

    // Accessors
    int64_t feature_dim() const { return feature_dim_; }
    int64_t num_local() const { return num_local_; }
    int64_t total_send() const { return total_send_; }
    int64_t total_recv() const { return total_recv_; }
    bool is_setup() const { return is_setup_; }
    torch::Tensor send_local_indices() const { return send_local_indices_; }

private:
    int64_t rank_ = 0;
    int64_t world_size_ = 0;
    int64_t feature_dim_ = 0;
    int64_t num_local_ = 0;
    int64_t total_send_ = 0;
    int64_t total_recv_ = 0;
    bool is_setup_ = false;

    // CSR maps (device tensors)
    torch::Tensor send_offsets_;        // [W+1]
    torch::Tensor send_local_indices_;  // [total_send]
    torch::Tensor recv_offsets_;        // [W+1]

    // Push targets for forward (write to peers' x_buffer ghost region)
    torch::Tensor fwd_targets_;   // [W] PushTarget, device memory

    // Push targets for backward (write to peers' grad_recv_buffer)
    torch::Tensor bwd_targets_;   // [W] PushTarget, device memory

    // Staging buffer for CE path (local contiguous, [total_send, F])
    torch::Tensor staging_buffer_;

    // Forward push targets (host copy) for CE path cudaMemcpyAsync calls
    std::vector<PushTarget> ce_fwd_targets_;

    // Opened IPC memory handles (for cleanup)
    std::vector<void *> opened_x_ptrs_;
    std::vector<void *> opened_grad_ptrs_;

    // IPC generation counter for one-sided push synchronization.
    // After each push, the producer writes step N to its own counter via
    // cudaStreamWriteValue32 (ordered after the push on the same stream).
    // Consumers wait for the peer's counter >= N via cudaStreamWaitValue32
    // before reading the ghost region. Monotonic — no slot reuse, and
    // cudaStreamWaitValue32 blocks correctly even if the producer has not
    // yet issued its write (unlike cudaStreamWaitEvent on an unrecorded
    // event, which is a silent no-op).
    uint32_t *local_counter_ = nullptr;             // cudaMalloc'd, IPC-shared
    std::vector<CUdeviceptr> remote_counters_;      // peers' counters opened via IPC
    uint32_t call_counter_ = 0;                     // step number, bumped each push
    bool sync_setup_ = false;
};

// High-level feature exchanger interface.
// Wraps IPCExchanger for NVLink-based one-sided push.
class FeatureExchanger
{
public:
    FeatureExchanger() = default;
    ~FeatureExchanger();

    // --- Public interface ---
    // Forward: copies local_features into IPC buffer, pushes to peers,
    // returns [N_local + N_ghost, F] with ghost region populated.
    torch::Tensor feature_push_forward(const torch::Tensor &local_features,
                                       const distnnp_state::PartitionMessagePassingInfo &partition_info);

    // Backward: pushes ghost grads to peers, scatter-adds received grads,
    // returns [N_local, F].
    torch::Tensor feature_push_backward(const torch::Tensor &grad_output,
                                        const torch::IntArrayRef &grad_input_shape,
                                        const distnnp_state::PartitionMessagePassingInfo &partition_info);

    // --- IPC configuration (call from Python before first exchange) ---
    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index)
    {
        return IPCExchanger::allocate_ipc_buffer(rows, cols, device_index);
    }

    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer)
    {
        return ipc_.export_ipc_handle(buffer);
    }

    void setup_ipc(
        torch::Tensor x_buffer,
        torch::Tensor grad_recv_buffer,
        const std::vector<std::vector<uint8_t>> &all_x_handles,
        const std::vector<std::vector<uint8_t>> &all_grad_handles,
        torch::Tensor send_offsets,
        torch::Tensor send_local_indices,
        torch::Tensor recv_offsets,
        torch::Tensor all_recv_offsets,
        torch::Tensor all_num_local,
        int64_t num_local,
        int64_t feature_dim,
        int64_t rank,
        int64_t world_size);

    std::vector<uint8_t> create_and_export_sync_handle() { return ipc_.create_and_export_sync_handle(); }
    void setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles) { ipc_.setup_sync(all_sync_handles); }
    void record_push_complete() { ipc_.record_push_complete(0); }
    void wait_for_pushes() { ipc_.wait_for_pushes(0); }

    void teardown();
    bool is_ipc_setup() const { return ipc_.is_setup(); }
    void set_ce_threshold_bytes(int64_t t) { ce_threshold_bytes_ = t; }
    int64_t ce_threshold_bytes() const { return ce_threshold_bytes_; }

private:
    IPCExchanger ipc_;
    int64_t ce_threshold_bytes_ = 0;

    // References to IPC-allocated buffers (kept alive via torch::Tensor refcount)
    torch::Tensor x_buffer_;           // [N_local + N_ghost, F]
    torch::Tensor grad_recv_buffer_;   // [total_send, F]
};

} // namespace distnnp_model
