#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace distnnp_state
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
    // This is REQUIRED for CUDA IPC: cudaIpcGetMemHandle needs the base pointer
    // of a cudaMalloc allocation. PyTorch's caching allocator sub-allocates from
    // larger blocks, so tensor.data_ptr() is NOT a cudaMalloc base pointer.
    //
    // Returns a float tensor of shape [rows, cols] on the specified CUDA device.
    // The tensor's destructor calls cudaFree.
    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index);

    // Export CUDA IPC handle for a buffer allocated via allocate_ipc_buffer().
    // Returns 64 bytes (sizeof cudaIpcMemHandle_t).
    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer);

    // Open remote IPC handles and compute push targets.
    //   all_x_handles:      [W] x 64 bytes — IPC handles for all ranks' x_buffers
    //   all_grad_handles:    [W] x 64 bytes — IPC handles for all ranks' grad_recv_buffers
    //   send_offsets:        [W+1] CSR
    //   send_local_indices:  [total_send]
    //   recv_offsets:        [W+1] CSR (this rank's)
    //   all_recv_offsets:    [W, W+1] — recv_offsets from every rank (for remote offset calc)
    //   all_num_local:       [W] — N_local for every rank
    //   num_local:           this rank's N_local
    //   feature_dim:         F
    //   rank, world_size
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
    //
    // Strategy based on ce_threshold_bytes:
    //   - Default 0: SM kernel push (all peers, single kernel, low latency for small data)
    //   - > 0: if any peer's data >= threshold, switch to CE mode for ALL peers
    //          (gather to staging + per-peer cudaMemcpyAsync via Copy Engine, high BW)
    void push_forward(torch::Tensor x_buffer, cudaStream_t stream = 0,
                      int64_t ce_threshold_bytes = 0);

    // Backward push: gather gradients from x_buffer's ghost region,
    // write into peers' grad_recv_buffer.
    // After sync, grad_recv_buffer is populated with incoming gradients.
    void push_backward(torch::Tensor grad_at_ghost, cudaStream_t stream = 0);

    // --- IPC Event synchronization ---

    // Create a local interprocess event and return its IPC handle (64 bytes).
    // The event is kept alive internally. Call once per rank before setup_events().
    std::vector<uint8_t> create_and_export_event();

    // Open remote events from all ranks for stream-level sync.
    //   all_event_handles:   [W] x 64 bytes, event handles from all ranks
    //   (all_event_handles[rank] is this rank's own handle, will be skipped)
    void setup_events(const std::vector<std::vector<uint8_t>> &all_event_handles);

    // Record this rank's completion event on the given stream.
    // Call after push_forward/push_backward kernel launch.
    void record_push_complete(cudaStream_t stream = 0);

    // Wait for all remote ranks' push to complete before proceeding on stream.
    // Call before reading ghost region.
    void wait_for_pushes(cudaStream_t stream = 0);

    void teardown();

    // Accessors
    int64_t feature_dim() const { return feature_dim_; }
    int64_t num_local() const { return num_local_; }
    int64_t total_send() const { return total_send_; }
    int64_t total_recv() const { return total_recv_; }
    bool is_setup() const { return is_setup_; }

    // Device tensor of send_local_indices (for autograd to save)
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

    // IPC events for stream-level synchronization
    cudaEvent_t local_event_ = nullptr;           // this rank's event
    std::vector<cudaEvent_t> remote_events_;      // opened events from peers
    bool events_setup_ = false;
};

} // namespace distnnp_state
