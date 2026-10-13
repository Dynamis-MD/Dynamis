#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <vector>
#include "dynamis_state/dynamis_state.hpp"

namespace distnnp_model
{

// Pre-computed push target for a single peer rank.
// Stored in device memory so the kernel can access it without host interaction.
struct PushTarget
{
    float *remote_ptr;      // IPC-opened or Fabric-mapped pointer to peer's buffer
    int64_t remote_offset;  // atom offset in peer's buffer where our data starts
    int64_t send_start;     // start index in send_local_indices for this peer
    int64_t send_count;     // number of atoms to push to this peer
};

// Check if fabric handles (CU_MEM_HANDLE_TYPE_FABRIC / multi-node NVLink) are supported.
bool supports_fabric_handles(int device_id = -1);

// One-sided push feature exchanger over NVLink (CUDA IPC / Fabric Handles).
//
// Lifecycle:
//   1. setup()         — allocate x_buffer, export/exchange IPC/Fabric handles,
//                        open/map remote pointers, compute remote offsets
//   2. push_forward()  — gather local features, write to peers' ghost regions
//   3. push_backward() — gather ghost gradients, write to peers' grad buffers
//   4. teardown()      — close IPC handles or unmap/free Fabric VMM allocations
//
// x_buffer layout: [N_local | ghost_from_R0 | ghost_from_R1 | ... ]
//   The ghost region is contiguous per source rank (CSR ordering from build_halo_maps).
//   Each sender writes one contiguous block into the receiver's ghost region.

class IPCExchanger
{
public:
    IPCExchanger() = default;
    ~IPCExchanger();

    // Check if fabric handles are supported on the device.
    static bool supports_fabric_handles(int device_id = -1);

    // Allocate a tensor via Fabric VMM (CUDA 12+) or raw cudaMalloc (bypasses PyTorch caching allocator).
    // Required for CUDA IPC / Fabric handles.
    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index);

    // Export CUDA IPC / Fabric handle for a buffer allocated via allocate_ipc_buffer().
    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer);

    // Open remote IPC / Fabric handles and compute push targets.
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

    // --- Drift-watchdog flag, piggybacked on the push-sync handshake ---
    // Every rank's sync word is kSyncWords uint32: [counter | flag[0] | flag[1] | pad].
    // When flags are set, record_push_complete() copies *local_flag into flag[g % 2]
    // BEFORE writing the counter (same ordering, and so the same visibility, as the
    // pushed ghost data), and wait_for_pushes() ORs flag[g % 2] (bitwise) over all ranks into
    // *global_flag. Two slots because a peer may already be one generation ahead;
    // it cannot be two ahead, since that needs this rank's counter for g + 1.
    // Both tensors are int32 [1] on this device. Unset (the default) = feature off.
    static constexpr int kSyncWords = 4;
    void set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag);
    void clear_watchdog_flags();

    void teardown();

    // Accessors
    int64_t feature_dim() const { return feature_dim_; }
    int64_t num_local() const { return num_local_; }
    int64_t total_send() const { return total_send_; }
    int64_t total_recv() const { return total_recv_; }
    bool is_setup() const { return is_setup_; }
    bool use_fabric() const { return use_fabric_; }
    torch::Tensor send_local_indices() const { return send_local_indices_; }

private:
    struct VmmMapping
    {
        CUdeviceptr vaddr = 0;
        size_t padded_size = 0;
        CUmemGenericAllocationHandle handle = 0;
    };

    int64_t rank_ = 0;
    int64_t world_size_ = 0;
    int64_t feature_dim_ = 0;
    int64_t num_local_ = 0;
    int64_t total_send_ = 0;
    int64_t total_recv_ = 0;
    bool is_setup_ = false;
    bool use_fabric_ = false;

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

    // Opened IPC / Fabric memory handles (for cleanup and targeting)
    std::vector<void *> opened_x_ptrs_;
    std::vector<void *> opened_grad_ptrs_;

    // Fabric VMM mappings on remote rank allocations
    std::vector<VmmMapping> remote_x_mappings_;
    std::vector<VmmMapping> remote_grad_mappings_;

    // IPC / Fabric generation counter for one-sided push synchronization.
    // After each push, the producer writes step N to its own counter via
    // cuStreamWriteValue32 (ordered after the push on the same stream).
    // Consumers wait for the peer's counter >= N via cuStreamWaitValue32
    // before reading the ghost region. Monotonic — no slot reuse.
    uint32_t *local_counter_ = nullptr;
    std::vector<CUdeviceptr> remote_counters_;      // peers' counters opened via IPC / Fabric
    std::vector<VmmMapping> remote_sync_mappings_;  // Fabric mappings for sync counters
    uint32_t call_counter_ = 0;                     // step number, bumped each push
    bool sync_setup_ = false;
    bool sync_use_fabric_ = false;

    // Local sync counter Fabric VMM resources (if sync_use_fabric_)
    CUdeviceptr local_counter_vaddr_ = 0;
    size_t local_counter_padded_size_ = 0;
    CUmemGenericAllocationHandle local_counter_handle_ = 0;

    // Drift-watchdog piggyback (see set_watchdog_flags). Owned by the caller.
    torch::Tensor watchdog_local_;   // int32 [1]
    torch::Tensor watchdog_global_;  // int32 [1]
    torch::Tensor sync_word_ptrs_;   // int64 [W] on device: base of every rank's sync word
    void build_sync_word_ptrs();
};

// High-level feature exchanger interface.
// Wraps IPCExchanger for NVLink-based one-sided push.
class FeatureExchanger
{
public:
    FeatureExchanger() = default;
    ~FeatureExchanger();

    // Check if fabric handles are supported on the device.
    static bool supports_fabric_handles(int device_id = -1)
    {
        return IPCExchanger::supports_fabric_handles(device_id);
    }

    // --- Public interface ---
    // Forward: copies local_features into IPC buffer, pushes to peers,
    // returns [N_local + N_ghost, F] with ghost region populated.
    torch::Tensor feature_push_forward(const torch::Tensor &local_features,
                                       const dynamis_state::PartitionMessagePassingInfo &partition_info);

    // Backward: pushes ghost grads to peers, scatter-adds received grads,
    // returns [N_local, F].
    torch::Tensor feature_push_backward(const torch::Tensor &grad_output,
                                        const torch::IntArrayRef &grad_input_shape,
                                        const dynamis_state::PartitionMessagePassingInfo &partition_info);

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
    void set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag)
    {
        ipc_.set_watchdog_flags(local_flag, global_flag);
    }
    void clear_watchdog_flags() { ipc_.clear_watchdog_flags(); }

    void teardown();
    bool is_ipc_setup() const { return ipc_.is_setup(); }
    bool use_fabric() const { return ipc_.use_fabric(); }
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

