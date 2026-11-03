#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "dynamis_state/dynamis_state.hpp"

namespace dynamis_model {

// Pre-computed push target for a single peer rank.
// Stored in device memory so the kernel can access it without host interaction.
struct PushTarget {
    float *remote_ptr;      // IPC-opened or Fabric-mapped pointer to peer's buffer (slot 0 base)
    int64_t slot_stride;    // float offset between slot 0 and slot 1 on peer rank
    int64_t remote_offset;  // atom offset in peer's slot where our data starts
    int64_t send_start;     // start index in send_local_indices for this peer
    int64_t send_count;     // number of atoms to push to this peer
};

// Check if fabric handles (CU_MEM_HANDLE_TYPE_FABRIC / multi-node NVLink) are supported.
bool supports_fabric_handles(int device_id = -1);

// Abstract C++ communication transport backend interface.
// Concrete implementations (IPCExchanger, future LibfabricBackend / NCCLBackend)
// manage transport-registered buffers, opaque connection metadata, and push execution
// for any runtime feature_dim <= max_feature_dim.
class CommBackend
{
public:
    virtual ~CommBackend() = default;

    // Step 1 of setup: allocate transport-registered buffers (2 slots of width max_feature_dim)
    // and return a single serialized metadata byte blob to be all-gathered across ranks.
    virtual std::vector<uint8_t>
    prepare_local_metadata(torch::Tensor &x_buffer,
                           torch::Tensor &grad_recv_buffer,
                           const dynamis_state::PartitionMessagePassingInfo &partition_info,
                           int64_t max_feature_dim,
                           int64_t rank,
                           int64_t world_size,
                           int64_t device_index) = 0;

    // Step 2 of setup: connect to peers using the all-gathered metadata blobs from all ranks.
    virtual void connect(const std::vector<std::vector<uint8_t>> &all_rank_metadata) = 0;

    // Return the slot index (0 or 1) that will be used by the next push operation.
    virtual int64_t next_slot() const = 0;

    // Execute forward push + stream synchronization for runtime width `feature_dim <= max_feature_dim`.
    virtual void push_forward_slot(torch::Tensor x_slot_view,
                                   int64_t feature_dim,
                                   int64_t slot,
                                   cudaStream_t stream = 0,
                                   int64_t ce_threshold_bytes = 0) = 0;

    // Execute backward push + stream synchronization for runtime width `feature_dim <= max_feature_dim`.
    virtual void push_backward_slot(torch::Tensor grad_output,
                                    int64_t feature_dim,
                                    int64_t slot,
                                    cudaStream_t stream = 0) = 0;

    // Optional drift-watchdog piggyback (no-op for backends that do not piggyback on sync words).
    virtual void set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag) {}
    virtual void clear_watchdog_flags() {}

    virtual void teardown() = 0;
    virtual bool is_setup() const = 0;
    virtual bool use_fabric() const { return false; }
};

// One-sided push communication backend over NVLink (CUDA IPC / Fabric Handles).
//
// Supports both:
//   - The unified CommBackend lifecycle (`prepare_local_metadata` -> single all_gather -> `connect`)
//     with 2-slot double-buffered allocations supporting any `feature_dim <= max_feature_dim`.
//   - Direct low-level IPC calls (`allocate_ipc_buffer`, `export_ipc_handle`, `setup`, `push_forward`)
//     for standalone microbenchmarks and unit tests.
class IPCExchanger : public CommBackend
{
public:
    IPCExchanger() = default;
    ~IPCExchanger() override;

    // Check if fabric handles are supported on the device.
    static bool supports_fabric_handles(int device_id = -1);

    // Allocate a tensor via Fabric VMM (CUDA 12+) or raw cudaMalloc (bypasses PyTorch caching
    // allocator). Required for CUDA IPC / Fabric handles.
    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index);

    // Export CUDA IPC / Fabric handle for a buffer allocated via allocate_ipc_buffer().
    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer);

    // --- CommBackend interface ---
    std::vector<uint8_t>
    prepare_local_metadata(torch::Tensor &x_buffer,
                           torch::Tensor &grad_recv_buffer,
                           const dynamis_state::PartitionMessagePassingInfo &partition_info,
                           int64_t max_feature_dim,
                           int64_t rank,
                           int64_t world_size,
                           int64_t device_index) override;

    void connect(const std::vector<std::vector<uint8_t>> &all_rank_metadata) override;

    int64_t next_slot() const override { return static_cast<int64_t>((call_counter_ + 1) & 1u); }

    void push_forward_slot(torch::Tensor x_slot_view,
                           int64_t feature_dim,
                           int64_t slot,
                           cudaStream_t stream = 0,
                           int64_t ce_threshold_bytes = 0) override;

    void push_backward_slot(torch::Tensor grad_output,
                            int64_t feature_dim,
                            int64_t slot,
                            cudaStream_t stream = 0) override;

    // --- Direct / low-level IPC interface ---
    void setup(torch::Tensor x_buffer, torch::Tensor grad_recv_buffer,
               const std::vector<std::vector<uint8_t>> &all_x_handles,
               const std::vector<std::vector<uint8_t>> &all_grad_handles,
               torch::Tensor send_offsets, torch::Tensor send_local_indices,
               torch::Tensor recv_offsets,
               torch::Tensor all_recv_offsets, // [W, W+1]
               torch::Tensor all_num_local,    // [W]
               int64_t num_local, int64_t feature_dim, int64_t rank, int64_t world_size);

    void push_forward(torch::Tensor x_buffer, cudaStream_t stream = 0,
                      int64_t ce_threshold_bytes = 0);

    void push_backward(torch::Tensor grad_at_ghost, cudaStream_t stream = 0);

    // --- IPC push synchronization (generation counter) ---
    std::vector<uint8_t> create_and_export_sync_handle();
    void setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles);
    void record_push_complete(cudaStream_t stream = 0);
    void wait_for_pushes(cudaStream_t stream = 0);

    // --- Drift-watchdog flag, piggybacked on the push-sync handshake ---
    static constexpr int kSyncWords = 4;
    void set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag) override;
    void clear_watchdog_flags() override;

    void teardown() override;

    // Accessors
    int64_t feature_dim() const { return max_feature_dim_; }
    int64_t max_feature_dim() const { return max_feature_dim_; }
    int64_t num_local() const { return num_local_; }
    int64_t total_send() const { return total_send_; }
    int64_t total_recv() const { return total_recv_; }
    bool is_setup() const override { return is_setup_; }
    bool use_fabric() const override { return use_fabric_; }
    torch::Tensor send_local_indices() const { return send_local_indices_; }

private:
    struct VmmMapping {
        CUdeviceptr vaddr = 0;
        size_t padded_size = 0;
        CUmemGenericAllocationHandle handle = 0;
    };

    void launch_forward_impl(torch::Tensor x_slot_view, int64_t feature_dim, int64_t slot,
                             cudaStream_t stream, int64_t ce_threshold_bytes);

    int64_t rank_ = 0;
    int64_t world_size_ = 0;
    int64_t max_feature_dim_ = 0;
    int64_t num_local_ = 0;
    int64_t total_send_ = 0;
    int64_t total_recv_ = 0;
    int64_t num_slots_ = 1;
    int device_id_ = 0;
    bool is_setup_ = false;
    bool use_fabric_ = false;

    // Host copies of CSR offsets cached during prepare_local_metadata
    torch::Tensor cpu_send_offsets_;
    torch::Tensor cpu_recv_offsets_;

    // CSR maps (device tensors)
    torch::Tensor send_offsets_;       // [W+1]
    torch::Tensor send_local_indices_; // [total_send]
    torch::Tensor recv_offsets_;       // [W+1]

    // Push targets for forward (write to peers' x_buffer ghost region)
    torch::Tensor fwd_targets_; // [W] PushTarget, device memory

    // Push targets for backward (write to peers' grad_recv_buffer)
    torch::Tensor bwd_targets_; // [W] PushTarget, device memory

    // Staging buffer for CE path (local contiguous, [total_send, max_feature_dim])
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
    uint32_t *local_counter_ = nullptr;
    std::vector<CUdeviceptr> remote_counters_;
    std::vector<VmmMapping> remote_sync_mappings_;
    uint32_t call_counter_ = 0;
    bool sync_setup_ = false;
    bool sync_use_fabric_ = false;

    // Local sync counter Fabric VMM resources (if sync_use_fabric_)
    CUdeviceptr local_counter_vaddr_ = 0;
    size_t local_counter_padded_size_ = 0;
    CUmemGenericAllocationHandle local_counter_handle_ = 0;

    // Drift-watchdog piggyback (see set_watchdog_flags). Owned by the caller.
    torch::Tensor watchdog_local_;  // int32 [1]
    torch::Tensor watchdog_global_; // int32 [1]
    torch::Tensor sync_word_ptrs_;  // int64 [W] on device: base of every rank's sync word
    void build_sync_word_ptrs();
};

using IPCBackend = IPCExchanger;

// High-level backend-agnostic feature exchanger.
// Owns transport-registered buffers sized to `max_feature_dim` (double-buffered across 2 slots)
// and delegates communication to a pluggable `CommBackend` (default: "ipc").
// Supports pushing any 2D tensor [N_local, F] with 1 <= F <= max_feature_dim on the same instance.
class FeatureExchanger
{
public:
    explicit FeatureExchanger(const std::string &backend_type = "ipc");
    ~FeatureExchanger();

    static bool supports_fabric_handles(int device_id = -1)
    {
        return IPCExchanger::supports_fabric_handles(device_id);
    }

    static torch::Tensor allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index)
    {
        return IPCExchanger::allocate_ipc_buffer(rows, cols, device_index);
    }

    // --- Unified 2-step backend-agnostic setup (1 all_gather in Python) ---
    std::vector<uint8_t>
    prepare_setup(const dynamis_state::PartitionMessagePassingInfo &partition_info,
                  int64_t max_feature_dim,
                  int64_t rank,
                  int64_t world_size,
                  int64_t device_index);

    void complete_setup(const std::vector<std::vector<uint8_t>> &all_rank_metadata);

    // --- Public exchange interface (works for any F <= max_feature_dim) ---
    torch::Tensor feature_push_forward(const torch::Tensor &local_features);

    torch::Tensor feature_push_forward(
        const torch::Tensor &local_features,
        const dynamis_state::PartitionMessagePassingInfo &partition_info)
    {
        return feature_push_forward(local_features);
    }

    torch::Tensor feature_push_backward(const torch::Tensor &grad_output);

    torch::Tensor feature_push_backward(
        const torch::Tensor &grad_output,
        const torch::IntArrayRef &grad_input_shape,
        const dynamis_state::PartitionMessagePassingInfo &partition_info)
    {
        return feature_push_backward(grad_output);
    }

    // --- Watchdog & lifecycle ---
    void set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag)
    {
        backend_->set_watchdog_flags(local_flag, global_flag);
    }
    void clear_watchdog_flags() { backend_->clear_watchdog_flags(); }

    void teardown();
    bool is_setup() const { return backend_ && backend_->is_setup(); }
    bool is_ipc_setup() const { return is_setup(); }
    bool use_fabric() const { return backend_ && backend_->use_fabric(); }
    int64_t max_feature_dim() const { return max_feature_dim_; }
    int64_t num_local() const { return num_local_; }
    void set_ce_threshold_bytes(int64_t t) { ce_threshold_bytes_ = t; }
    int64_t ce_threshold_bytes() const { return ce_threshold_bytes_; }

    // --- Legacy IPC setup helpers (retained for backward compatibility) ---
    std::vector<uint8_t> export_ipc_handle(torch::Tensor buffer);
    void setup_ipc(torch::Tensor x_buffer, torch::Tensor grad_recv_buffer,
                   const std::vector<std::vector<uint8_t>> &all_x_handles,
                   const std::vector<std::vector<uint8_t>> &all_grad_handles,
                   torch::Tensor send_offsets, torch::Tensor send_local_indices,
                   torch::Tensor recv_offsets, torch::Tensor all_recv_offsets,
                   torch::Tensor all_num_local, int64_t num_local, int64_t feature_dim,
                   int64_t rank, int64_t world_size);
    std::vector<uint8_t> create_and_export_sync_handle();
    void setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles);
    void record_push_complete();
    void wait_for_pushes();

private:
    std::unique_ptr<CommBackend> backend_;
    int64_t ce_threshold_bytes_ = 0;
    int64_t num_local_ = 0;
    int64_t total_send_ = 0;
    int64_t total_recv_ = 0;
    int64_t max_feature_dim_ = 0;
    int64_t num_slots_ = 2;

    torch::Tensor send_local_indices_;
    torch::Tensor x_buffer_;         // [num_slots_ * max(1, N_local + N_ghost), max_feature_dim_]
    torch::Tensor grad_recv_buffer_; // [num_slots_ * max(1, total_send), max_feature_dim_]
};

} // namespace dynamis_model
