// ipc_exchanger.cpp — CUDA IPC handle management and push target construction.

#include "distnnp_model/exchanger.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <cstring>

namespace distnnp_model
{

// Forward declarations for kernel launchers (defined in ipc_exchange_kernels.cu)
void launch_gather_and_push(
    torch::Tensor x_buffer,
    torch::Tensor send_offsets,
    torch::Tensor send_local_indices,
    torch::Tensor fwd_targets,
    int64_t total_send,
    int64_t feature_dim,
    int64_t world_size,
    cudaStream_t stream);

void launch_gather_ghost_grad_and_push(
    torch::Tensor grad,
    torch::Tensor recv_offsets,
    torch::Tensor bwd_targets,
    int64_t num_local,
    int64_t total_recv,
    int64_t feature_dim,
    int64_t world_size,
    cudaStream_t stream);

void launch_gather_to_staging(
    torch::Tensor x_buffer,
    torch::Tensor send_local_indices,
    torch::Tensor staging,
    int64_t total_send,
    int64_t feature_dim,
    cudaStream_t stream);

// ---------------------------------------------------------------------------

torch::Tensor IPCExchanger::allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index)
{
    TORCH_CHECK(rows >= 0 && cols >= 0, "rows and cols must be non-negative");

    int prev_device;
    auto err = cudaGetDevice(&prev_device);
    TORCH_CHECK(err == cudaSuccess,
                "cudaGetDevice failed: ", cudaGetErrorString(err));
    err = cudaSetDevice(static_cast<int>(device_index));
    TORCH_CHECK(err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(err));

    size_t num_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    // Minimum 1 byte to avoid cudaMalloc(0) edge case
    if (num_bytes == 0)
        num_bytes = sizeof(float);

    float *ptr = nullptr;
    err = cudaMalloc(&ptr, num_bytes);
    TORCH_CHECK(err == cudaSuccess,
                "cudaMalloc failed for IPC buffer: ", cudaGetErrorString(err));

    // Zero-initialize
    err = cudaMemset(ptr, 0, num_bytes);
    TORCH_CHECK(err == cudaSuccess,
                "cudaMemset failed: ", cudaGetErrorString(err));

    err = cudaSetDevice(prev_device);
    TORCH_CHECK(err == cudaSuccess,
                "cudaSetDevice (restore) failed: ", cudaGetErrorString(err));

    // Wrap as a PyTorch tensor with a custom deleter that calls cudaFree.
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_index);
    auto tensor = torch::from_blob(
        ptr, {rows, cols}, /*deleter=*/[](void *p)
        { cudaFree(p); },
        options);

    return tensor;
}

IPCExchanger::~IPCExchanger()
{
    if (is_setup_)
        teardown();
}

std::vector<uint8_t> IPCExchanger::export_ipc_handle(torch::Tensor buffer)
{
    TORCH_CHECK(buffer.is_cuda(), "export_ipc_handle: buffer must be on CUDA device");
    TORCH_CHECK(buffer.is_contiguous(), "export_ipc_handle: buffer must be contiguous");

    cudaIpcMemHandle_t handle;
    auto err = cudaIpcGetMemHandle(&handle, buffer.data_ptr());
    TORCH_CHECK(err == cudaSuccess,
                "cudaIpcGetMemHandle failed: ", cudaGetErrorString(err));

    std::vector<uint8_t> bytes(sizeof(cudaIpcMemHandle_t));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
}

void IPCExchanger::setup(
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
    int64_t world_size)
{
    if (is_setup_)
        teardown();

    rank_ = rank;
    world_size_ = world_size;
    feature_dim_ = feature_dim;
    num_local_ = num_local;

    // Store CSR maps — ensure on CPU for host-side indexing during setup,
    // keep device copies for kernel use.
    auto cpu_send_offsets = send_offsets.to(torch::kCPU).contiguous();
    auto cpu_recv_offsets = recv_offsets.to(torch::kCPU).contiguous();
    auto cpu_all_recv_offsets = all_recv_offsets.to(torch::kCPU).contiguous();
    auto cpu_all_num_local = all_num_local.to(torch::kCPU).contiguous();

    send_offsets_ = send_offsets.to(x_buffer.device()).contiguous();
    send_local_indices_ = send_local_indices.to(x_buffer.device()).contiguous();
    recv_offsets_ = recv_offsets.to(x_buffer.device()).contiguous();

    auto *so = cpu_send_offsets.data_ptr<int64_t>();
    auto *ro = cpu_recv_offsets.data_ptr<int64_t>();
    total_send_ = so[world_size];
    total_recv_ = ro[world_size];

    // --- Open IPC handles ---
    opened_x_ptrs_.resize(world_size, nullptr);
    opened_grad_ptrs_.resize(world_size, nullptr);

    for (int64_t r = 0; r < world_size; ++r)
    {
        if (r == rank)
            continue;

        // Open x_buffer handle from rank r
        {
            cudaIpcMemHandle_t h;
            TORCH_CHECK(all_x_handles[r].size() == sizeof(h),
                        "Invalid IPC handle size from rank ", r);
            std::memcpy(&h, all_x_handles[r].data(), sizeof(h));

            void *ptr = nullptr;
            auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
            TORCH_CHECK(err == cudaSuccess,
                        "cudaIpcOpenMemHandle (x) for rank ", r,
                        " failed: ", cudaGetErrorString(err));
            opened_x_ptrs_[r] = ptr;
        }

        // Open grad_recv_buffer handle from rank r
        {
            cudaIpcMemHandle_t h;
            TORCH_CHECK(all_grad_handles[r].size() == sizeof(h),
                        "Invalid IPC grad handle size from rank ", r);
            std::memcpy(&h, all_grad_handles[r].data(), sizeof(h));

            void *ptr = nullptr;
            auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
            TORCH_CHECK(err == cudaSuccess,
                        "cudaIpcOpenMemHandle (grad) for rank ", r,
                        " failed: ", cudaGetErrorString(err));
            opened_grad_ptrs_[r] = ptr;
        }
    }

    // --- Build forward push targets ---
    // For forward: sender A writes to receiver R's x_buffer ghost region.
    // Receiver R's ghost region starts at x_buffer[N_local_R].
    // Within ghost region, atoms from sender A start at recv_offsets_R[A].
    // So remote offset (in atoms) = N_local_R + recv_offsets_R[A].
    auto *aro = cpu_all_recv_offsets.data_ptr<int64_t>();  // [W, W+1]
    auto *anl = cpu_all_num_local.data_ptr<int64_t>();     // [W]

    std::vector<PushTarget> fwd_targets_host(world_size);
    for (int64_t r = 0; r < world_size; ++r)
    {
        if (r == rank)
        {
            fwd_targets_host[r] = {nullptr, 0, so[r], 0};
            continue;
        }

        // Rank r's recv_offsets[rank] = offset in rank r's ghost region for data from us
        int64_t recv_offset_on_r_from_us = aro[r * (world_size + 1) + rank];
        int64_t n_local_r = anl[r];

        fwd_targets_host[r].remote_ptr =
            static_cast<float *>(opened_x_ptrs_[r]);
        fwd_targets_host[r].remote_offset = n_local_r + recv_offset_on_r_from_us;
        fwd_targets_host[r].send_start = so[r];
        fwd_targets_host[r].send_count = so[r + 1] - so[r];
    }

    // --- Build backward push targets ---
    // For backward: this rank gathers ghost gradients and pushes to the owner.
    // Ghost atoms from peer r are at recv_offsets[r]:recv_offsets[r+1] in ghost region.
    // We push them to peer r's grad_recv_buffer.
    // Peer r's grad_recv_buffer has CSR layout matching peer r's send_offsets.
    // We need: where does peer r's send_offsets put data destined for us?
    //   → peer r's send_offsets[rank] (= offset in peer r's send_local_indices for us)
    //   → This is also the offset in peer r's grad_recv_buffer where we write.
    //
    // We can derive peer r's send_offsets[rank] from all_recv_offsets:
    // Due to the invariant send_count(A→B) == recv_count(B←A),
    // and the CSR ordering, peer r's send_offsets[rank] is the sum of
    // peer r's send counts to ranks 0..rank-1.
    //
    // But we don't have peer r's send_offsets directly. We DO have the
    // global_send_recv_matrix, but that's not passed here.
    //
    // Simpler: the all_recv_offsets[r] gives us rank r's recv CSR.
    // The send count from rank r to rank j = recv count on rank j from rank r
    //   = all_recv_offsets[j, r+1] - all_recv_offsets[j, r]
    //
    // So peer r's send_offsets[rank] = sum over j < rank of (send_count(r→j))
    //   = sum over j < rank of (all_recv_offsets[j, r+1] - all_recv_offsets[j, r])

    std::vector<PushTarget> bwd_targets_host(world_size);
    for (int64_t r = 0; r < world_size; ++r)
    {
        if (r == rank)
        {
            bwd_targets_host[r] = {nullptr, 0, ro[r], 0};
            continue;
        }

        // Compute peer r's send_offsets[rank_]:
        // = sum over j in [0, rank_) of send_count(r → j)
        // = sum over j in [0, rank_) of (all_recv_offsets[j][r+1] - all_recv_offsets[j][r])
        int64_t offset_in_peer_grad = 0;
        for (int64_t j = 0; j < rank_; ++j)
        {
            int64_t count_r_to_j = aro[j * (world_size + 1) + r + 1] -
                                   aro[j * (world_size + 1) + r];
            offset_in_peer_grad += count_r_to_j;
        }

        bwd_targets_host[r].remote_ptr =
            static_cast<float *>(opened_grad_ptrs_[r]);
        bwd_targets_host[r].remote_offset = offset_in_peer_grad;
        bwd_targets_host[r].send_start = ro[r];  // not used by bwd kernel, but fill for consistency
        bwd_targets_host[r].send_count = ro[r + 1] - ro[r];
    }

    // Upload push targets to device memory
    auto device = x_buffer.device();
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    fwd_targets_ = torch::empty({static_cast<int64_t>(sizeof(PushTarget) * world_size)}, byte_opts);
    {
        auto err = cudaMemcpy(fwd_targets_.data_ptr(), fwd_targets_host.data(),
                              sizeof(PushTarget) * world_size, cudaMemcpyHostToDevice);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaMemcpy fwd_targets failed: ", cudaGetErrorString(err));
    }

    bwd_targets_ = torch::empty({static_cast<int64_t>(sizeof(PushTarget) * world_size)}, byte_opts);
    {
        auto err = cudaMemcpy(bwd_targets_.data_ptr(), bwd_targets_host.data(),
                              sizeof(PushTarget) * world_size, cudaMemcpyHostToDevice);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaMemcpy bwd_targets failed: ", cudaGetErrorString(err));
    }

    // Allocate staging buffer for CE path (local contiguous gather target).
    // No IPC handle needed — only used as local gather destination + memcpy source.
    if (total_send_ > 0)
    {
        auto buf_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
        staging_buffer_ = torch::empty({total_send_, feature_dim}, buf_opts);
    }

    // Keep host copy of forward push targets for CE path cudaMemcpyAsync calls
    ce_fwd_targets_ = fwd_targets_host;

    is_setup_ = true;
}

void IPCExchanger::push_forward(torch::Tensor x_buffer, cudaStream_t stream,
                                int64_t ce_threshold_bytes)
{
    TORCH_CHECK(is_setup_, "IPCExchanger::push_forward: not set up");
    TORCH_CHECK(x_buffer.is_cuda() && x_buffer.is_contiguous(),
                "x_buffer must be contiguous CUDA tensor");

    if (total_send_ == 0)
        return;

    if (ce_threshold_bytes <= 0)
    {
        // Pure SM kernel path (all peers via single kernel)
        launch_gather_and_push(
            x_buffer,
            send_offsets_,
            send_local_indices_,
            fwd_targets_,
            total_send_,
            feature_dim_,
            world_size_,
            stream);
        return;
    }

    // CE mode decision: if any peer exceeds the threshold, use CE for all peers.
    // CE mode = gather to contiguous staging buffer, then cudaMemcpyAsync per peer.
    // This uses the Copy Engine DMA for high bandwidth NVLink transfers.
    // If all peers are below threshold, fall back to SM kernel push (lower latency).
    bool use_ce = false;
    for (int64_t r = 0; r < world_size_; ++r)
    {
        if (r == rank_)
            continue;
        int64_t bytes = ce_fwd_targets_[r].send_count * feature_dim_ * sizeof(float);
        if (bytes >= ce_threshold_bytes)
        {
            use_ce = true;
            break;
        }
    }

    if (!use_ce)
    {
        // All peers below threshold → SM kernel (lower latency for small messages)
        launch_gather_and_push(
            x_buffer,
            send_offsets_,
            send_local_indices_,
            fwd_targets_,
            total_send_,
            feature_dim_,
            world_size_,
            stream);
        return;
    }

    // CE path stage 1: gather scattered send data to contiguous staging buffer
    launch_gather_to_staging(
        x_buffer,
        send_local_indices_,
        staging_buffer_,
        total_send_,
        feature_dim_,
        stream);

    // CE path stage 2: per-peer DMA transfer from staging to remote
    for (int64_t r = 0; r < world_size_; ++r)
    {
        if (r == rank_)
            continue;

        const auto &tgt = ce_fwd_targets_[r];
        if (tgt.send_count == 0)
            continue;

        int64_t bytes = tgt.send_count * feature_dim_ * sizeof(float);
        float *src = staging_buffer_.data_ptr<float>() + tgt.send_start * feature_dim_;
        float *dst = tgt.remote_ptr + tgt.remote_offset * feature_dim_;

        // CE path: cudaMemcpyAsync uses the Copy Engine for large DMA transfers.
        // cudaMemcpyDefault works for both local and IPC-opened remote pointers.
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
    }
}

void IPCExchanger::push_backward(torch::Tensor grad_at_ghost, cudaStream_t stream)
{
    TORCH_CHECK(is_setup_, "IPCExchanger::push_backward: not set up");
    TORCH_CHECK(grad_at_ghost.is_cuda() && grad_at_ghost.is_contiguous(),
                "grad tensor must be contiguous CUDA tensor");

    launch_gather_ghost_grad_and_push(
        grad_at_ghost,
        recv_offsets_,
        bwd_targets_,
        num_local_,
        total_recv_,
        feature_dim_,
        world_size_,
        stream);
}

// --- IPC push synchronization (generation counter) ---

std::vector<uint8_t> IPCExchanger::create_and_export_sync_handle()
{
    // Allocate a single uint32 counter in IPC-shareable memory.
    if (local_counter_)
    {
        cudaFree(local_counter_);
        local_counter_ = nullptr;
    }
    call_counter_ = 0;

    auto err = cudaMalloc(&local_counter_, sizeof(uint32_t));
    TORCH_CHECK(err == cudaSuccess, 
                "cudaMalloc (counter) failed: ", cudaGetErrorString(err));
    err = cudaMemset(local_counter_, 0, sizeof(uint32_t));
    TORCH_CHECK(err == cudaSuccess, 
                "cudaMemset (counter) failed: ", cudaGetErrorString(err));

    cudaIpcMemHandle_t handle;
    err = cudaIpcGetMemHandle(&handle, local_counter_);
    TORCH_CHECK(err == cudaSuccess,
                "cudaIpcGetMemHandle (counter) failed: ", cudaGetErrorString(err));

    std::vector<uint8_t> bytes(sizeof(handle));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
}

void IPCExchanger::setup_sync(
    const std::vector<std::vector<uint8_t>> &all_sync_handles)
{
    TORCH_CHECK(is_setup_, "setup_sync: must call setup() first");
    TORCH_CHECK(local_counter_ != nullptr,
                "setup_sync: must call create_and_export_sync_handle() first");

    // Close any previously opened remote counters.
    for (auto &p : remote_counters_)
    {
        if (p)
            cudaIpcCloseMemHandle(reinterpret_cast<void *>(p));
    }
    remote_counters_.assign(world_size_, 0);

    for (int64_t r = 0; r < world_size_; ++r)
    {
        if (r == rank_)
            continue;

        cudaIpcMemHandle_t h;
        TORCH_CHECK(all_sync_handles[r].size() == sizeof(h),
                    "Invalid IPC counter handle size from rank ", r);
        std::memcpy(&h, all_sync_handles[r].data(), sizeof(h));

        void *ptr = nullptr;
        auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaIpcOpenMemHandle (counter) for rank ", r,
                    " failed: ", cudaGetErrorString(err));
        remote_counters_[r] = reinterpret_cast<CUdeviceptr>(ptr);
    }

    sync_setup_ = true;
}

void IPCExchanger::record_push_complete(cudaStream_t stream)
{
    TORCH_CHECK(sync_setup_, "record_push_complete: counter not set up");
    ++call_counter_;
    // cuStreamWriteValue32 with default flags is ordered after prior stream
    // operations — i.e. the write follows the push memcpy on this stream.
    CUresult err = cuStreamWriteValue32(
        reinterpret_cast<CUstream>(stream),
        reinterpret_cast<CUdeviceptr>(local_counter_),
        call_counter_, 0);
    if (err != CUDA_SUCCESS)
    {
        const char *msg = nullptr;
        cuGetErrorString(err, &msg);
        TORCH_CHECK(false, "cuStreamWriteValue32 failed: ",
                    (msg ? msg : "(unknown)"));
    }
}

void IPCExchanger::wait_for_pushes(cudaStream_t stream)
{
    TORCH_CHECK(sync_setup_, "wait_for_pushes: counter not set up");
    for (int64_t r = 0; r < world_size_; ++r)
    {
        if (r == rank_ || !remote_counters_[r])
            continue;
        CUresult err = cuStreamWaitValue32(
            reinterpret_cast<CUstream>(stream),
            remote_counters_[r],
            call_counter_, CU_STREAM_WAIT_VALUE_GEQ);
        if (err != CUDA_SUCCESS)
        {
            const char *msg = nullptr;
            cuGetErrorString(err, &msg);
            TORCH_CHECK(false, "cuStreamWaitValue32 for rank ", r,
                        " failed: ", (msg ? msg : "(unknown)"));
        }
    }
}

void IPCExchanger::teardown()
{
    // Clean up generation counter
    if (local_counter_)
    {
        cudaFree(local_counter_);
        local_counter_ = nullptr;
    }
    for (auto &p : remote_counters_)
    {
        if (p)
        {
            cudaIpcCloseMemHandle(reinterpret_cast<void *>(p));
            p = 0;
        }
    }
    remote_counters_.clear();
    call_counter_ = 0;
    sync_setup_ = false;

    if (!is_setup_)
        return;

    for (int64_t r = 0; r < world_size_; ++r)
    {
        if (r == rank_)
            continue;
        if (opened_x_ptrs_[r])
        {
            cudaIpcCloseMemHandle(opened_x_ptrs_[r]);
            opened_x_ptrs_[r] = nullptr;
        }
        if (opened_grad_ptrs_[r])
        {
            cudaIpcCloseMemHandle(opened_grad_ptrs_[r]);
            opened_grad_ptrs_[r] = nullptr;
        }
    }

    opened_x_ptrs_.clear();
    opened_grad_ptrs_.clear();

    // Release device tensors and clear CE targets (contains dangling IPC pointers)
    staging_buffer_.reset();
    ce_fwd_targets_.clear();
    fwd_targets_.reset();
    bwd_targets_.reset();

    is_setup_ = false;
}

} // namespace distnnp_model
