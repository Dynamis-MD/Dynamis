// ipc_exchanger.cpp — CUDA IPC and Fabric handle management and push target construction.

#include "dynamis_model/exchanger.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <algorithm>

namespace dynamis_model {

// Forward declarations for kernel launchers (defined in ipc_exchange_kernels.cu)
void launch_gather_and_push(torch::Tensor x_buffer, torch::Tensor send_offsets,
                            torch::Tensor send_local_indices, torch::Tensor fwd_targets,
                            int64_t total_send, int64_t feature_dim, int64_t world_size,
                            int64_t slot, cudaStream_t stream);

void launch_gather_ghost_grad_and_push(torch::Tensor grad, torch::Tensor recv_offsets,
                                       torch::Tensor bwd_targets, int64_t num_local,
                                       int64_t total_recv, int64_t feature_dim, int64_t world_size,
                                       int64_t slot, cudaStream_t stream);

void launch_gather_to_staging(torch::Tensor x_buffer, torch::Tensor send_local_indices,
                              torch::Tensor staging, int64_t total_send, int64_t feature_dim,
                              cudaStream_t stream);

void launch_watchdog_or(torch::Tensor sync_word_ptrs, int64_t world_size, uint32_t slot,
                        torch::Tensor global_flag, cudaStream_t stream);

// Bytes in one rank's sync word: [counter | flag[0] | flag[1] | pad].
static constexpr size_t kSyncBytes = IPCExchanger::kSyncWords * sizeof(uint32_t);

namespace {

struct IPCRankMetadataHeader {
    int64_t num_local;
    int64_t max_feature_dim;
    int64_t num_slots;
    uint64_t handle_size;
};

} // namespace

// ---------------------------------------------------------------------------
// Error checking helper for CUDA Driver API
// ---------------------------------------------------------------------------
static inline void check_cu_call(CUresult res, const char *msg)
{
    if (res != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(res, &err_str);
        TORCH_CHECK(false, msg, ": ", (err_str ? err_str : "unknown driver error"), " (code ",
                    static_cast<int>(res), ")");
    }
}

// ---------------------------------------------------------------------------
// Capability detection for Fabric handles (CUDA 12+ / Multi-node NVLink / NVL72)
// ---------------------------------------------------------------------------
bool supports_fabric_handles(int device_id)
{
    int dev = device_id;
    if (dev < 0) {
        auto err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
            return false;
    }

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    cuInit(0);
    int fabric_supported = 0;
    CUresult res = cuDeviceGetAttribute(&fabric_supported,
                                        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
    if (res == CUDA_SUCCESS && fabric_supported != 0) {
        return true;
    }
#endif
    return false;
}

bool IPCExchanger::supports_fabric_handles(int device_id)
{
    return dynamis_model::supports_fabric_handles(device_id);
}

// ---------------------------------------------------------------------------
// Thread-safe registry for locally allocated Fabric VMM buffers
// ---------------------------------------------------------------------------
#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
struct FabricAllocRecord {
    CUmemGenericAllocationHandle alloc_handle;
    CUDA_MEM_FABRIC_HANDLE fabric_handle;
    size_t padded_size;
    int device_id;
};

static std::mutex g_fabric_registry_mutex;
static std::unordered_map<void *, FabricAllocRecord> g_fabric_registry;

static void register_fabric_alloc(void *ptr, const FabricAllocRecord &record)
{
    std::lock_guard<std::mutex> lock(g_fabric_registry_mutex);
    g_fabric_registry[ptr] = record;
}

static bool lookup_fabric_alloc(void *ptr, FabricAllocRecord &record)
{
    std::lock_guard<std::mutex> lock(g_fabric_registry_mutex);
    auto it = g_fabric_registry.find(ptr);
    if (it != g_fabric_registry.end()) {
        record = it->second;
        return true;
    }
    return false;
}

static void unregister_fabric_alloc(void *ptr)
{
    std::lock_guard<std::mutex> lock(g_fabric_registry_mutex);
    g_fabric_registry.erase(ptr);
}
#endif

// ---------------------------------------------------------------------------

torch::Tensor IPCExchanger::allocate_ipc_buffer(int64_t rows, int64_t cols, int64_t device_index)
{
    TORCH_CHECK(rows >= 0 && cols >= 0, "rows and cols must be non-negative");

    int prev_device;
    auto err = cudaGetDevice(&prev_device);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    err = cudaSetDevice(static_cast<int>(device_index));
    TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));

    size_t num_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    // Minimum 1 float to avoid 0-byte allocation edge cases
    if (num_bytes == 0)
        num_bytes = sizeof(float);

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    if (supports_fabric_handles(static_cast<int>(device_index))) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = static_cast<int>(device_index);
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        check_cu_call(res, "cuMemGetAllocationGranularity failed");

        size_t padded_size = ((num_bytes + granularity - 1) / granularity) * granularity;

        CUmemGenericAllocationHandle alloc_handle;
        res = cuMemCreate(&alloc_handle, padded_size, &prop, 0);
        check_cu_call(res, "cuMemCreate failed");

        CUDA_MEM_FABRIC_HANDLE fabric_handle;
        std::memset(&fabric_handle, 0, sizeof(fabric_handle));
        res = cuMemExportToShareableHandle(&fabric_handle, alloc_handle, CU_MEM_HANDLE_TYPE_FABRIC,
                                           0);
        check_cu_call(res, "cuMemExportToShareableHandle failed");

        CUdeviceptr d_ptr = 0;
        res = cuMemAddressReserve(&d_ptr, padded_size, 0, 0, 0);
        check_cu_call(res, "cuMemAddressReserve failed");

        res = cuMemMap(d_ptr, padded_size, 0, alloc_handle, 0);
        check_cu_call(res, "cuMemMap failed");

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = static_cast<int>(device_index);
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        res = cuMemSetAccess(d_ptr, padded_size, &access_desc, 1);
        check_cu_call(res, "cuMemSetAccess failed");

        void *raw_ptr = reinterpret_cast<void *>(d_ptr);
        err = cudaMemset(raw_ptr, 0, num_bytes);
        TORCH_CHECK(err == cudaSuccess, "cudaMemset failed: ", cudaGetErrorString(err));

        err = cudaSetDevice(prev_device);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaSetDevice (restore) failed: ", cudaGetErrorString(err));

        FabricAllocRecord record{ alloc_handle, fabric_handle, padded_size,
                                  static_cast<int>(device_index) };
        register_fabric_alloc(raw_ptr, record);

        auto options =
                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_index);
        auto tensor = torch::from_blob(
                raw_ptr, { rows, cols },
                [d_ptr, padded_size, alloc_handle](void *p) {
                    unregister_fabric_alloc(p);
                    cuMemUnmap(d_ptr, padded_size);
                    cuMemRelease(alloc_handle);
                    cuMemAddressFree(d_ptr, padded_size);
                },
                options);

        return tensor;
    }
#endif

    // Legacy cudaMalloc fallback path
    float *ptr = nullptr;
    err = cudaMalloc(&ptr, num_bytes);
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed for IPC buffer: ", cudaGetErrorString(err));

    err = cudaMemset(ptr, 0, num_bytes);
    TORCH_CHECK(err == cudaSuccess, "cudaMemset failed: ", cudaGetErrorString(err));

    err = cudaSetDevice(prev_device);
    TORCH_CHECK(err == cudaSuccess, "cudaSetDevice (restore) failed: ", cudaGetErrorString(err));

    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, device_index);
    auto tensor = torch::from_blob(
            ptr, { rows, cols }, /*deleter=*/[](void *p) { cudaFree(p); }, options);

    return tensor;
}

IPCExchanger::~IPCExchanger()
{
    if (is_setup_ || sync_setup_)
        teardown();
}

std::vector<uint8_t> IPCExchanger::export_ipc_handle(torch::Tensor buffer)
{
    TORCH_CHECK(buffer.is_cuda(), "export_ipc_handle: buffer must be on CUDA device");
    TORCH_CHECK(buffer.is_contiguous(), "export_ipc_handle: buffer must be contiguous");

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    FabricAllocRecord record;
    if (lookup_fabric_alloc(buffer.data_ptr(), record)) {
        std::vector<uint8_t> bytes(sizeof(CUDA_MEM_FABRIC_HANDLE));
        std::memcpy(bytes.data(), &record.fabric_handle, sizeof(CUDA_MEM_FABRIC_HANDLE));
        return bytes;
    }
#endif

    cudaIpcMemHandle_t handle;
    auto err = cudaIpcGetMemHandle(&handle, buffer.data_ptr());
    TORCH_CHECK(err == cudaSuccess, "cudaIpcGetMemHandle failed: ", cudaGetErrorString(err));

    std::vector<uint8_t> bytes(sizeof(cudaIpcMemHandle_t));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
}

// ---------------------------------------------------------------------------
// Unified 2-step CommBackend setup (single collective in Python)
// ---------------------------------------------------------------------------

std::vector<uint8_t> IPCExchanger::prepare_local_metadata(
    torch::Tensor &x_buffer,
    torch::Tensor &grad_recv_buffer,
    const dynamis_state::PartitionMessagePassingInfo &partition_info,
    int64_t max_feature_dim,
    int64_t rank,
    int64_t world_size,
    int64_t device_index)
{
    if (is_setup_ || sync_setup_)
        teardown();

    TORCH_CHECK(max_feature_dim > 0, "max_feature_dim must be positive, got ", max_feature_dim);
    TORCH_CHECK(world_size >= 1 && rank >= 0 && rank < world_size,
                "Invalid rank (", rank, ") or world_size (", world_size, ")");

    int prev_device = 0;
    auto err = cudaGetDevice(&prev_device);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    err = cudaSetDevice(static_cast<int>(device_index));
    TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));

    int64_t num_local = partition_info.partition_indices.numel();
    auto cpu_send_offsets = partition_info.send_offsets.to(torch::kCPU, torch::kInt64).contiguous();
    auto cpu_recv_offsets = partition_info.recv_offsets.to(torch::kCPU, torch::kInt64).contiguous();
    TORCH_CHECK(cpu_send_offsets.numel() == world_size + 1,
                "send_offsets must have world_size + 1 elements");
    TORCH_CHECK(cpu_recv_offsets.numel() == world_size + 1,
                "recv_offsets must have world_size + 1 elements");

    int64_t total_send = cpu_send_offsets.data_ptr<int64_t>()[world_size];
    int64_t total_recv = cpu_recv_offsets.data_ptr<int64_t>()[world_size];

    rank_ = rank;
    world_size_ = world_size;
    max_feature_dim_ = max_feature_dim;
    num_local_ = num_local;
    total_send_ = total_send;
    total_recv_ = total_recv;
    num_slots_ = 2; // Double-buffered across consecutive pushes
    device_id_ = static_cast<int>(device_index);
    use_fabric_ = supports_fabric_handles(device_id_);

    auto dev = torch::Device(torch::kCUDA, device_index);
    send_offsets_ = partition_info.send_offsets.to(dev, torch::kInt64).contiguous();
    send_local_indices_ = partition_info.send_local_indices.to(dev, torch::kInt64).contiguous();
    recv_offsets_ = partition_info.recv_offsets.to(dev, torch::kInt64).contiguous();
    cpu_send_offsets_ = cpu_send_offsets;
    cpu_recv_offsets_ = cpu_recv_offsets;

    int64_t x_rows_per_slot = std::max<int64_t>(1, num_local_ + total_recv_);
    int64_t grad_rows_per_slot = std::max<int64_t>(1, total_send_);

    x_buffer = allocate_ipc_buffer(num_slots_ * x_rows_per_slot, max_feature_dim_, device_index);
    grad_recv_buffer =
            allocate_ipc_buffer(num_slots_ * grad_rows_per_slot, max_feature_dim_, device_index);

    auto x_handle = export_ipc_handle(x_buffer);
    auto grad_handle = export_ipc_handle(grad_recv_buffer);
    auto sync_handle = create_and_export_sync_handle();

    TORCH_CHECK(x_handle.size() == grad_handle.size() && x_handle.size() == sync_handle.size(),
                "Mismatched IPC/Fabric handle sizes");

    err = cudaSetDevice(prev_device);
    TORCH_CHECK(err == cudaSuccess, "cudaSetDevice (restore) failed: ", cudaGetErrorString(err));

    IPCRankMetadataHeader hdr{};
    hdr.num_local = num_local_;
    hdr.max_feature_dim = max_feature_dim_;
    hdr.num_slots = num_slots_;
    hdr.handle_size = static_cast<uint64_t>(x_handle.size());

    const size_t ro_bytes = static_cast<size_t>(world_size_ + 1) * sizeof(int64_t);
    const size_t total_meta_bytes =
            sizeof(IPCRankMetadataHeader) + 3 * x_handle.size() + ro_bytes;

    std::vector<uint8_t> meta(total_meta_bytes);
    uint8_t *p = meta.data();
    std::memcpy(p, &hdr, sizeof(hdr));
    p += sizeof(hdr);
    std::memcpy(p, x_handle.data(), x_handle.size());
    p += x_handle.size();
    std::memcpy(p, grad_handle.data(), grad_handle.size());
    p += grad_handle.size();
    std::memcpy(p, sync_handle.data(), sync_handle.size());
    p += sync_handle.size();
    std::memcpy(p, cpu_recv_offsets_.data_ptr<int64_t>(), ro_bytes);

    return meta;
}

void IPCExchanger::connect(const std::vector<std::vector<uint8_t>> &all_rank_metadata)
{
    TORCH_CHECK(static_cast<int64_t>(all_rank_metadata.size()) == world_size_,
                "connect: expected metadata for ", world_size_, " ranks, got ",
                all_rank_metadata.size());

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id_);

    std::vector<int64_t> all_num_local(world_size_);
    std::vector<int64_t> all_max_f(world_size_);
    std::vector<int64_t> all_slots(world_size_);
    std::vector<std::vector<uint8_t>> all_x_handles(world_size_);
    std::vector<std::vector<uint8_t>> all_grad_handles(world_size_);
    std::vector<std::vector<uint8_t>> all_sync_handles(world_size_);
    std::vector<int64_t> all_recv_offsets_flat(world_size_ * (world_size_ + 1));

    const size_t ro_bytes = static_cast<size_t>(world_size_ + 1) * sizeof(int64_t);

    for (int64_t r = 0; r < world_size_; ++r) {
        const auto &blob = all_rank_metadata[r];
        TORCH_CHECK(blob.size() >= sizeof(IPCRankMetadataHeader),
                    "Metadata blob from rank ", r, " is too small");

        IPCRankMetadataHeader hdr{};
        std::memcpy(&hdr, blob.data(), sizeof(hdr));

        size_t expected_size =
                sizeof(IPCRankMetadataHeader) + 3 * hdr.handle_size + ro_bytes;
        TORCH_CHECK(blob.size() == expected_size, "Metadata blob size mismatch from rank ", r,
                    ": expected ", expected_size, ", got ", blob.size());

        all_num_local[r] = hdr.num_local;
        all_max_f[r] = hdr.max_feature_dim;
        all_slots[r] = hdr.num_slots;

        const uint8_t *p = blob.data() + sizeof(hdr);
        all_x_handles[r].assign(p, p + hdr.handle_size);
        p += hdr.handle_size;
        all_grad_handles[r].assign(p, p + hdr.handle_size);
        p += hdr.handle_size;
        all_sync_handles[r].assign(p, p + hdr.handle_size);
        p += hdr.handle_size;
        std::memcpy(&all_recv_offsets_flat[r * (world_size_ + 1)], p, ro_bytes);
    }

    const int64_t *so = cpu_send_offsets_.data_ptr<int64_t>();
    const int64_t *ro = cpu_recv_offsets_.data_ptr<int64_t>();
    const int64_t *aro = all_recv_offsets_flat.data();

    opened_x_ptrs_.assign(world_size_, nullptr);
    opened_grad_ptrs_.assign(world_size_, nullptr);

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    if (use_fabric_) {
        remote_x_mappings_.assign(world_size_, VmmMapping{});
        remote_grad_mappings_.assign(world_size_, VmmMapping{});

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id_;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        check_cu_call(res, "cuMemGetAllocationGranularity failed");

        for (int64_t r = 0; r < world_size_; ++r) {
            if (r == rank_)
                continue;

            int64_t n_local_r = all_num_local[r];
            int64_t total_recv_r = aro[r * (world_size_ + 1) + world_size_];
            int64_t slots_r = all_slots[r];
            int64_t max_f_r = all_max_f[r];

            // 1. Import x_buffer from rank r
            {
                size_t num_bytes_x = static_cast<size_t>(slots_r) *
                        static_cast<size_t>(std::max<int64_t>(1, n_local_r + total_recv_r)) *
                        max_f_r * sizeof(float);
                size_t padded_size_x =
                        ((num_bytes_x + granularity - 1) / granularity) * granularity;

                CUDA_MEM_FABRIC_HANDLE f_handle;
                TORCH_CHECK(all_x_handles[r].size() == sizeof(f_handle),
                            "Invalid fabric handle size from rank ", r);
                std::memcpy(&f_handle, all_x_handles[r].data(), sizeof(f_handle));

                CUmemGenericAllocationHandle imported_handle;
                res = cuMemImportFromShareableHandle(&imported_handle, (void *)&f_handle,
                                                     CU_MEM_HANDLE_TYPE_FABRIC);
                check_cu_call(res, "cuMemImportFromShareableHandle (x) failed");

                CUdeviceptr remote_vaddr = 0;
                res = cuMemAddressReserve(&remote_vaddr, padded_size_x, 0, 0, 0);
                check_cu_call(res, "cuMemAddressReserve (x) failed");

                res = cuMemMap(remote_vaddr, padded_size_x, 0, imported_handle, 0);
                check_cu_call(res, "cuMemMap (x) failed");

                CUmemAccessDesc access_desc = {};
                access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access_desc.location.id = device_id_;
                access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                res = cuMemSetAccess(remote_vaddr, padded_size_x, &access_desc, 1);
                check_cu_call(res, "cuMemSetAccess (x) failed");

                opened_x_ptrs_[r] = reinterpret_cast<void *>(remote_vaddr);
                remote_x_mappings_[r] = { remote_vaddr, padded_size_x, imported_handle };
            }

            // 2. Import grad_recv_buffer from rank r
            {
                int64_t total_send_r = 0;
                for (int64_t j = 0; j < world_size_; ++j) {
                    int64_t count_r_to_j =
                            aro[j * (world_size_ + 1) + r + 1] - aro[j * (world_size_ + 1) + r];
                    total_send_r += count_r_to_j;
                }
                size_t num_bytes_grad = static_cast<size_t>(slots_r) *
                        static_cast<size_t>(std::max<int64_t>(1, total_send_r)) *
                        max_f_r * sizeof(float);
                size_t padded_size_grad =
                        ((num_bytes_grad + granularity - 1) / granularity) * granularity;

                CUDA_MEM_FABRIC_HANDLE f_handle;
                TORCH_CHECK(all_grad_handles[r].size() == sizeof(f_handle),
                            "Invalid fabric grad handle size from rank ", r);
                std::memcpy(&f_handle, all_grad_handles[r].data(), sizeof(f_handle));

                CUmemGenericAllocationHandle imported_handle;
                res = cuMemImportFromShareableHandle(&imported_handle, (void *)&f_handle,
                                                     CU_MEM_HANDLE_TYPE_FABRIC);
                check_cu_call(res, "cuMemImportFromShareableHandle (grad) failed");

                CUdeviceptr remote_vaddr = 0;
                res = cuMemAddressReserve(&remote_vaddr, padded_size_grad, 0, 0, 0);
                check_cu_call(res, "cuMemAddressReserve (grad) failed");

                res = cuMemMap(remote_vaddr, padded_size_grad, 0, imported_handle, 0);
                check_cu_call(res, "cuMemMap (grad) failed");

                CUmemAccessDesc access_desc = {};
                access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access_desc.location.id = device_id_;
                access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                res = cuMemSetAccess(remote_vaddr, padded_size_grad, &access_desc, 1);
                check_cu_call(res, "cuMemSetAccess (grad) failed");

                opened_grad_ptrs_[r] = reinterpret_cast<void *>(remote_vaddr);
                remote_grad_mappings_[r] = { remote_vaddr, padded_size_grad, imported_handle };
            }
        }
    } else
#endif
    {
        for (int64_t r = 0; r < world_size_; ++r) {
            if (r == rank_)
                continue;

            {
                cudaIpcMemHandle_t h;
                TORCH_CHECK(all_x_handles[r].size() == sizeof(h),
                            "Invalid IPC handle size from rank ", r);
                std::memcpy(&h, all_x_handles[r].data(), sizeof(h));

                void *ptr = nullptr;
                auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
                TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle (x) for rank ", r,
                            " failed: ", cudaGetErrorString(err));
                opened_x_ptrs_[r] = ptr;
            }

            {
                cudaIpcMemHandle_t h;
                TORCH_CHECK(all_grad_handles[r].size() == sizeof(h),
                            "Invalid IPC grad handle size from rank ", r);
                std::memcpy(&h, all_grad_handles[r].data(), sizeof(h));

                void *ptr = nullptr;
                auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
                TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle (grad) for rank ", r,
                            " failed: ", cudaGetErrorString(err));
                opened_grad_ptrs_[r] = ptr;
            }
        }
    }

    // Build forward push targets (with 2-slot stride on remote rank)
    std::vector<PushTarget> fwd_targets_host(world_size_);
    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_) {
            fwd_targets_host[r] = { nullptr, 0, 0, so[r], 0 };
            continue;
        }

        int64_t recv_offset_on_r_from_us = aro[r * (world_size_ + 1) + rank_];
        int64_t n_local_r = all_num_local[r];
        int64_t total_recv_r = aro[r * (world_size_ + 1) + world_size_];
        int64_t x_rows_per_slot_r = std::max<int64_t>(1, n_local_r + total_recv_r);
        int64_t slot_stride_r = (all_slots[r] > 1) ? (x_rows_per_slot_r * all_max_f[r]) : 0;

        fwd_targets_host[r].remote_ptr = static_cast<float *>(opened_x_ptrs_[r]);
        fwd_targets_host[r].slot_stride = slot_stride_r;
        fwd_targets_host[r].remote_offset = n_local_r + recv_offset_on_r_from_us;
        fwd_targets_host[r].send_start = so[r];
        fwd_targets_host[r].send_count = so[r + 1] - so[r];
    }

    // Build backward push targets (with 2-slot stride on remote rank)
    std::vector<PushTarget> bwd_targets_host(world_size_);
    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_) {
            bwd_targets_host[r] = { nullptr, 0, 0, ro[r], 0 };
            continue;
        }

        int64_t offset_in_peer_grad = 0;
        int64_t total_send_r = 0;
        for (int64_t j = 0; j < world_size_; ++j) {
            int64_t count_r_to_j =
                    aro[j * (world_size_ + 1) + r + 1] - aro[j * (world_size_ + 1) + r];
            if (j < rank_)
                offset_in_peer_grad += count_r_to_j;
            total_send_r += count_r_to_j;
        }
        int64_t grad_rows_per_slot_r = std::max<int64_t>(1, total_send_r);
        int64_t slot_stride_r = (all_slots[r] > 1) ? (grad_rows_per_slot_r * all_max_f[r]) : 0;

        bwd_targets_host[r].remote_ptr = static_cast<float *>(opened_grad_ptrs_[r]);
        bwd_targets_host[r].slot_stride = slot_stride_r;
        bwd_targets_host[r].remote_offset = offset_in_peer_grad;
        bwd_targets_host[r].send_start = ro[r];
        bwd_targets_host[r].send_count = ro[r + 1] - ro[r];
    }

    auto device = torch::Device(torch::kCUDA, device_id_);
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    fwd_targets_ =
            torch::empty({ static_cast<int64_t>(sizeof(PushTarget) * world_size_) }, byte_opts);
    {
        auto err_cpy = cudaMemcpy(fwd_targets_.data_ptr(), fwd_targets_host.data(),
                                  sizeof(PushTarget) * world_size_, cudaMemcpyHostToDevice);
        TORCH_CHECK(err_cpy == cudaSuccess,
                    "cudaMemcpy fwd_targets failed: ", cudaGetErrorString(err_cpy));
    }

    bwd_targets_ =
            torch::empty({ static_cast<int64_t>(sizeof(PushTarget) * world_size_) }, byte_opts);
    {
        auto err_cpy = cudaMemcpy(bwd_targets_.data_ptr(), bwd_targets_host.data(),
                                  sizeof(PushTarget) * world_size_, cudaMemcpyHostToDevice);
        TORCH_CHECK(err_cpy == cudaSuccess,
                    "cudaMemcpy bwd_targets failed: ", cudaGetErrorString(err_cpy));
    }

    if (total_send_ > 0) {
        auto buf_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
        staging_buffer_ = torch::empty({ total_send_, max_feature_dim_ }, buf_opts);
    }

    ce_fwd_targets_ = fwd_targets_host;
    is_setup_ = true;

    setup_sync(all_sync_handles);

    cudaSetDevice(prev_device);
}

// ---------------------------------------------------------------------------
// Direct / legacy setup (retained for standalone IPCExchanger tests)
// ---------------------------------------------------------------------------

void IPCExchanger::setup(torch::Tensor x_buffer, torch::Tensor grad_recv_buffer,
                         const std::vector<std::vector<uint8_t>> &all_x_handles,
                         const std::vector<std::vector<uint8_t>> &all_grad_handles,
                         torch::Tensor send_offsets, torch::Tensor send_local_indices,
                         torch::Tensor recv_offsets, torch::Tensor all_recv_offsets,
                         torch::Tensor all_num_local, int64_t num_local, int64_t feature_dim,
                         int64_t rank, int64_t world_size)
{
    if (is_setup_)
        teardown();

    rank_ = rank;
    world_size_ = world_size;
    max_feature_dim_ = feature_dim;
    num_local_ = num_local;
    num_slots_ = 1;

    device_id_ = x_buffer.device().index();
    use_fabric_ = supports_fabric_handles(device_id_);

    auto cpu_send_offsets = send_offsets.to(torch::kCPU).contiguous();
    auto cpu_recv_offsets = recv_offsets.to(torch::kCPU).contiguous();
    auto cpu_all_recv_offsets = all_recv_offsets.to(torch::kCPU).contiguous();
    auto cpu_all_num_local = all_num_local.to(torch::kCPU).contiguous();

    send_offsets_ = send_offsets.to(x_buffer.device()).contiguous();
    send_local_indices_ = send_local_indices.to(x_buffer.device()).contiguous();
    recv_offsets_ = recv_offsets.to(x_buffer.device()).contiguous();

    auto *so = cpu_send_offsets.data_ptr<int64_t>();
    auto *ro = cpu_recv_offsets.data_ptr<int64_t>();
    auto *aro = cpu_all_recv_offsets.data_ptr<int64_t>();
    auto *anl = cpu_all_num_local.data_ptr<int64_t>();

    total_send_ = so[world_size];
    total_recv_ = ro[world_size];

    opened_x_ptrs_.assign(world_size, nullptr);
    opened_grad_ptrs_.assign(world_size, nullptr);

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    if (use_fabric_) {
        remote_x_mappings_.assign(world_size, VmmMapping{});
        remote_grad_mappings_.assign(world_size, VmmMapping{});

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id_;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        check_cu_call(res, "cuMemGetAllocationGranularity failed");

        for (int64_t r = 0; r < world_size; ++r) {
            if (r == rank)
                continue;

            {
                int64_t n_local_r = anl[r];
                int64_t total_recv_r = aro[r * (world_size + 1) + world_size];
                size_t num_bytes_x =
                        static_cast<size_t>(std::max<int64_t>(1, n_local_r + total_recv_r)) *
                        max_feature_dim_ * sizeof(float);
                size_t padded_size_x =
                        ((num_bytes_x + granularity - 1) / granularity) * granularity;

                CUDA_MEM_FABRIC_HANDLE f_handle;
                TORCH_CHECK(all_x_handles[r].size() == sizeof(f_handle),
                            "Invalid fabric handle size from rank ", r);
                std::memcpy(&f_handle, all_x_handles[r].data(), sizeof(f_handle));

                CUmemGenericAllocationHandle imported_handle;
                res = cuMemImportFromShareableHandle(&imported_handle, (void *)&f_handle,
                                                     CU_MEM_HANDLE_TYPE_FABRIC);
                check_cu_call(res, "cuMemImportFromShareableHandle (x) failed");

                CUdeviceptr remote_vaddr = 0;
                res = cuMemAddressReserve(&remote_vaddr, padded_size_x, 0, 0, 0);
                check_cu_call(res, "cuMemAddressReserve (x) failed");

                res = cuMemMap(remote_vaddr, padded_size_x, 0, imported_handle, 0);
                check_cu_call(res, "cuMemMap (x) failed");

                CUmemAccessDesc access_desc = {};
                access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access_desc.location.id = device_id_;
                access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                res = cuMemSetAccess(remote_vaddr, padded_size_x, &access_desc, 1);
                check_cu_call(res, "cuMemSetAccess (x) failed");

                opened_x_ptrs_[r] = reinterpret_cast<void *>(remote_vaddr);
                remote_x_mappings_[r] = { remote_vaddr, padded_size_x, imported_handle };
            }

            {
                int64_t total_send_r = 0;
                for (int64_t j = 0; j < world_size; ++j) {
                    int64_t count_r_to_j =
                            aro[j * (world_size + 1) + r + 1] - aro[j * (world_size + 1) + r];
                    total_send_r += count_r_to_j;
                }
                size_t num_bytes_grad = static_cast<size_t>(std::max<int64_t>(1, total_send_r)) *
                        max_feature_dim_ * sizeof(float);
                size_t padded_size_grad =
                        ((num_bytes_grad + granularity - 1) / granularity) * granularity;

                CUDA_MEM_FABRIC_HANDLE f_handle;
                TORCH_CHECK(all_grad_handles[r].size() == sizeof(f_handle),
                            "Invalid fabric grad handle size from rank ", r);
                std::memcpy(&f_handle, all_grad_handles[r].data(), sizeof(f_handle));

                CUmemGenericAllocationHandle imported_handle;
                res = cuMemImportFromShareableHandle(&imported_handle, (void *)&f_handle,
                                                     CU_MEM_HANDLE_TYPE_FABRIC);
                check_cu_call(res, "cuMemImportFromShareableHandle (grad) failed");

                CUdeviceptr remote_vaddr = 0;
                res = cuMemAddressReserve(&remote_vaddr, padded_size_grad, 0, 0, 0);
                check_cu_call(res, "cuMemAddressReserve (grad) failed");

                res = cuMemMap(remote_vaddr, padded_size_grad, 0, imported_handle, 0);
                check_cu_call(res, "cuMemMap (grad) failed");

                CUmemAccessDesc access_desc = {};
                access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access_desc.location.id = device_id_;
                access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                res = cuMemSetAccess(remote_vaddr, padded_size_grad, &access_desc, 1);
                check_cu_call(res, "cuMemSetAccess (grad) failed");

                opened_grad_ptrs_[r] = reinterpret_cast<void *>(remote_vaddr);
                remote_grad_mappings_[r] = { remote_vaddr, padded_size_grad, imported_handle };
            }
        }
    } else
#endif
    {
        for (int64_t r = 0; r < world_size; ++r) {
            if (r == rank)
                continue;

            {
                cudaIpcMemHandle_t h;
                TORCH_CHECK(all_x_handles[r].size() == sizeof(h),
                            "Invalid IPC handle size from rank ", r);
                std::memcpy(&h, all_x_handles[r].data(), sizeof(h));

                void *ptr = nullptr;
                auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
                TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle (x) for rank ", r,
                            " failed: ", cudaGetErrorString(err));
                opened_x_ptrs_[r] = ptr;
            }

            {
                cudaIpcMemHandle_t h;
                TORCH_CHECK(all_grad_handles[r].size() == sizeof(h),
                            "Invalid IPC grad handle size from rank ", r);
                std::memcpy(&h, all_grad_handles[r].data(), sizeof(h));

                void *ptr = nullptr;
                auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
                TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle (grad) for rank ", r,
                            " failed: ", cudaGetErrorString(err));
                opened_grad_ptrs_[r] = ptr;
            }
        }
    }

    std::vector<PushTarget> fwd_targets_host(world_size);
    for (int64_t r = 0; r < world_size; ++r) {
        if (r == rank) {
            fwd_targets_host[r] = { nullptr, 0, 0, so[r], 0 };
            continue;
        }

        int64_t recv_offset_on_r_from_us = aro[r * (world_size + 1) + rank];
        int64_t n_local_r = anl[r];

        fwd_targets_host[r].remote_ptr = static_cast<float *>(opened_x_ptrs_[r]);
        fwd_targets_host[r].slot_stride = 0;
        fwd_targets_host[r].remote_offset = n_local_r + recv_offset_on_r_from_us;
        fwd_targets_host[r].send_start = so[r];
        fwd_targets_host[r].send_count = so[r + 1] - so[r];
    }

    std::vector<PushTarget> bwd_targets_host(world_size);
    for (int64_t r = 0; r < world_size; ++r) {
        if (r == rank) {
            bwd_targets_host[r] = { nullptr, 0, 0, ro[r], 0 };
            continue;
        }

        int64_t offset_in_peer_grad = 0;
        for (int64_t j = 0; j < rank_; ++j) {
            int64_t count_r_to_j =
                    aro[j * (world_size + 1) + r + 1] - aro[j * (world_size + 1) + r];
            offset_in_peer_grad += count_r_to_j;
        }

        bwd_targets_host[r].remote_ptr = static_cast<float *>(opened_grad_ptrs_[r]);
        bwd_targets_host[r].slot_stride = 0;
        bwd_targets_host[r].remote_offset = offset_in_peer_grad;
        bwd_targets_host[r].send_start = ro[r];
        bwd_targets_host[r].send_count = ro[r + 1] - ro[r];
    }

    auto device = x_buffer.device();
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    fwd_targets_ =
            torch::empty({ static_cast<int64_t>(sizeof(PushTarget) * world_size) }, byte_opts);
    {
        auto err_cpy = cudaMemcpy(fwd_targets_.data_ptr(), fwd_targets_host.data(),
                                  sizeof(PushTarget) * world_size, cudaMemcpyHostToDevice);
        TORCH_CHECK(err_cpy == cudaSuccess,
                    "cudaMemcpy fwd_targets failed: ", cudaGetErrorString(err_cpy));
    }

    bwd_targets_ =
            torch::empty({ static_cast<int64_t>(sizeof(PushTarget) * world_size) }, byte_opts);
    {
        auto err_cpy = cudaMemcpy(bwd_targets_.data_ptr(), bwd_targets_host.data(),
                                  sizeof(PushTarget) * world_size, cudaMemcpyHostToDevice);
        TORCH_CHECK(err_cpy == cudaSuccess,
                    "cudaMemcpy bwd_targets failed: ", cudaGetErrorString(err_cpy));
    }

    if (total_send_ > 0) {
        auto buf_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
        staging_buffer_ = torch::empty({ total_send_, max_feature_dim_ }, buf_opts);
    }

    ce_fwd_targets_ = fwd_targets_host;
    is_setup_ = true;
}

// ---------------------------------------------------------------------------
// Push Forward / Backward
// ---------------------------------------------------------------------------

void IPCExchanger::launch_forward_impl(torch::Tensor x_slot_view, int64_t feature_dim,
                                       int64_t slot, cudaStream_t stream,
                                       int64_t ce_threshold_bytes)
{
    TORCH_CHECK(is_setup_, "IPCExchanger::push_forward: not set up");
    TORCH_CHECK(x_slot_view.is_cuda() && x_slot_view.is_contiguous(),
                "x_buffer must be contiguous CUDA tensor");
    TORCH_CHECK(feature_dim > 0 && feature_dim <= max_feature_dim_,
                "feature_dim (", feature_dim, ") must be in [1, ", max_feature_dim_, "]");

    if (total_send_ == 0)
        return;

    if (ce_threshold_bytes <= 0) {
        launch_gather_and_push(x_slot_view, send_offsets_, send_local_indices_, fwd_targets_,
                               total_send_, feature_dim, world_size_, slot, stream);
        return;
    }

    bool use_ce = false;
    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_)
            continue;
        int64_t bytes = ce_fwd_targets_[r].send_count * feature_dim * sizeof(float);
        if (bytes >= ce_threshold_bytes) {
            use_ce = true;
            break;
        }
    }

    if (!use_ce) {
        launch_gather_and_push(x_slot_view, send_offsets_, send_local_indices_, fwd_targets_,
                               total_send_, feature_dim, world_size_, slot, stream);
        return;
    }

    launch_gather_to_staging(x_slot_view, send_local_indices_, staging_buffer_, total_send_,
                             feature_dim, stream);

    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_)
            continue;

        const auto &tgt = ce_fwd_targets_[r];
        if (tgt.send_count == 0)
            continue;

        int64_t bytes = tgt.send_count * feature_dim * sizeof(float);
        float *src = staging_buffer_.data_ptr<float>() + tgt.send_start * feature_dim;
        float *dst = (tgt.remote_ptr + slot * tgt.slot_stride) + tgt.remote_offset * feature_dim;

        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
    }
}

void IPCExchanger::push_forward_slot(torch::Tensor x_slot_view, int64_t feature_dim, int64_t slot,
                                     cudaStream_t stream, int64_t ce_threshold_bytes)
{
    launch_forward_impl(x_slot_view, feature_dim, slot, stream, ce_threshold_bytes);
    record_push_complete(stream);
    wait_for_pushes(stream);
}

void IPCExchanger::push_backward_slot(torch::Tensor grad_output, int64_t feature_dim, int64_t slot,
                                      cudaStream_t stream)
{
    TORCH_CHECK(is_setup_, "IPCExchanger::push_backward: not set up");
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "grad tensor must be contiguous CUDA tensor");
    TORCH_CHECK(feature_dim > 0 && feature_dim <= max_feature_dim_,
                "feature_dim (", feature_dim, ") must be in [1, ", max_feature_dim_, "]");

    launch_gather_ghost_grad_and_push(grad_output, recv_offsets_, bwd_targets_, num_local_,
                                      total_recv_, feature_dim, world_size_, slot, stream);
    record_push_complete(stream);
    wait_for_pushes(stream);
}

void IPCExchanger::push_forward(torch::Tensor x_buffer, cudaStream_t stream,
                                int64_t ce_threshold_bytes)
{
    int64_t f = x_buffer.dim() >= 2 ? x_buffer.size(1) : max_feature_dim_;
    launch_forward_impl(x_buffer, f, 0, stream, ce_threshold_bytes);
}

void IPCExchanger::push_backward(torch::Tensor grad_at_ghost, cudaStream_t stream)
{
    TORCH_CHECK(is_setup_, "IPCExchanger::push_backward: not set up");
    TORCH_CHECK(grad_at_ghost.is_cuda() && grad_at_ghost.is_contiguous(),
                "grad tensor must be contiguous CUDA tensor");
    int64_t f = grad_at_ghost.dim() >= 2 ? grad_at_ghost.size(1) : max_feature_dim_;
    launch_gather_ghost_grad_and_push(grad_at_ghost, recv_offsets_, bwd_targets_, num_local_,
                                      total_recv_, f, world_size_, 0, stream);
}

// --- IPC / Fabric push synchronization (generation counter) ---

std::vector<uint8_t> IPCExchanger::create_and_export_sync_handle()
{
    if (local_counter_ || local_counter_vaddr_) {
        if (sync_use_fabric_) {
#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
            if (local_counter_vaddr_) {
                cuMemUnmap(local_counter_vaddr_, local_counter_padded_size_);
                cuMemRelease(local_counter_handle_);
                cuMemAddressFree(local_counter_vaddr_, local_counter_padded_size_);
                local_counter_vaddr_ = 0;
                local_counter_padded_size_ = 0;
                local_counter_handle_ = 0;
            }
#endif
        } else {
            if (local_counter_) {
                cudaFree(local_counter_);
            }
        }
        local_counter_ = nullptr;
    }
    call_counter_ = 0;
    sync_word_ptrs_.reset();

    int device_id = 0;
    auto err_dev = cudaGetDevice(&device_id);
    TORCH_CHECK(err_dev == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err_dev));
    sync_use_fabric_ = supports_fabric_handles(device_id);

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    if (sync_use_fabric_) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        check_cu_call(res, "cuMemGetAllocationGranularity (sync) failed");

        size_t padded_size = ((kSyncBytes + granularity - 1) / granularity) * granularity;
        local_counter_padded_size_ = padded_size;

        res = cuMemCreate(&local_counter_handle_, padded_size, &prop, 0);
        check_cu_call(res, "cuMemCreate (sync) failed");

        CUDA_MEM_FABRIC_HANDLE fabric_handle;
        std::memset(&fabric_handle, 0, sizeof(fabric_handle));
        res = cuMemExportToShareableHandle(&fabric_handle, local_counter_handle_,
                                           CU_MEM_HANDLE_TYPE_FABRIC, 0);
        check_cu_call(res, "cuMemExportToShareableHandle (sync) failed");

        res = cuMemAddressReserve(&local_counter_vaddr_, padded_size, 0, 0, 0);
        check_cu_call(res, "cuMemAddressReserve (sync) failed");

        res = cuMemMap(local_counter_vaddr_, padded_size, 0, local_counter_handle_, 0);
        check_cu_call(res, "cuMemMap (sync) failed");

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_id;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        res = cuMemSetAccess(local_counter_vaddr_, padded_size, &access_desc, 1);
        check_cu_call(res, "cuMemSetAccess (sync) failed");

        local_counter_ = reinterpret_cast<uint32_t *>(local_counter_vaddr_);
        auto err_set = cudaMemset(local_counter_, 0, kSyncBytes);
        TORCH_CHECK(err_set == cudaSuccess,
                    "cudaMemset (sync) failed: ", cudaGetErrorString(err_set));

        std::vector<uint8_t> bytes(sizeof(CUDA_MEM_FABRIC_HANDLE));
        std::memcpy(bytes.data(), &fabric_handle, sizeof(CUDA_MEM_FABRIC_HANDLE));
        return bytes;
    }
#endif

    auto err = cudaMalloc(&local_counter_, kSyncBytes);
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc (counter) failed: ", cudaGetErrorString(err));
    err = cudaMemset(local_counter_, 0, kSyncBytes);
    TORCH_CHECK(err == cudaSuccess, "cudaMemset (counter) failed: ", cudaGetErrorString(err));

    cudaIpcMemHandle_t handle;
    err = cudaIpcGetMemHandle(&handle, local_counter_);
    TORCH_CHECK(err == cudaSuccess,
                "cudaIpcGetMemHandle (counter) failed: ", cudaGetErrorString(err));

    std::vector<uint8_t> bytes(sizeof(handle));
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
}

void IPCExchanger::setup_sync(const std::vector<std::vector<uint8_t>> &all_sync_handles)
{
    TORCH_CHECK(is_setup_, "setup_sync: must call setup() first");
    TORCH_CHECK(local_counter_ != nullptr || local_counter_vaddr_ != 0,
                "setup_sync: must call create_and_export_sync_handle() first");

    if (sync_use_fabric_) {
#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
        for (auto &mapping : remote_sync_mappings_) {
            if (mapping.vaddr) {
                cuMemUnmap(mapping.vaddr, mapping.padded_size);
                cuMemRelease(mapping.handle);
                cuMemAddressFree(mapping.vaddr, mapping.padded_size);
            }
        }
        remote_sync_mappings_.clear();
#endif
    } else {
        for (auto &p : remote_counters_) {
            if (p)
                cudaIpcCloseMemHandle(reinterpret_cast<void *>(p));
        }
    }
    remote_counters_.assign(world_size_, 0);

    int device_id = 0;
    cudaGetDevice(&device_id);

#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
    if (sync_use_fabric_) {
        remote_sync_mappings_.assign(world_size_, VmmMapping{});

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        size_t granularity = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        check_cu_call(res, "cuMemGetAllocationGranularity (sync) failed");

        size_t padded_size = ((kSyncBytes + granularity - 1) / granularity) * granularity;

        for (int64_t r = 0; r < world_size_; ++r) {
            if (r == rank_)
                continue;

            CUDA_MEM_FABRIC_HANDLE f_handle;
            TORCH_CHECK(all_sync_handles[r].size() == sizeof(f_handle),
                        "Invalid fabric counter handle size from rank ", r);
            std::memcpy(&f_handle, all_sync_handles[r].data(), sizeof(f_handle));

            CUmemGenericAllocationHandle imported_handle;
            res = cuMemImportFromShareableHandle(&imported_handle, (void *)&f_handle,
                                                 CU_MEM_HANDLE_TYPE_FABRIC);
            check_cu_call(res, "cuMemImportFromShareableHandle (counter) failed");

            CUdeviceptr remote_vaddr = 0;
            res = cuMemAddressReserve(&remote_vaddr, padded_size, 0, 0, 0);
            check_cu_call(res, "cuMemAddressReserve (counter) failed");

            res = cuMemMap(remote_vaddr, padded_size, 0, imported_handle, 0);
            check_cu_call(res, "cuMemMap (counter) failed");

            CUmemAccessDesc access_desc = {};
            access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device_id;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            res = cuMemSetAccess(remote_vaddr, padded_size, &access_desc, 1);
            check_cu_call(res, "cuMemSetAccess (counter) failed");

            remote_counters_[r] = remote_vaddr;
            remote_sync_mappings_[r] = { remote_vaddr, padded_size, imported_handle };
        }

        build_sync_word_ptrs();
        sync_setup_ = true;
        return;
    }
#endif

    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_)
            continue;

        cudaIpcMemHandle_t h;
        TORCH_CHECK(all_sync_handles[r].size() == sizeof(h),
                    "Invalid IPC counter handle size from rank ", r);
        std::memcpy(&h, all_sync_handles[r].data(), sizeof(h));

        void *ptr = nullptr;
        auto err = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
        TORCH_CHECK(err == cudaSuccess, "cudaIpcOpenMemHandle (counter) for rank ", r,
                    " failed: ", cudaGetErrorString(err));
        remote_counters_[r] = reinterpret_cast<CUdeviceptr>(ptr);
    }

    build_sync_word_ptrs();
    sync_setup_ = true;
}

void IPCExchanger::record_push_complete(cudaStream_t stream)
{
    TORCH_CHECK(sync_setup_, "record_push_complete: counter not set up");
    ++call_counter_;
    if (watchdog_local_.defined()) {
        uint32_t *slot = local_counter_ + 1 + (call_counter_ & 1u);
        auto err = cudaMemcpyAsync(slot, watchdog_local_.data_ptr(), sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice, stream);
        TORCH_CHECK(err == cudaSuccess, "watchdog flag write failed: ", cudaGetErrorString(err));
    }
    CUresult err =
            cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                 reinterpret_cast<CUdeviceptr>(local_counter_), call_counter_, 0);
    if (err != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(err, &msg);
        TORCH_CHECK(false, "cuStreamWriteValue32 failed: ", (msg ? msg : "(unknown)"));
    }
}

void IPCExchanger::wait_for_pushes(cudaStream_t stream)
{
    TORCH_CHECK(sync_setup_, "wait_for_pushes: counter not set up");
    for (int64_t r = 0; r < world_size_; ++r) {
        if (r == rank_ || !remote_counters_[r])
            continue;
        CUresult err = cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), remote_counters_[r],
                                           call_counter_, CU_STREAM_WAIT_VALUE_GEQ);
        if (err != CUDA_SUCCESS) {
            const char *msg = nullptr;
            cuGetErrorString(err, &msg);
            TORCH_CHECK(false, "cuStreamWaitValue32 for rank ", r,
                        " failed: ", (msg ? msg : "(unknown)"));
        }
    }
    if (watchdog_global_.defined() && sync_word_ptrs_.defined())
        launch_watchdog_or(sync_word_ptrs_, world_size_, call_counter_ & 1u, watchdog_global_,
                           stream);
}

void IPCExchanger::set_watchdog_flags(torch::Tensor local_flag, torch::Tensor global_flag)
{
    for (const auto *t : { &local_flag, &global_flag }) {
        TORCH_CHECK(t->is_cuda(), "set_watchdog_flags: flags must be CUDA tensors");
        TORCH_CHECK(t->scalar_type() == torch::kInt32 && t->numel() == 1 && t->is_contiguous(),
                    "set_watchdog_flags: flags must be contiguous int32 tensors with one element");
    }
    watchdog_local_ = local_flag;
    watchdog_global_ = global_flag;
}

void IPCExchanger::clear_watchdog_flags()
{
    watchdog_local_.reset();
    watchdog_global_.reset();
}

void IPCExchanger::build_sync_word_ptrs()
{
    std::vector<int64_t> ptrs(world_size_, 0);
    for (int64_t r = 0; r < world_size_; ++r)
        ptrs[r] = (r == rank_) ? reinterpret_cast<int64_t>(local_counter_)
                               : static_cast<int64_t>(remote_counters_[r]);
    int device_id = 0;
    cudaGetDevice(&device_id);
    sync_word_ptrs_ = torch::tensor(ptrs, torch::dtype(torch::kInt64))
                              .to(torch::Device(torch::kCUDA, device_id));
}

void IPCExchanger::teardown()
{
    if (sync_use_fabric_) {
#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
        for (auto &mapping : remote_sync_mappings_) {
            if (mapping.vaddr) {
                cuMemUnmap(mapping.vaddr, mapping.padded_size);
                cuMemRelease(mapping.handle);
                cuMemAddressFree(mapping.vaddr, mapping.padded_size);
            }
        }
        remote_sync_mappings_.clear();

        if (local_counter_vaddr_) {
            cuMemUnmap(local_counter_vaddr_, local_counter_padded_size_);
            cuMemRelease(local_counter_handle_);
            cuMemAddressFree(local_counter_vaddr_, local_counter_padded_size_);
            local_counter_vaddr_ = 0;
            local_counter_padded_size_ = 0;
            local_counter_handle_ = 0;
        }
#endif
    } else {
        if (local_counter_) {
            cudaFree(local_counter_);
        }
        for (auto &p : remote_counters_) {
            if (p) {
                cudaIpcCloseMemHandle(reinterpret_cast<void *>(p));
                p = 0;
            }
        }
    }

    local_counter_ = nullptr;
    remote_counters_.clear();
    sync_word_ptrs_.reset();
    call_counter_ = 0;
    sync_setup_ = false;
    sync_use_fabric_ = false;

    if (!is_setup_)
        return;

    if (use_fabric_) {
#if defined(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) && defined(CU_MEM_HANDLE_TYPE_FABRIC)
        for (auto &mapping : remote_x_mappings_) {
            if (mapping.vaddr) {
                cuMemUnmap(mapping.vaddr, mapping.padded_size);
                cuMemRelease(mapping.handle);
                cuMemAddressFree(mapping.vaddr, mapping.padded_size);
            }
        }
        remote_x_mappings_.clear();

        for (auto &mapping : remote_grad_mappings_) {
            if (mapping.vaddr) {
                cuMemUnmap(mapping.vaddr, mapping.padded_size);
                cuMemRelease(mapping.handle);
                cuMemAddressFree(mapping.vaddr, mapping.padded_size);
            }
        }
        remote_grad_mappings_.clear();
#endif
    } else {
        for (int64_t r = 0; r < world_size_; ++r) {
            if (r == rank_)
                continue;
            if (opened_x_ptrs_[r]) {
                cudaIpcCloseMemHandle(opened_x_ptrs_[r]);
                opened_x_ptrs_[r] = nullptr;
            }
            if (opened_grad_ptrs_[r]) {
                cudaIpcCloseMemHandle(opened_grad_ptrs_[r]);
                opened_grad_ptrs_[r] = nullptr;
            }
        }
    }

    opened_x_ptrs_.clear();
    opened_grad_ptrs_.clear();

    staging_buffer_.reset();
    ce_fwd_targets_.clear();
    fwd_targets_.reset();
    bwd_targets_.reset();
    cpu_send_offsets_.reset();
    cpu_recv_offsets_.reset();

    is_setup_ = false;
    use_fabric_ = false;
}

} // namespace dynamis_model
