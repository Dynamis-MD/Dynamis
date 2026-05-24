# `FeatureExchanger` Developer and Design Document

The `FeatureExchanger` is the high-performance, low-latency custom low-level communication engine written specifically for the per-layer feature exchange for `DistNNPModel`s. This documentation is for the C++ implementation of the underlying communication engine. The purpose of these docs are to enable developers write new communication backends or understand the underlying feature exchange happening on the Python frontend. 

## `FeatureExchanger` - Communication Engine

Declared in `include/distnnp_model/exchanger.hpp`, implemented in `src/crc/exchanger.cu`. Exposed to Python via pybind11. 

### Role

The `FeatureExchanger` is the abstract class that defines the communication primitives and setup required to interface with the Python front end and supported `DistNNPModel`s. 

### Public API

The public API for the `FeatureExchanger` is frozen and is the contract that interfaces with the distributed models `(DistNNPModel)`. Users implementing new models are expected to use this public interface. Backend developers are expected to develop communication backends that conform to this contract.

#### FeatureExchange::setup

The `setup` method is performed to set up the necessary network topology structures, communication patterns, buffers, and memory handles. It is expected to be called at the simulation initialization. 

```C++
void setup(const DistEnvironment &dist_env,
           const PartitionMessagePassingInfo &partition_comm_info,
           const size_t max_feature_dim);
```
The `max_feature_dim` is the largest expected size of each feature vector for an atom. This is dependendent on the particular model being distributed. 

#### FeatureExchange::reset

Resets the internal buffers. This is used during the simulation after repartitioning. 

```C++
void reset(const PartitionMessagePassingInfo &partition_comm_info,
           const size_t max_feature_dim);
```

#### FeatureExchange::push_feature_forward

The `push_feature_forward` method used during the `forward` phase of the force calculation. Local features are passed in and the output features are updated in place.

```C++
void feature_push_forward(
      torch::Tensor return_features, const torch::Tensor local_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info);
```

If the feature dimension of `return_features` or `local_features` is greater than the `max_feature_dim`, that is undefined behavior. Depending on the implementation, that would raise an error, cause memory errors, etc.  

#### FeatureExchange.push_feature_grad
The `push_feature_grad` method used during the `backward` phase of the force calculation. 

```C++
void feature_push_grad(
      torch::Tensor grad_local_features,
      const torch::Tensor grad_output_features,
      const distnnp_state::PartitionMessagePassingInfo &partition_info);
```

Same as `feature_push_forward` if the passed tensors are larger than the predefined `max_feature_dim`, it will cause undefined behavior. 


> Note: The FeatureExchange provides to different methods for exchanging informated, one for the forward phase, and one for the reverse phase. This is due to the couplped nature of exchange pattern with autograd. Any received data has to be sent back to the sender. FeatureExchangers should pre-calculate the sends required during backprop.
> We don't expose a general purpose `put` intentionally. Specialization allows us to purpose optimizations over the costly communication process and reduce memory overhead. While the primitives may not work for all models, it is expected that additional functionality is added via subclass-ing.   


# NVFeatureExchanger

The `NVFeatureExchanger` wraps the `P2PExchanger` and `RDMAExchanger` exchange code under a single public interface. All communication should go through this class. The user is not expected to decide whether to use `P2PExchanger` or `RDMAExchanger` exchange given some send request. It is the reponsibility of the `NVFeatureExchanger` to route communication request through the appropriate method based on the hardware topology. 

## P2PExchanger (Intra-Node)

The P2P exchanger is incharge of communicating data between peer-connected GPUs. Peer connected GPUs are GPUs connected via PCI-E or NVLink. The communication is performed using the `CUDA IPC` mechanism.

### Intra-node: Copy Engine: GPUDirect P2P (NVLink / PCIe)

Large transfers between GPUs are performed using the DMA Engine. The data is sent via `cudaMemcpyPeerAsync`. Completion is guaranteed using a low-latency poll on each receiver.  

### Intra-node: SM Kernel: GPUDirect P2P (NVLink / PCIe)

For small transferes, data is transferred through a data movement kernel with from one peer-acessible memory location to another.

> The P2PExchanger `setup` creates the required peer accessible buffers and enables peer access on visible devices.  

**Mechanism:**
1. At setup, enable peer-to-peer access: `cudaDeviceEnablePeerAccess(peer_device, 0)` for each neighbour GPU on the same node. This makes the peer GPU's memory directly addressable from the local GPU.
2. Allocate persistent send and recv GPU buffers (sized for `max_send_atoms × max_feature_dim`).
3. At each exchange:
   a. `push_buffer_create` gathers send atoms into the local send buffer (on the local GPU, compute stream).
   b. `cudaMemcpyAsync(peer_recv_buffer, local_send_buffer, ..., cudaMemcpyDeviceToDevice, stream)` — the copy engine moves data directly. On NVLink this saturates high-bandwidth peer links; on PCIe it goes through the host bridge but still avoids CPU copies.
   c. A CUDA event is recorded on the stream and signalled to the peer to indicate the write is complete.

**NVLink vs PCIe:** Both use the same API. NVLink provides ~600 GB/s aggregate per A100, whereas PCIe 4.0 ×16 provides ~32 GB/s. The halo volume at ~4,000 atoms with 128-dimensional embeddings is small enough (~2 MB per step) that both interconnects complete well within one timestep.

## RDMAExchanger (Inter-Node)

For multi-node setup with GPUs connected via high-speed interconnect such as Infiniband, Slingshot, or RoCE is handled by the `RDMAExchanger`. The RDMA setup initializes the required endponts, completions queues, address vectors, and device buffers.  

### Inter-node: LibFabric: GPUDirect RDMA (Infiniband / Slingshot)

We use `libfabric` to implement GPUDirect-enabled one-sided RDMA. `libfabric` must be built with `ibverbs` (for Infiniband) or `cxi` (for Slingshot), and `HMEM` (heterogenous memory) support. 

**One-sided RDMA WRITE:** The sender reads from its registered GPU send buffer and writes directly into the receiver's registered GPU recv buffer. The receiver's CPU is not involved; only the RDMA NIC and GPU DMA engine participate.

**Setup (once at `FeatureExchanger` initialisation):**

1. **Memory registration** — For each communicating rank pair, register the send and recv GPU buffers with the RDMA device:
   ```
   ibv_mr* mr = ibv_reg_mr(pd, gpu_buffer_ptr, size, IBV_ACCESS_* | GDR_flags)
   ```
   The GDR (GPUDirect) flags expose the GPU buffer to the NIC's DMA engine. Registered buffers must remain allocated and pinned for the lifetime of the `FeatureExchanger`.

2. **Address exchange** — Each rank broadcasts its registered buffer address (`mr->addr`) and remote key (`mr->rkey`) to its communicating peers. This is the only collective operation needed at setup time; it uses `torch.distributed.all_gather` over a CPU tensor.

3. **Queue Pair (QP) creation** — Create one RC (Reliable Connection) QP per communicating rank pair. Only pairs with `global_send_recv_matrix[r, r'] > 0` need a QP. Fewer QPs → lower memory overhead and faster setup.

4. **QP state machine** — Transition each QP through RESET → INIT → RTR → RTS using the exchanged address handles (LID / GID for IB/RoCE).

**Per-step exchange:**

1. `push_buffer_create` packs the send atoms into the registered GPU send buffer (compute stream).
2. A CUDA event is recorded on the compute stream and waited on by the RDMA stream, serialising the gather before the transfer.
3. `ibv_post_send` with `IBV_WR_RDMA_WRITE` is posted on the RDMA stream. The NIC reads from the local registered buffer and writes directly into the remote peer's registered recv buffer (identified by remote `addr` + `rkey`).
4. A completion signal (small RDMA SEND/RECV doorbell) notifies the receiver that the write is complete. The receiver waits on this signal before reading from its recv buffer.
5. A CUDA event on the receiver side gates the subsequent scatter step.

**Buffer sizing:** Persistent buffers are allocated at setup time for `max_atoms_per_rank × max_feature_dim` floats. Because feature dimensions vary per GNN layer, the max over all layers is used. This trades memory for avoiding per-step buffer allocation and re-registration (re-registration is expensive — it triggers a TLB shootdown on the GPU).

**Synchronisation model:**

```
Sender                                Receiver
──────                                ────────
[compute stream]                      [compute stream]
  push_buffer_create (gather)
  record CUDA event E_gather

[RDMA stream]
  wait on E_gather
  ibv_post_send (RDMA WRITE)
  ibv_post_send (doorbell SEND)
                                      [RDMA stream]
                                        ibv_post_recv (doorbell RECV) ← blocks until doorbell arrives
                                        record CUDA event E_recv

                                      [compute stream]
                                        wait on E_recv
                                        scatter recv_buffer → ghost slots
                                        (continue NNP forward pass)
```