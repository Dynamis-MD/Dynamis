# `distnnp_model` Design Document

`distnnp_model` is the force-computation module of the DistNNP framework. It defines:

1. The **`DistNNPModel` ABC** — the contract every NNP implementation must satisfy.
2. The **`FeaturePushToExchange` autograd Function** — the differentiable bridge between PyTorch's computation graph and inter-GPU communication.
3. The **`FeatureExchanger` C++ class** — the communication engine that executes halo exchanges, with a pluggable backend for intra-node (NVLink / PCIe copy engine) and inter-node (GPUDirect RDMA) transport.
4. The **`push_buffer_create` CUDA kernel** — the GPU-side gather that packs atoms to be sent into a contiguous send buffer.

The module boundary is deliberately narrow: `distnnp_model` only knows about the halo maps in `DistNNPState` (from `distnnp_state`). It does not know about integrators, trajectory output, or repartitioning.

---

## Module Layout

```
distnnp_model/
├── include/distnnp_model/
│   ├── exchanger.hpp          # FeatureExchanger C++ class declaration
│   ├── push_buffer.hpp        # push_buffer_create declaration
│   └── macros.hpp             # CHECK_INPUT, CUDACHECK helpers
├── src/
│   ├── csrc/
│   │   ├── exchanger.cu       # FeatureExchanger forward + backward implementations
│   │   └── push_buffer.cu     # push_buffer_create CUDA kernel
│   ├── distnnp_model_bindings.cpp   # pybind11 module definition
│   └── distnnp_model/
│       ├── potential.py       # DistNNPModel ABC
│       ├── comm/
│       │   ├── feature_push.py      # FeaturePushToExchange autograd Function
│       │   └── backends/            # (planned) transport backend implementations
│       │       ├── base.py          # AbstractCommBackend
│       │       ├── copy_engine.py   # intra-node: CUDA P2P / cudaMemcpyAsync
│       │       └── rdma.py          # inter-node: GPUDirect RDMA
│       └── empirical/
│           └── lj_potential_model.py  # LennardJonesPotentialModel (reference impl)
```

---

## `DistNNPModel` ABC

Declared in `src/distnnp_model/potential.py`.

```python
class DistNNPModel(ABC):
    def __init__(self) -> None:
        self._feature_exchanger = FeatureExchanger()

    @abstractmethod
    def __call__(
        self, state: DistNNPState, *args, **kwargs
    ) -> Tuple[Tensor | float, Tensor]:
        """
        Returns: (local_energy, forces)
          local_energy : scalar Tensor — potential energy for this rank's owned atoms
          forces       : Tensor [num_local, 3] — forces on owned atoms only
        """
```

### Contract

Every `DistNNPModel` subclass must satisfy these invariants:

- **Energy is local.** `local_energy` covers only the atoms owned by this rank. Cross-rank energy reduction is the caller's responsibility (`Simulator._log_energy`).
- **Forces are owned-atoms only.** `forces` has shape `[num_local, 3]` where `num_local = state.local_partition_info.partition_indices.shape[0]`. Ghost atom contributions are accumulated onto owned atoms by PyTorch autograd during `energy.backward()`.
- **Halo exchange is internal.** Each `__call__` implementation is responsible for fetching the ghost atoms it needs via `FeaturePushToExchange.apply(...)` before computing the graph or model forward pass. The `Simulator` and integrator do not call any exchange function.
- **The computation graph is local.** After halo exchange, each rank builds its own radius graph over `[local + ghost]` atoms and runs the NNP forward pass without further cross-rank communication (except for per-layer GNN exchanges; see below).

### `_feature_exchanger`

`DistNNPModel.__init__` creates one `FeatureExchanger` instance, stored as `self._feature_exchanger`. Subclasses use it through `FeaturePushToExchange.apply(tensor, state, self._feature_exchanger)` rather than calling it directly. This ensures the exchange participates in autograd.

For GNN models that need a different `FeatureExchanger` per message-passing layer (because each layer exchanges a different feature tensor), subclasses may create additional exchanger instances.

---

## Data Flow Through `__call__`

Using `LennardJonesPotentialModel` as the canonical reference:

```
state.local_partition_info
  ├── local_atom_positions   [num_local, 3]   ← owned atom positions (requires_grad after clone)
  ├── partition_indices      [num_local]       ← global atom indices / types
  ├── send_local_indices     [total_send]      ← CSR: which local atoms to push to neighbours
  └── recv_ghost_indices     [total_recv]      ← CSR: ghost slots to receive into

Step 1: Halo exchange — positions
  boundary_positions = FeaturePushToExchange.apply(
      local_positions, state, self._feature_exchanger
  )  → [num_ghost, 3]

Step 2: Halo exchange — atom types (for type-dependent potentials)
  boundary_atom_types = FeaturePushToExchange.apply(
      local_atom_types.float(), state, self._feature_exchanger
  ).long()

Step 3: Concatenate local + ghost
  all_positions  = cat([local_positions, boundary_positions], dim=0)   [num_local + num_ghost, 3]
  all_atom_types = cat([local_atom_types, boundary_atom_types], dim=0)

Step 4: Build radius graph (no_grad — graph topology is not differentiable)
  with torch.no_grad():
      edge_index, _, _ = radius_graph_pbc_cuda(all_positions, ...)

Step 5: NNP forward pass — scalar energy
  energy = model_fn(all_atom_types, all_positions, edge_index, ...)

Step 6: Autograd backward — forces
  energy.backward()
  forces = -local_positions.grad   [num_local, 3]

Step 7: Return
  return energy.detach(), forces.detach()
```

**Why ghost forces appear on owned atoms:** Because `local_positions` was created with `requires_grad=True` before the exchange, and `boundary_positions` was produced by `FeaturePushToExchange` (which records the exchange in the autograd graph), any gradient flowing back through `boundary_positions` is routed through `FeaturePushToExchange.backward`, which performs the reverse communication and accumulates ghost gradients onto the appropriate owned atoms via `feature_push_backward`.

---

## `FeaturePushToExchange` — Autograd Interface

Declared in `src/distnnp_model/comm/feature_push.py`. This is a `torch.autograd.Function` that wraps a halo exchange and makes it differentiable.

```python
all_features = FeaturePushToExchange.apply(
    features_to_push,  # [num_local, F]  — owned-atom features
    state,             # DistNNPState
    exchanger,         # Exchanger instance
)  # → [num_local + num_ghost, F]
```

`FeaturePushToExchange.apply` is the only correct entry point when forces are needed. Calling `exchanger.feature_push_forward` directly is only appropriate for inference without force computation (i.e., no `energy.backward()`).

#### Forward semantics

Input: `features_to_push [num_local, F]` — features on owned atoms.

Operation:
1. Gather the subset of rows identified by `send_local_indices` (CSR) into a contiguous send buffer using `push_buffer_create`.
2. Transmit the send buffer to each neighbour rank according to the halo map.
3. Receive ghost features from neighbour ranks into a recv buffer.
4. Return `ghost_features [num_ghost, F]` — the received features, ordered by `recv_ghost_indices` (CSR).

The returned tensor covers only ghost atoms, not local atoms. The caller concatenates them: `all_features = cat([local, ghost], dim=0)`.

Input: `grad_output [num_local + num_ghost, F]` — loss gradients flowing back through the full feature matrix.

Input: `grad_output [num_ghost, F]` — loss gradients flowing back through the ghost atom features.

The scatter-add is correct because multiple remote ranks may use the same local atom as a ghost; each contributes a partial gradient that must be accumulated.

---

## `Exchanger` ABC — Communication Engine

Declared in `src/distnnp_model/comm/exchanger.py`. `Exchanger` owns all persistent communication state and exposes four methods that `FeaturePushToExchange` and model subclasses call.

```python
class Exchanger(ABC):
    def setup_exchanger(
        self,
        distributed_env: DistEnvironment,
        partition_info: PartitionMessagePassingInfo,
        max_feature_dim: int,
    ) -> None: ...

    def reset_buffers(
        self,
        feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None: ...

    def feature_push_forward(
        self,
        features_to_push: torch.Tensor,       # [num_local, F]
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor: ...                     # [num_local + num_ghost, F]

    def feature_push_backward(
        self,
        grad_output: torch.Tensor,             # [num_local + num_ghost, F]
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor: ...                     # [num_local, F]
```

### `setup_exchanger`

## `FeatureExchanger` — Communication Engine

Declared in `include/distnnp_model/exchanger.hpp`, implemented in `src/csrc/exchanger.cu`. Exposed to Python via pybind11.

```cpp
class FeatureExchanger {
public:
    torch::Tensor feature_push_forward(
        const torch::Tensor &local_features,           // [num_local, F]
        const distnnp_state::PartitionMessagePassingInfo &partition_info
    );  // returns ghost_features [num_ghost, F]

    torch::Tensor feature_push_backward(
        const torch::Tensor &grad_output,              // [num_ghost, F]
        const torch::IntArrayRef &grad_input_shape,    // [num_local, F]
        const distnnp_state::PartitionMessagePassingInfo &partition_info
    );  // returns grad_input [num_local, F]
};
```

Pass the largest feature dimension that will appear across all layers (or across all steps if repartitioning changes atom counts) as `max_feature_dim`.

`FeatureExchanger` owns all state required for communication: registered memory buffers, connection handles (QPs for RDMA), and cached send/recv split sizes derived from the halo maps. It is the only object in `distnnp_model` that talks to the network or to peer GPUs.

### How halo maps are consumed

Both `feature_push_forward` and `feature_push_backward` read the following fields from `PartitionMessagePassingInfo`:

| Field | Usage |
|---|---|
| `send_offsets [W+1]` | CSR row pointers — `send_offsets[r]..send_offsets[r+1]` indexes `send_local_indices` for rank `r` |
| `send_local_indices [total_send]` | Ordered list of local atom indices to send, sorted by destination rank |
| `recv_offsets [W+1]` | CSR row pointers — `recv_offsets[r]..recv_offsets[r+1]` indexes `recv_ghost_indices` for rank `r` |
| `recv_ghost_indices [total_recv]` | Ordered list of ghost slot positions (row offsets ≥ `num_local`) where received atoms land |
| `global_send_recv_matrix [W, W]` | Used at setup time to identify which rank pairs communicate |

### Intra-node: Copy Engine (NVLink / PCIe)

Used when sender and receiver are GPUs on the same physical node. The copy engine is the CUDA DMA hardware that moves data between GPU memory regions without CPU involvement and without occupying CUDA compute SMs.

**Mechanism:**
1. At setup, enable peer-to-peer access: `cudaDeviceEnablePeerAccess(peer_device, 0)` for each neighbour GPU on the same node. This makes the peer GPU's memory directly addressable from the local GPU.
2. Allocate persistent send and recv GPU buffers (sized for `max_send_atoms × max_feature_dim`).
3. At each exchange:
   a. `push_buffer_create` gathers send atoms into the local send buffer (on the local GPU, compute stream).
   b. `cudaMemcpyAsync(peer_recv_buffer, local_send_buffer, ..., cudaMemcpyDeviceToDevice, stream)` — the copy engine moves data directly. On NVLink this saturates high-bandwidth peer links; on PCIe it goes through the host bridge but still avoids CPU copies.
   c. A CUDA event is recorded on the stream and signalled to the peer to indicate the write is complete.

**NVLink vs PCIe:** Both use the same API. NVLink provides ~600 GB/s aggregate per A100, whereas PCIe 4.0 ×16 provides ~32 GB/s. The halo volume at ~4,000 atoms with 128-dimensional embeddings is small enough (~2 MB per step) that both interconnects complete well within one timestep.

**Why not NCCL all_to_all?** NCCL `all_to_all` is a collective that requires all ranks to participate simultaneously. At the halo exchange granularity (typically only a few rank pairs communicate non-zero volume), an all_to_all wastes bandwidth for zero-volume pairs. The copy engine backend sends only to ranks with non-zero send count, matching the sparsity of `global_send_recv_matrix`.

### Inter-node: GPUDirect RDMA

Used when sender and receiver are on different physical nodes connected by InfiniBand or RoCE. GPUDirect RDMA allows the RDMA NIC to read from and write to GPU memory directly, bypassing both the CPU and host DRAM.

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

---

## Interface Between PyTorch Autograd and Communication

`FeaturePushToExchange` is the explicit boundary between CUDA/RDMA operations and PyTorch autograd.

### Forward pass autograd contract

When `FeaturePushToExchange.apply(local_features, state, exchanger)` is called:
- `local_features` must have `requires_grad=True` if forces are needed.
- The returned `[num_local + num_ghost, F]` tensor is a leaf output of the autograd Function. Gradients flowing back through it during `energy.backward()` trigger `FeaturePushToExchange.backward`.
- `state` and `exchanger` are not differentiable. `backward` returns `None` for both.

### Backward pass gradient routing

```
energy.backward()
  → grad flows through NNP → arrives at grad_all_features [num_local + num_ghost, F]

  FeaturePushToExchange.backward receives grad_all_features as grad_output.
  exchanger.feature_push_backward(grad_output, partition_info):
    → local slice  grad_output[:num_local]  — flows back directly to owned-atom gradients
    → ghost slice  grad_output[num_local:]  — sent back to owning ranks via reverse exchange
    → owning rank scatter-adds received ghost grads into local positions (send_local_indices)

  Final grad_input [num_local, F] = direct local grad + accumulated ghost grads from peers
```

Every ghost atom is a copy of an owned atom. The gradient at a ghost slot is the partial derivative of energy through that ghost's participation in the NNP. Accumulating all ghost gradients for the same owned atom gives the total gradient for that atom.

### Detaching the radius graph

Radius graph construction is performed inside `torch.no_grad()`. Graph topology (which atom pairs are neighbours) is a discrete function of positions and is not differentiable. Excluding it from autograd prevents incorrect gradient flow through graph construction and avoids storing intermediates for the graph builder.

---

## Multi-layer GNN Exchange

For GNN-based NNP models with `L` message-passing layers, a single position exchange before the first layer is insufficient. After each layer, each atom's embedding depends on its neighbours' embeddings from the previous layer. Rank `R` needs the layer-`l` embeddings of its ghost atoms to compute layer `l+1`.

The exchange pattern:

```
reset_buffers(feature_dim=3)
exchange positions                               → all_pos [num_local + num_ghost, 3]
layer 0 message passing                          → h [num_local + num_ghost, F_0]

reset_buffers(feature_dim=F_0)
exchange h[:num_local]                           → h [num_local + num_ghost, F_0]
layer 1 message passing                          → h [num_local + num_ghost, F_1]

...

reset_buffers(feature_dim=F_{L-2})
exchange h[:num_local]                           → h [num_local + num_ghost, F_{L-2}]
layer L-1 message passing                        → output embeddings / energy
```

Each exchange uses the same halo maps (fixed topology for the step) but a different feature dimension.

### Implementation pattern for GNN subclasses

```python
class MyGNNModel(DistNNPModel):
    def setup(self, distributed_env, partition_info, max_feature_dim):
        self._feature_exchanger.setup_exchanger(
            distributed_env, partition_info, max_feature_dim
        )

    def __call__(self, state):
        local_pos = state.local_partition_info.local_atom_positions.clone()
        local_pos.requires_grad_(True)
        num_local = local_pos.shape[0]
        device = local_pos.device

        # Exchange positions
        self._feature_exchanger.reset_buffers(3, state.local_partition_info, device)
        all_pos = FeaturePushToExchange.apply(local_pos, state, self._feature_exchanger)
        # all_pos: [num_local + num_ghost, 3]

        with torch.no_grad():
            edge_index = radius_graph(all_pos, ...)

        h = self.embedding(all_pos)  # [num_local + num_ghost, F_0]

        for layer in self.message_passing_layers:
            self._feature_exchanger.reset_buffers(
                layer.feature_dim, state.local_partition_info, device
            )
            h = FeaturePushToExchange.apply(h[:num_local], state, self._feature_exchanger)
            # h: [num_local + num_ghost, F_l]
            h = layer(h, edge_index)

        energy = self.output_head(h[:num_local]).sum()
        energy.backward()
        forces = -local_pos.grad
        return energy.detach(), forces
```

Because `FeaturePushToExchange` is called `L` times in the forward pass, `energy.backward()` triggers `L` reverse exchanges — one per layer in reverse order — automatically propagating gradients through the full communication graph.

---

## Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| `push_buffer_create` | Implemented | CUDA gather kernel with scalar and vec4 paths |
| `FeaturePushToExchange` | Implemented | Autograd Function wiring is complete |
| `FeatureExchanger` (C++ class) | Stub | `feature_push_forward` returns `zeros_like`; `feature_push_backward` returns zeros |
| `LennardJonesPotentialModel` | Implemented | End-to-end model using `FeaturePushToExchange` |
| Intra-node copy engine backend | Not implemented | Requires CUDA IPC / peer-to-peer setup |
| Inter-node RDMA backend | Not implemented | Requires GDR memory registration, QP setup, RDMA WRITE |
| Memory registration | Not implemented | See RDMA setup above |
| Connection setup | Not implemented | QP creation and address exchange via all_gather |

---

## Assumed Future Interfaces

| Symbol | Description |
|---|---|
| `AbstractCommBackend.setup(...)` | Allocate buffers, register memory, establish connections for one rank pair topology |
| `AbstractCommBackend.send_recv(send_buf, recv_buf, send_counts, recv_counts)` | Execute one bulk feature transfer, potentially asynchronous |
| `AbstractCommBackend.wait()` | Block until the last send_recv has landed in the recv buffer |
| `CopyEngineBackend` | Intra-node backend using cudaMemcpyAsync over NVLink or PCIe |
| `RDMABackend` | Inter-node backend using GPUDirect RDMA WRITE |
| `FeatureExchanger._select_backend(rank_a, rank_b, dist_env)` | Returns the appropriate backend for a given rank pair based on node topology |

---

## Open Design Questions

1. **Backend dispatch granularity**: The current plan dispatches per rank pair. An alternative is to batch all intra-node sends into one fused copy and all inter-node sends into one RDMA operation, which reduces the number of kernel launches and QP posts but requires more complex buffer management.

2. **Buffer reuse across feature dimensions**: Resolved — allocate once for `max_feature_dim` at topology setup time; switch the active transfer width cheaply via `reconfigure_feature_dim(F_l)` per layer (no IPC operations). Layers with `F_l < max_feature_dim` leave the tail columns of each row unused, which is acceptable. See *IPC Buffer Lifecycle and Persistent Buffers* above.

3. **Asynchronous exchange and computation overlap**: The baseline (`cfg.dist.overlap`) can overlap the NNP compute on interior atoms (identified by `interior_mask` in `PartitionMessagePassingInfo`) with the halo exchange for boundary atoms. This requires splitting the NNP forward pass into two phases: interior atoms first (while communication runs), then boundary atoms after the exchange completes. The `FeatureExchanger` must expose an async `begin_send_recv()` / `wait()` interface for this.

4. **Gradient correctness for non-differentiable position exchange**: In the `LennardJonesPotentialModel`, `local_positions` is detached and re-created with `requires_grad=True` before the exchange. This is necessary to make `energy.backward()` produce forces via `local_positions.grad`. However, it breaks the gradient path from `state.local_partition_info.local_atom_positions` through to `energy` for any use case that needs end-to-end differentiability (e.g., training the NNP). A future design should clarify whether `DistNNPModel` is always in inference/MD mode or must also support training.

5. **RDMA completion notification**: The doorbell (RDMA SEND/RECV) approach described above is one option. An alternative is polling on a GPU-side flag written by the NIC after the RDMA WRITE completes (GPU direct notification via `nv_peer_mem` or NVSHMEM's `nvshmemx_getmem_nbi`). The polling approach avoids the extra QP verb but requires careful fence ordering to prevent the flag read from being speculated ahead of the data.
