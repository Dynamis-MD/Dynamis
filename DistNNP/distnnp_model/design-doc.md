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
├── ├── NVFeatureExhanger
│       ├── NVExchanger.hpp
│       ├── p2p_exchanger.hpp
│       └── rdma_exchanger.hpp
├── ├── NCCLFeatureExhanger
│       ├── NCCLExchanger.hpp
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
## The Communication Primitives

The two main communication components in `DistNNP_model` are the `FeaturePushToExchange` function and the `Exchange` class. 


### `FeaturePushToExchange` — Autograd Interface

Declared in `src/distnnp_model/comm/feature_push.py`. This is a `torch.autograd.Function` that makes halo exchange differentiable.

```python
class FeaturePushToExchange(Function):
    @staticmethod
    def forward(ctx, features_to_push, state, exchanger) -> Tensor:
        # Returns: ghost_features [num_ghost, F]
        ctx.save_for_backward(...)
        return exchanger.feature_push_forward(features_to_push, state.local_partition_info)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        # grad_output: [num_ghost, F] — gradients at ghost slots
        # Returns: [num_local, F] — gradients accumulated on owned atoms
        return exchanger.feature_push_backward(
            grad_output, ctx._features_to_push_shape, state.local_partition_info
        ), None, None
```

### Forward semantics

Input: `features_to_push [num_local, F]` — features on owned atoms.

Operation:
1. Gather the subset of rows identified by `send_local_indices` (CSR) into a contiguous send buffer using `push_buffer_create`.
2. Transmit the send buffer to each neighbour rank according to the halo map.
3. Receive ghost features from neighbour ranks into a recv buffer.
4. Return `ghost_features [num_local + num_ghost, F]` — the received features, ordered by `recv_ghost_indices` (CSR).


### Backward semantics

Input: `grad_output [num_local + num_ghost, F]` — loss gradients flowing back through the ghost atom features.

Operation (reverse of forward):
1. Gather the ghost grad rows (mirror of recv in forward) into a contiguous send buffer.
2. Transmit ghost gradients back to the ranks that originally sent those atoms.
3. Receive and scatter-add into `grad_input [num_local, F]` at the positions identified by `send_local_indices`.

The scatter-add is correct because multiple remote ranks may have used the same local atom as a ghost — each contributes a gradient that must be accumulated.

### Non-differentiable path

For position exchange during inference or when forces are computed without autograd (e.g., a tabulated potential), the exchange can be performed without wrapping in `FeaturePushToExchange`. This avoids the autograd overhead and allows the backend to skip recording context. This is exposed as a direct call to `exchanger.feature_push_forward(...)` rather than via `.apply(...)`.

---

## `push_buffer_create` — GPU Gather Kernel

Declared in `include/distnnp_model/push_buffer.hpp`, implemented in `src/csrc/push_buffer.cu`.

```cpp
at::Tensor push_buffer_create(
    at::Tensor input,       // [num_local, F]  — source feature matrix
    at::Tensor buffer,      // [num_send, F]   — pre-allocated send buffer
    at::Tensor input_locs   // [num_send]  int64 — indices of rows to gather
);
```

**What it does:** Gathers `input[input_locs[i], :]` into `buffer[i, :]` for all `i`. This is the indirection-gather step that packs the atoms to be sent into a layout suitable for a bulk transfer.

**Performance path:** When `F % 4 == 0`, the kernel uses a `float4` vector path (`push_buffer_gather_kernel_vec4`) that moves 16 bytes per thread instead of 4, giving up to 4× better memory throughput for large feature dimensions. NNP embedding dimensions (e.g., 128, 256, 512) are almost always divisible by 4, so this path is the common case.

**Thread layout:** 32×32 threads per block. Row index covers `num_send`, column index covers `F` (or `F/4` in the vec4 path). Grid is sized to cover all rows and columns without overflow.

**Device pinning:** `at::cuda::CUDAGuard device_guard(input.device())` ensures the kernel runs on the correct GPU when multiple GPUs are visible. Uses the current CUDA stream so it integrates with PyTorch's stream management.

**Python binding:**
```python
from distnnp_model.distnnp_model_impl import push_buffer_create
send_buffer = push_buffer_create(local_features, send_buffer, send_indices)
```

---

## `Exchanger` — Communication Engine

### Role

`Exchanger` owns all state required for communication and exposes the necessary API for a model for the halo atom exchange. Is the only object in `distnnp_model` that talks to the network or to peer GPUs.

### How halo maps are consumed

`feature_push_forward` reads the following fields from `PartitionMessagePassingInfo`:

| Field | Usage |
|---|---|
| `send_offsets [W+1]` | CSR row pointers — `send_offsets[r]..send_offsets[r+1]` indexes `send_local_indices` for rank `r` |
| `send_local_indices [total_send]` | Ordered list of local atom indices to send, sorted by destination rank |
| `recv_offsets [W+1]` | CSR row pointers — `recv_offsets[r]..recv_offsets[r+1]` indexes `recv_ghost_indices` for rank `r` |
| `recv_ghost_indices [total_recv]` | Ordered list of ghost slot positions (indices ≥ `num_local`) where received atoms land |
| `global_send_recv_matrix [W, W]` | Used at setup time to know which rank pairs have non-zero communication volume |

`feature_push_backward` uses the same maps but with send and recv roles reversed: ghost gradients flow back from ghost slots (recv side in forward) to owned atom positions (send side in forward).

---

## Communication Backends

`FeatureExchanger` dispatches to one of two transport backends based on whether the communicating rank pair is on the same physical node (intra-node) or on different nodes (inter-node). The dispatch decision is made once at setup time, using `DistEnvironment.gpus_per_node` and rank arithmetic.

```
rank_a // gpus_per_node == rank_b // gpus_per_node  →  intra-node backend
rank_a // gpus_per_node != rank_b // gpus_per_node  →  inter-node backend
```

Each backend satisfies the same abstract interface:

```python
class AbstractCommBackend:
    def setup(self, local_rank, peer_ranks, send_counts, recv_counts, feature_dim_max):
        """Allocate persistent buffers, register memory, establish connections."""

    def send_recv(self, send_buffer, recv_buffer, send_counts, recv_counts):
        """Execute one round of bulk feature transfer. May be asynchronous."""

    def wait(self):
        """Block until the last send_recv has completed on the CUDA stream."""
```


---

## Interface Between PyTorch Autograd and Communication

The central design challenge is that CUDA kernels and RDMA operations are not natively understood by PyTorch's autograd engine. `FeaturePushToExchange` is the explicit boundary.

### Forward pass autograd contract

When `FeaturePushToExchange.apply(local_features, state, exchanger)` is called:
- `local_features` must have `requires_grad=True` if forces are needed.
- The returned `ghost_features` tensor is a leaf output of the autograd Function. Gradients flowing back through `ghost_features` during `energy.backward()` will trigger `FeaturePushToExchange.backward`.
- `state` and `exchanger` have no gradient (they are metadata and a stateful object, respectively). `backward` returns `None` for both.

### Backward pass gradient routing

```
energy.backward()
  → grad flows through NNP → arrives at grad_all_positions [num_local + num_ghost, 3]
  → local slice:  grad_all_positions[:num_local]   = direct gradient on owned atoms
  → ghost slice:  grad_all_positions[num_local:]   = gradient on ghost atoms

  Ghost gradient slice is the grad_output seen by FeaturePushToExchange.backward.
  backward() calls exchanger.feature_push_backward(grad_output, ...)
  → reverse communication: ghost grad slice is sent back to the owning rank
  → owning rank accumulates into its local grad_input via scatter-add

  Final grad for local_positions = direct gradient + accumulated ghost gradient
```

This gradient routing is correct because every ghost atom is a copy of some owned atom. The gradient at the ghost slot is the partial derivative of energy with respect to that owned atom's position, computed through the ghost's participation in the NNP. Scatter-adding all ghost gradients for the same owned atom gives the total gradient.

### Detaching the radius graph

The radius graph construction (`radius_graph_pbc_cuda`) is performed inside `torch.no_grad()`. The graph topology (which atom pairs are neighbours) is a function of positions but is not differentiable in the subgraph sense — edges are discrete. Excluding it from autograd prevents incorrect gradient flow through the graph construction and reduces memory overhead from storing the graph's intermediates.

---

## Multi-layer GNN Exchange

For GNN-based NNP models with `L` message-passing layers, a single pre-step position exchange is insufficient. After each message-passing layer, each atom's embedding depends on its neighbours' embeddings from the previous layer. To compute layer `l+1`, rank `R` needs the layer-`l` embeddings of its ghost atoms — but those were computed on a different rank.

The exchange pattern is:

```
[all ranks]
  exchange positions (layer 0 input)
  layer 0 message passing  →  new embeddings [num_local + num_ghost, F_0]

  exchange updated embeddings (layer 0 output / layer 1 input)
  layer 1 message passing  →  new embeddings [num_local + num_ghost, F_1]

  ...

  exchange updated embeddings (layer L-2 output / layer L-1 input)
  layer L-1 message passing  →  output embeddings / energy
```

Each exchange call uses the same halo maps from `partition_info` (the maps encode the graph topology which is fixed for the step), but a different feature tensor with potentially different feature dimension `F_l`.

### IPC Buffer Lifecycle and Persistent Buffers

CUDA IPC registration is a **destination** requirement, not a source requirement. `cudaIpcGetMemHandle` — which requires the exact base pointer of a raw `cudaMalloc` allocation — is only called on the buffer that **receives** peer writes. The rank doing the writing reads from its own local memory (any device pointer, including a caching-allocator tensor) and writes into the IPC-opened pointer on the remote rank. No IPC registration is needed on the sender side.

This means:

- `x_buffer_` on each rank is the **receive** buffer for ghost atoms written by peers. It must be a `cudaMalloc` allocation so its handle can be exported and opened by peers.
- `local_features` (a standard PyTorch caching-allocator tensor) is the **source** read by the push kernel. No IPC handle is needed for it.
- **No copy of local features into the IPC buffer is required.** The push kernel reads directly from `local_features[send_local_indices[i]]` and writes to `peer_x_buffer[remote_offset + ghost_idx]`.

As a result, `x_buffer_` only needs to be sized `[N_ghost, max_F]` — it holds only the ghost receive region. After the exchange, the caller concatenates:

```python
ghost_h = FeaturePushToExchange.apply(h[:N_local], state, exchanger)
h = cat([h[:N_local], ghost_h], dim=0)  # caller's responsibility
```

This matches the `FeaturePushToExchange` API contract: it returns ghost features only; the caller concatenates with local features.

**One-time IPC setup, cheap per-layer reconfiguration.** Opening IPC handles (`cudaIpcOpenMemHandle`) is an expensive kernel-mode operation requiring cross-process coordination. It must not be repeated per GNN layer. The design separates two concerns:

- **Topology setup** (once per partition): allocate `x_buffer [N_ghost, max_F]` and `grad_recv_buffer [total_send, max_F]` for the maximum feature dimension across all layers; export and exchange handles via a single `all_gather`; open remote handles; compute and upload `PushTarget` arrays.
- **Feature dimension reconfiguration** (once per layer, O(1)): call `reconfigure_feature_dim(F_l)` to update the active column count. This writes a single integer field — no IPC operations, no collectives, no buffer reallocation.

```python
# One-time initialization (per partition topology)
exchanger.initialize(max_feature_dim=256, partition_info=..., device=rank)
    # → cudaMalloc [N_ghost, 256], all_gather handles, cudaIpcOpenMemHandle, PushTarget upload

# Per-layer exchange (no copy of local features needed)
for layer in gnn_layers:
    exchanger.set_feature_dim(layer.feature_dim)  # O(1): no IPC ops
    ghost_h = FeaturePushToExchange.apply(h[:N_local], state, exchanger)
    h = cat([h[:N_local], ghost_h], dim=0)
    h = layer(h, edge_index)
```

`set_feature_dim` / `reconfigure_feature_dim` only updates `feature_dim_` (active transfer width) and validates `new_F <= max_feature_dim_` (the allocated row stride). `PushTarget.remote_offset` is stored in atoms (rows), remaining valid for any `feature_dim <= max_feature_dim`.

The memory overhead of this approach is `(max_F - active_F) * sizeof(float)` per ghost row for layers with smaller F — negligible compared to the cost of repeated handle exchange.

**Implementation pattern for GNN subclasses:**

```python
class MyGNNModel(DistNNPModel):
    def __call__(self, state):
        # Layer 0: exchange positions
        ghost_pos = FeaturePushToExchange.apply(local_pos, state, self._feature_exchanger)
        all_pos = cat([local_pos, ghost_pos], dim=0)

        # Layer 0 forward
        h = self.embedding(all_pos)  # [num_local + num_ghost, F_0]

        for layer in self.message_passing_layers:
            # Exchange embeddings before each layer (ghost slots are updated)
            ghost_h = FeaturePushToExchange.apply(h[:num_local], state, self._feature_exchanger)
            h = cat([h[:num_local], ghost_h], dim=0)
            h = layer(h, edge_index)  # [num_local + num_ghost, F_l]

        energy = self.output_head(h[:num_local]).sum()
        energy.backward()
        forces = -local_pos.grad
        return energy.detach(), forces
```

Because `FeaturePushToExchange` is called `L` times in the forward pass, `energy.backward()` triggers `L` reverse exchanges — one per layer in reverse order — automatically propagating gradients through the full communication graph.
