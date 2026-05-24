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

The two main communication components in `distnnp_model` are the `FeaturePushToExchange` autograd Function and the `Exchanger` ABC. Together they form the boundary between PyTorch's computation graph and inter-GPU communication.

### `FeaturePushToExchange` — Autograd Interface

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

Output: `[num_local + num_ghost, F]` — owned atom features concatenated with features received from neighboring partitions. Rows `0..num_local-1` are the unmodified local features; rows `num_local..num_local+num_ghost-1` are the ghost features received from peer ranks, ordered according to `recv_ghost_indices` in `PartitionMessagePassingInfo`.

#### Backward semantics

Input: `grad_output [num_local + num_ghost, F]` — loss gradients flowing back through the full feature matrix.

The local slice `grad_output[:num_local]` flows directly back to the owned atom features. The ghost slice `grad_output[num_local:]` is sent back to the ranks that originally sent those atoms, where it is scatter-added into `grad_input` at the positions identified by `send_local_indices`. The returned gradient `[num_local, F]` is the sum of the direct local gradient and the accumulated ghost gradients received from peers.

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

Called **once per partition** before any exchange. Takes the full `DistEnvironment` and `PartitionMessagePassingInfo` to establish connectivity between ranks, and allocates persistent internal buffers sized for `max_feature_dim` columns. After this call, the exchanger is ready to handle any `feature_dim <= max_feature_dim` without further setup.

Pass the largest feature dimension that will appear across all layers (or across all steps if repartitioning changes atom counts) as `max_feature_dim`.

### `reset_buffers`

Called **once per exchange** to reconfigure the active feature dimension to `feature_dim`. This is a lightweight operation — it does not reallocate buffers or re-establish connections. It allows the same exchanger to be reused across GNN layers with different embedding widths.

`reset_buffers` must be called before each call to `FeaturePushToExchange.apply`. It is also the right place to re-upload any device-side metadata that is feature-dim-dependent.

### `feature_push_forward`

Executes the halo exchange. Reads `send_local_indices` from `partition_info` to gather rows from `features_to_push`, sends them to peer ranks, receives ghost features, and returns them concatenated with the input as `[num_local + num_ghost, F]`. Called internally by `FeaturePushToExchange.forward`.

### `feature_push_backward`

Executes the reverse exchange. Separates `grad_output` into the local slice (rows `0..num_local-1`) and the ghost slice (rows `num_local..`). Communicates the ghost slice back to owning ranks, scatter-adds received gradients at positions `send_local_indices`, and returns `[num_local, F]`. Called internally by `FeaturePushToExchange.backward`.

### How halo maps are consumed

Both `feature_push_forward` and `feature_push_backward` read the following fields from `PartitionMessagePassingInfo`:

| Field | Usage |
|---|---|
| `send_offsets [W+1]` | CSR row pointers — `send_offsets[r]..send_offsets[r+1]` indexes `send_local_indices` for rank `r` |
| `send_local_indices [total_send]` | Ordered list of local atom indices to send, sorted by destination rank |
| `recv_offsets [W+1]` | CSR row pointers — `recv_offsets[r]..recv_offsets[r+1]` indexes `recv_ghost_indices` for rank `r` |
| `recv_ghost_indices [total_recv]` | Ordered list of ghost slot positions (row offsets ≥ `num_local`) where received atoms land |
| `global_send_recv_matrix [W, W]` | Used at setup time to identify which rank pairs communicate |

`feature_push_backward` uses the same maps with send and recv roles swapped: ghost gradients flow back from ghost slots (recv in forward) to owned atom positions (send in forward).

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

Because `FeaturePushToExchange` is called `L+1` times in the forward pass, `energy.backward()` triggers `L+1` reverse exchanges automatically, propagating gradients through the full communication graph.

### Single-exchange models

For simple pairwise potentials that only need positions (e.g., `LennardJonesPotentialModel`), one exchange per step is sufficient:

```python
self._feature_exchanger.reset_buffers(3, state.local_partition_info, device)
all_pos = FeaturePushToExchange.apply(local_pos, state, self._feature_exchanger)
```

`reset_buffers` is still called each step because `partition_info` may change across steps if repartitioning occurs.
