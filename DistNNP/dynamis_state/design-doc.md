# dynamis_state Design Document

`dynamis_state` is the foundational sub-package of DistNNP. It defines the data structures shared across all other sub-packages, the algorithms for partitioning atoms across GPUs, and the infrastructure for building the halo communication maps that enable boundary atom exchange. All types are defined in `include/dynamis_state/` and exposed to Python via pybind11.

---

## State Structs

All structs live in `include/dynamis_state/dynamis_state.hpp` inside the `dynamis_state` namespace.

### `MolecularInfo`

Holds the raw molecular system data for a partition (or the full system, depending on context).

```cpp
struct MolecularInfo {
    unsigned long  num_atoms;        // number of atoms in this view
    torch::Tensor  atom_indices;     // [num_atoms]    global atom indices (int64)
    torch::Tensor  atom_positions;   // [num_atoms, 3] float32 positions (Å)
    torch::Tensor  atom_velocities;  // [num_atoms, 3] float32 velocities (Å/fs)
    torch::Tensor  edge_counts;      // [num_atoms]    optional per-atom edge cost
};
```

`edge_counts` is optional. When defined, partitioners use it as a load-balance weight (heavier atoms — those with more neighbours — are spread more evenly across ranks). When undefined, all atoms are treated as equal-weight. Set `edge_counts` on the global `MolecularInfo` before calling `partition()` to use user-supplied weights; pass a `SimulationConfig` to `partition()` instead to have the partitioner compute weights automatically from a radius graph.

### `DistEnvironment`

Captures the MPI/NCCL process layout. The fields mirror PyTorch's `dist` terminology.

```cpp
struct DistEnvironment {
    unsigned long global_rank;    // rank in [0, world_size)
    unsigned long world_size;     // total number of processes / GPUs
    unsigned long gpus_per_node;  // GPUs on a single physical node
    unsigned long num_nodes;      // number of physical nodes
    unsigned long local_rank;     // rank within the node, maps to GPU index
};
```

### `SimulationConfig`

Stores the physical parameters of the simulation cell. Required for any operation that builds a neighbour graph (radius-based halo construction, auto-weighted partitioning).

```cpp
struct SimulationConfig {
    double        cutoff;              // neighbour cutoff radius (Å)
    double        ghost_cutoff_buffer; // extra buffer for ghost atoms (Å, default 0.5)
    torch::Tensor cell;                // [3, 3] float32 — simulation cell matrix (Å)
    torch::Tensor pbc;                 // [3]    int32  — periodic boundary flags (0 or 1)
};
```

The effective ghost radius used in halo construction is `cutoff + ghost_cutoff_buffer`. The buffer ensures that atoms just outside the strict cutoff are still captured in the ghost layer, preventing boundary effects from repartitioning drift.

### `PartitionMessagePassingInfo`

The primary output of the partitioner pipeline. Each rank holds one of these, populated in two phases: first by `partition()`, then by `build_halo_maps()`.

**After `partition()`:**

| Field | Shape | Description |
|---|---|---|
| `partition_id` | `[N]` int64 | Rank assignment for every atom in the global system |
| `partition_indices` | `[num_local]` int64 | Global atom indices owned by this rank |
| `global_send_recv_matrix` | `[W, W]` int64 | Atom-count matrix; only `row[rank]` diagonal is filled initially |

**After `build_halo_maps()`:**

| Field | Shape | Description |
|---|---|---|
| `send_offsets` | `[W + 1]` int64 | CSR row pointers for the send map |
| `send_local_indices` | `[total_send]` int64 | Local atom indices (0-based within this rank) to push to neighbours |
| `recv_offsets` | `[W + 1]` int64 | CSR row pointers for the receive map |
| `recv_ghost_indices` | `[total_recv]` int64 | Ghost slot indices (starting at `num_local`) for received atoms |
| `interior_mask` | `[num_local]` bool | `true` = atom has no neighbours on another rank (no halo needed) |
| `edge_index_halo` | `[2, E']` int64 | Neighbour graph built at `cutoff + ghost_buffer`; stored for downstream NNP use |
| `edge_index` | `[2, E]` int64 | Neighbour graph built at `cutoff`; only present in auto-weighted partition mode |

**CSR layout:** `send_offsets[r]..send_offsets[r+1]` indexes the slice of `send_local_indices` that holds atoms destined for rank `r`. The same convention applies to `recv_offsets` / `recv_ghost_indices`. After halo exchange, the local tensor on a given rank has shape `[num_local + total_recv, ...]`, with ghost atoms occupying slots `num_local..num_local+total_recv-1`.

**`global_send_recv_matrix` semantics:** After `build_halo_maps()`, `row[rank]` of this matrix is updated with the number of halo atoms sent to each neighbour. The diagonal entry `[rank, rank]` is set to `num_local` (not a halo count). The full matrix view requires an all-reduce across ranks (not performed here; left to the caller).

### `DistNNPState`

Top-level aggregate passed into `distnnp_model` and `dynamis_integrator` at every timestep.

```cpp
struct DistNNPState {
    DistEnvironment              dist_env;               // MPI/NCCL layout
    MolecularInfo                global_molecular_info;  // full system (all atoms)
    PartitionMessagePassingInfo  local_partition_info;   // this rank's view
};
```

`global_molecular_info` is present on every rank at startup (loaded from file) but is not replicated dynamically at runtime. After partitioning, each rank works only from `local_partition_info`. The global info is retained for restart checkpointing and repartitioning.

---

## Partitioner Design

All partitioners are declared in `include/dynamis_state/partitioner.hpp` and implemented in `src/partition/`.

### `AbstractPartitioner`

```cpp
class AbstractPartitioner {
public:
    virtual PartitionMessagePassingInfo partition(
        const MolecularInfo  &state,
        const DistEnvironment &env) = 0;

    void build_halo_maps(
        PartitionMessagePassingInfo &info,
        const MolecularInfo         &global_state,
        const DistEnvironment       &env,
        const SimulationConfig      &config);
};
```

The interface separates two concerns:

1. **`partition()`** — assigns atoms to ranks. Each subclass implements its own spatial decomposition strategy. The method reads `global_molecular_info` and writes `partition_id`, `partition_indices`, and the diagonal of `global_send_recv_matrix`.

2. **`build_halo_maps()`** — shared across all subclasses. Given the `partition_id` tensor produced by any `partition()` call and a `SimulationConfig`, it builds the full send/receive CSR maps. It runs on whatever device the input tensors are on (CUDA required for the internal `radius_graph_pbc` call).

The separation exists so that `partition()` can be called once at startup (or on repartition), and `build_halo_maps()` is re-called whenever the simulation cell or atom positions have drifted enough to invalidate the ghost map.

### Calling sequence

```python
partitioner = SFCPartitioner(curve_type=SFCCurveType.Morton)

# Phase 1: assign atoms to ranks
info = partitioner.partition(global_mol_info, dist_env, sim_config)  # auto-weighted

# Phase 2: build halo communication maps
partitioner.build_halo_maps(info, global_mol_info, dist_env, sim_config)

# Assemble top-level state
state = DistNNPState()
state.dist_env = dist_env
state.global_molecular_info = global_mol_info
state.local_partition_info = info
```

### `SimpleSpatialPartitioner`

The cheapest partitioner. Assigns atoms in index order: rank `r` receives the contiguous slice `[r * N/W, (r+1) * N/W)` of the global atom array. The last rank absorbs any remainder.

- **Cost:** O(1), no CUDA kernels.
- **Quality:** Good only when the input atom ordering already has spatial locality (e.g., a pre-sorted trajectory file). Random or layered orderings produce poor halo volume.
- **Use case:** Baseline benchmarking, validation, or when atoms are ordered along a natural scan axis.
- **Produced fields:** `partition_id` (via `arange // atoms_per_rank` clamped to `W-1`), `partition_indices`, `global_send_recv_matrix` diagonal.

### `KDTreePartitioner`

Recursive binary space partitioning. Repeatedly bisects the atom set along the longest spatial axis until `W` sub-trees are formed, each becoming one rank's partition.

**Three overloaded modes via `partition()`:**

| Mode | When | Behaviour |
|---|---|---|
| Unweighted | `edge_counts` undefined, no `SimulationConfig` | Each bisection split is at the median atom position |
| User-weighted | `edge_counts` defined on `MolecularInfo` | Split point is the cumulative-weight median |
| Auto-weighted | `edge_counts` undefined + `SimulationConfig` passed | Builds radius graph at `cutoff`, counts dst-side edges per atom, uses those counts as weights; stores graph in `info.edge_index` |

On CUDA, bisection is delegated to `kdtree_bisect_cuda()`. On CPU, a recursive C++ fallback (`kdtree_bisect()`) is used. Both produce the same `partition_id` tensor.

**Why dst-based edge counts?** The NNP message-passing compute is on the destination atom (aggregation). Balancing dst-edge count per rank minimises GNN compute imbalance.

### `SFCPartitioner`

Space-filling curve partitioning. Atoms are mapped to a 1D SFC key (Morton or Hilbert), sorted along the curve, and split into `W` contiguous segments of equal cumulative weight.

```cpp
enum class SFCCurveType : int { Morton = 0, Hilbert = 1 };

SFCPartitioner(SFCCurveType ct = SFCCurveType::Morton);
```

**Three modes** identical in structure to `KDTreePartitioner`.

**Morton (Z-curve):** 21-bit per-axis quantisation, bit-interleave to 63-bit key. Cheapest to compute; good locality in most cases.

**Hilbert:** order-7 (128³ grid) Hilbert curve. Better locality preservation than Morton near partition boundaries, at the cost of a more complex key computation. Matches the baseline `sfc.py` implementation.

On CUDA, key computation and sort are done by `sfc_partition_cuda()`. CPU fallbacks are implemented directly in `sfc_partitioner.cpp` using ATen bit-manipulation ops.

**SFC vs KDTree tradeoff:** SFC is faster (one GPU sort vs recursive bisection) and tends to produce rounder partition shapes. KDTree adapts better to highly anisotropic distributions (e.g., a slab geometry). In practice on the DistNNP benchmark systems, both produce similar edge imbalance when auto-weighted.

### `build_halo_maps()` — algorithm

Implemented in `src/partition/build_halo_maps.cpp`. Shared by all three partitioners as a non-virtual method on `AbstractPartitioner`.

**Input:** `PartitionMessagePassingInfo &info` (with `partition_id` already set), `global_state` (all atom positions), `env`, `config`.

**Steps:**

1. **Build full radius graph** at `ghost_cutoff = cutoff + ghost_cutoff_buffer` using `radius_graph_pbc`. Produces `edge_index [2, E]` where `src` and `dst` are global atom indices.

2. **Propagate needed_by matrix** `[N, W]` bool. An atom `a` is `needed_by` rank `r` if: (a) `a` belongs to rank `r` (self-need), or (b) there exists an edge `(a → dst)` where `dst` belongs to rank `r` (1-hop propagation).

3. **Build send map (CSR):** For each locally owned atom, collect all target ranks it must be sent to (i.e., ranks that need it but are not this rank). Sort by target rank, build CSR offsets via `bincount` + `cumsum`.

4. **Build receive map (CSR):** Find all atoms owned by other ranks that this rank needs (i.e., `needed_by[:, R]` excluding own atoms). Sort by source rank, build CSR offsets.

5. **Build interior mask:** An atom is interior if none of its graph neighbours belong to a different rank. Computed by checking, for each edge `(src, dst)` landing in this rank, whether `src`'s partition differs from `R`.

6. **Store results** in `info` fields and update `global_send_recv_matrix[R, :]` with send counts (diagonal reset to `num_local`).

All tensor operations are ATen — they run on the same device as the input. The `radius_graph_pbc` call requires CUDA.

---

## Free Functions

The lower-level CUDA primitives are declared in their own headers and called from the partitioner and halo map code. They are also exposed to Python directly via the `radius_graph_pbc` binding in `dynamis_state_impl`.

### `radius_graph_pbc`

Declared in `include/dynamis_state/radius_pbc.hpp`. CUDA implementation in `src/radius_pbc/radius_pbc.cu`.

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor>
radius_graph_pbc(
    at::Tensor pos,                 // [N, 3] float32, CUDA
    at::Tensor cell,                // [3, 3] float32 — simulation cell matrix
    at::Tensor pbc,                 // [3]    bool    — axis-wise PBC flags
    float      radius,              // cutoff distance (Å)
    int64_t    max_num_neighbors,   // cap on neighbours per atom
    bool       enforce_max_strictly // if true, error when cap exceeded; else silently truncate
);
// Returns: (edge_index [2, E], edge_weight [E], num_neighbors [N])
```

**What it does:** Builds a directed neighbour graph over all `N` atoms. For each atom, it finds all atoms within `radius` Å, applying periodic boundary conditions selectively per axis (PBC image tiling). Uses a GPU cell-list for O(N) construction. Self-loops are excluded.

**Periodic image handling:** Only axes where `pbc[i] == 1` are tiled. Atoms near a periodic boundary are compared against images of neighbours across the cell, not just the primary image. The cell matrix allows non-orthogonal (triclinic) simulation boxes.

**Callers and their radii:**
- `KDTreePartitioner::partition(config)` and `SFCPartitioner::partition(config)` — called at `config.cutoff` to count edges for load balancing
- `AbstractPartitioner::build_halo_maps()` — called at `config.cutoff + config.ghost_cutoff_buffer` to build the communication maps

**Python binding:**
```python
edge_index, edge_weight, num_neighbors = dynamis_state_impl.radius_graph_pbc(
    pos, cell, pbc, radius=6.0, max_num_neighbors=50, enforce_max_strictly=False
)
```

### `kdtree_bisect_cuda`

Declared in `include/dynamis_state/kdtree_cuda.hpp`. CUDA implementation in `src/partition/kdtree_cuda.cu`.

```cpp
at::Tensor kdtree_bisect_cuda(
    at::Tensor pos,      // [N, 3] float32, CUDA
    at::Tensor weights,  // [N]    float32 or undefined (unweighted mode)
    int64_t    W         // number of output partitions
);
// Returns: partition_labels [N] int64 — value in [0, W)
```

**What it does:** GPU-accelerated recursive bisection. Mirrors the CPU `kdtree_bisect()` logic: find the longest-extent axis, sort atoms, split at the median (or weighted median), recurse on both halves. CUDA parallelism is exploited within each bisection level via parallel sort and prefix sums. The CPU path in `kdtree_partitioner.cpp` is a reference fallback used when the input tensors are on CPU (e.g., in unit tests without a GPU).

### `sfc_partition_cuda`

Declared in `include/dynamis_state/sfc_cuda.hpp`. CUDA implementation in `src/partition/sfc_cuda.cu`.

```cpp
at::Tensor sfc_partition_cuda(
    at::Tensor pos,         // [N, 3] float32, CUDA
    at::Tensor weights,     // [N]    float32 or undefined
    int64_t    W,           // number of output partitions
    int64_t    curve_type   // 0 = Morton, 1 = Hilbert
);
// Returns: partition_labels [N] int64 — value in [0, W)
```

**What it does:** On GPU, normalises positions into the unit cube, computes per-atom Morton or Hilbert keys (63-bit for Morton, order-7 for Hilbert), sorts by key with `thrust::sort`, and assigns partition labels by equal-weight bucket splitting. The weighted path accumulates a prefix sum of weights and assigns each atom to the bucket whose cumulative share crosses the proportional threshold.

The CPU fallbacks in `sfc_partitioner.cpp` implement the same key computation using ATen bit-manipulation ops and a simple element-wise loop for label assignment. They are numerically identical to the CUDA path.

---

## Data Flow Summary

```
MolecularInfo (global)
DistEnvironment
SimulationConfig
         │
         ▼
 partitioner.partition()
         │
         ▼
 PartitionMessagePassingInfo
   ├── partition_id [N]
   ├── partition_indices [num_local]
   └── global_send_recv_matrix diagonal
         │
         ▼
 partitioner.build_halo_maps()          ← uses radius_graph_pbc internally
         │
         ▼
 PartitionMessagePassingInfo (complete)
   ├── send_offsets / send_local_indices (CSR)
   ├── recv_offsets / recv_ghost_indices (CSR)
   ├── interior_mask [num_local]
   ├── edge_index_halo [2, E']
   └── global_send_recv_matrix (row[rank] filled)
         │
         ▼
 DistNNPState (passed to distnnp_model and dynamis_integrator)
```
