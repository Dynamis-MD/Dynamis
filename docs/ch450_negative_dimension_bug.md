# ch450 "negative dimension in build graph" — root cause and fix

**Branch:** `writer-integration` (commit `e8bd0d6`)
**Reproduced:** 2026-06-26, 2× A100 (Delta gpua002), `torchrun --nproc_per_node=2 distnnp/experiments/ch450/main.py`
**Status:** fixed in working tree (two pure-Python files), verified, not committed.

## Symptom

Running the ch450 experiment (multi-GPU, `LennardJonesPotentialModel`) crashes with:

```
RuntimeError: Trying to create tensor with negative dimension -924662644: [-924662644]
  File ".../distnnp_state/radius_graph_pbc_cuda.py", line 57, in radius_graph_pbc_cuda
    edge_index, cell_offsets, num_neighbors = _impl.radius_graph_pbc(...)
```

The negative value changes from run to run — the signature of an **int32 overflow**, not a fixed bad size.

> Reproduction prerequisite: after `git checkout writer-integration` the installed `.so` files are stale (built from another branch — even `TrajectoryWriter`'s constructor signature differs). Rebuild all sub-packages with `pip install -e ./<pkg> --no-build-isolation` first.

## Root cause — a two-bug chain

### Bug 1 (the actual crash): LJ `atom_types` left 2D

In `distnnp_model/src/distnnp_model/empirical/lj_potential_model.py`, `atom_types` stays shape `[N, 1]` — the `reshape(-1, 1)` required by the exchanger is never undone. In `lennard_jones_potential` this makes `sigma_ij` shape `[E, 1]`, so:

```python
sr6 = (sigma_ij / distances) ** 6   # [E,1] / [E]  ->  broadcasts to [E, E]
```

Energy (and therefore forces) is inflated by ~E×. The integrator's `halfstep1` then flings atoms to ~1e5 Å. On the next `model()` call the radius graph computes its cell grid from these exploded coordinates, and

```cpp
int num_cells = grid_nx * grid_ny * grid_nz;   // radius_pbc.cu
... at::full({num_cells}, ...)                 // overflows int32 -> negative
```

overflows int32 to a negative value → "negative dimension".

### Bug 2 (multi-GPU setup crash + corrupted ghosts): exchanger buffer sizing

`GPUDirectExchanger.setup_exchanger` sized its IPC buffers from
`global_send_recv_matrix.sum(dim=0/1)[rank]`. But that matrix is only populated
for the **local rank's own row** (`build_halo_maps` fills row `R`; it is **never
all-reduced** in the pipeline — even the unit test hard-codes a full matrix
"as if all-reduced"). So the column-sum returns `num_local` instead of the true
recv count.

Measured at W=4 (450 atoms): `matrix_recv = 112` vs true `recv_offsets[-1] = 337`
on every rank → the ghost region is under-sized → IPC peers overflow `x_buffer`.
At W=1 it instead appends `num_local` phantom rows (zeros = origin atoms).
Additionally `send_local_indices = torch.zeros(...)` made every ghost equal the
sender's atom-0.

## Fixes

### `lj_potential_model.py`

```python
with torch.no_grad():
    # exchanger needs a 2D [N, 1] feature, but the LJ energy needs 1D [N] type
    # indices. Without reshape(-1), sigma_ij is [E,1] and sigma_ij / distances
    # broadcasts to [E,E] -> energy/forces inflated ~E x -> positions explode.
    atom_types = (
        FeaturePushToExchange.apply(
            local_atom_types.float(), distributed_state, self.atom_type_exchanger
        )
        .long()
        .reshape(-1)          # <-- added
    )
```

### `comm/exchanger.py` (in `setup_exchanger`)

```python
send_offsets = partition_info.send_offsets
recv_offsets = partition_info.recv_offsets

# Use this rank's own CSR offsets (build_halo_maps fills them correctly).
# global_send_recv_matrix is only locally populated (never all-reduced), so
# sum(dim=0)[rank] wrongly returns num_local -> under-sized ghost region.
total_recv = int(recv_offsets[-1].item())
total_send = int(send_offsets[-1].item())

# max(1, ...) because cudaIpcGetMemHandle rejects a 0-byte allocation (W=1).
x_buffer = FeatureExchanger.allocate_ipc_buffer(
    max(1, n_local + total_recv), feature_dim, rank
)
grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(
    max(1, total_send), feature_dim, rank
)
...
# use the real send map, not zeros
send_local_indices = partition_info.send_local_indices.to(
    device=device, dtype=torch.int64
)
```

Both changes are pure Python (no rebuild needed).

## Verification

| Check | Result |
|---|---|
| W=1 model energy vs ground-truth LJ over all 450 atoms | **1081634.875, bit-exact** |
| W=1 exchanger output rows | 450 → 450 (was 900, incl. 450 phantom origin rows) |
| W=2 ghost data | 225 distinct exact copies of real atoms (was: all identical = atom-0) |
| Full ch450 experiment, W=2 | **EXIT 0**, 10 steps, 20 trajectory writes |
| Original code, W=2 (fixes reverted) | reproduces the negative-dimension crash |
| Existing `test_gpu_direct_exchanger.py` | pre-existing failure (test never sets `partition_indices`); fails on original code too — not a regression |

## Out of scope (found, not fixed)

- **Cell hardcoded:** `simulator._get_simulation_config` sets a cubic
  `side_length_angstrom = 20` cell, and `load_xyz_file` ignores the file's
  `Lattice="15.0 ... 30.0 ... 30.0"` header. ch450 atoms span ~30 Å and get
  folded into a 20 Å box → wrong physics (does not crash).
- **mkdir race:** `TrajectoryWriter` constructor fails on `mkdir ./results` with
  EEXIST when both ranks create the directory simultaneously.
- **Defensive hardening:** `radius_pbc.cu` could use `int64`/guard `num_cells`
  so that exploded positions fail with a clear message instead of a cryptic
  negative dimension.
