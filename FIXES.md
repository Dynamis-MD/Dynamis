# DistNNP Multi-GPU Fixes — What Was Broken and What Was Changed

This document records every problem encountered getting
`DistNNP/distnnp/experiments/synthetic/main.py` to run on NCSA Delta
(2× A100 GPUs, SLURM, `torchrun --nproc_per_node=2`) and the exact
changes made to fix each one.  The simulation now completes all 1 000
Verlet steps and writes checkpoint files for both ranks.

---


## 5. `DistNNPModel` had no `setup()` method — IPC exchanger was never initialised

**File:** `DistNNP/distnnp_model/src/distnnp_model/potential.py`

**Problem:**
`DistNNPModel.__init__` created a `FeatureExchanger()` object but never
called `setup_ipc()` or `setup_events()` on it.  Any call to
`FeaturePushToExchange.apply` raised
`RuntimeError: FeatureExchanger: IPC not set up`.

**Fix:** Add a `setup(state, feature_dim=3)` method to `DistNNPModel`
that:

1. Allocates two shared IPC buffers per rank:
   - `x_buffer` — holds `n_local + total_recv` rows of features
   - `grad_recv_buffer` — holds `total_send` rows for backward gradients
2. Exchanges IPC handles with all ranks via `dist.all_gather_object`.
3. Gathers each rank's `recv_offsets` and `n_local` counts.
4. Calls `self._feature_exchanger.setup_ipc(…)`.
5. Creates and exchanges CUDA IPC events via
   `create_and_export_event()` / `setup_events()`.

Key details that were non-obvious:
- `export_ipc_handle(buffer)` is an **instance** method, not static.
- `grad_recv_buffer` must have at least 1 row even when `total_send==0`
  (use `max(total_send, 1)`).
- The event exchange must happen *after* `setup_ipc`.

---



### 6e. Double IPC exchange for atom_types — CUDA event race corruption

The original code called `FeaturePushToExchange.apply` a second time
with the same `FeatureExchanger` to exchange atom types.  The exchanger
has a single set of IPC buffers and CUDA events; those events are
consumed by the first exchange (positions).  The second exchange runs
without valid synchronization, so each rank reads stale buffer contents.
`boundary_atom_types` contained garbage values (e.g. `613502353408`),
causing a CUDA device-side index-out-of-bounds assertion that surfaced
one kernel later at `sigma[type_i]`.

**Diagnosis note:** CUDA device-side assertions are deferred even with
`CUDA_LAUNCH_BLOCKING=1` — the error was *reported* at `sigma[type_i]`
but *triggered* in an earlier gather kernel.

**Fix:** Remove the second exchange.  Use `pi.recv_ghost_indices`, which
is already populated by the partitioner with the global atom IDs of
every ghost atom received by this rank:

```python
pi = distributed_state.local_partition_info
boundary_atom_types = pi.recv_ghost_indices
```


### 6g. Positions drift outside the simulation box, crashing `radius_graph_pbc`

Large LJ forces cause positions to drift far outside the box within a
few steps.  The C++ cell-list code computes the number of periodic
images from raw positions, hitting an int32 overflow at large
displacements (`Trying to create tensor with negative dimension …`).

**Fix 1 — fold positions into the unit cell before the neighbour search:**
```python
box_lengths = sim_config.cell.diagonal()
positions_folded = positions - torch.floor(positions / box_lengths) * box_lengths
# pass positions_folded to radius_graph_pbc_cuda
```

**Fix 2 — clamp forces to prevent runaway velocities:**
```python
forces = -local_positions.grad
forces = forces.clamp(-1.0, 1.0)
```

---

## 7. SLURM batch script (`run_test.sh`)

A new file was created to submit the job on Delta.  `CUDA_LAUNCH_BLOCKING=1`
was included during debugging to make CUDA errors synchronous; it can be
removed for production runs.

---

## Summary table

| # | File | What was wrong | What was changed |
|---|------|----------------|-----------------|
| 1 | `distnnp/experiments/synthetic/main.py` | Wrong import path (`distnnp.src.distnnp.simulator`) | Fixed to `distnnp.simulator` |
| 2 | `distnnp_integrator/src/distnnp_integrator/euler.py` | Imported non-existent `check_shapes`; called with wrong args | Changed to `check_shapes_match(pos, vel, frc)` |
| 3 | `distnnp/src/distnnp/simulator.py` | `check_integrator` compared instance to string list | Fixed to `type(integrator).__name__ not in …` |
| 4a | `distnnp/src/distnnp/simulator.py` | `DistNNPState` (C++ pybind11) has no dynamic attributes; `state.pos/vel/forces/mass` failed | Added `SimState` wrapper class; updated `_set_state` |
| 4b | same | Verlet reads `state.forces` before first force computation | Initialized `self.forces = torch.zeros_like(pos)` in `SimState.__init__` |
| 4c | same | `local_atom_masses` does not exist on `PartitionMessagePassingInfo`; property raised `AttributeError` caught by `__getattr__` | Initialized `self.mass = torch.ones(n_local, …)` as concrete Python attribute |
| 5 | `distnnp_model/src/distnnp_model/potential.py` | `FeatureExchanger` IPC never initialised | Added full `setup()` method with buffer allocation, handle exchange, and event setup |
| 6a | `distnnp_model/src/distnnp_model/empirical/lj_potential_model.py` | `distributed_state.simulation_config` does not exist | Changed to `distributed_state.sim_config` |
| 6b | same | `natoms=n_atoms` (int) instead of list | Changed to `natoms=[n_atoms]` |
| 6c | same | `cell` passed as `[3,3]` instead of `[1,3,3]` | Added `.unsqueeze(0)` |
| 6d | same | `epsilon`/`sigma` on CPU, `positions` on CUDA | Added `.to(positions.device)` |
| 6e | same | Second IPC exchange caused CUDA event race, returning garbage atom types | Replaced with `pi.recv_ghost_indices` |
| 6f | same | LJ singularity at r≈0 → NaN energy/forces | Added `distances.clamp(min=0.1)` |
| 6g | same | Drifting positions caused int32 overflow inside `radius_graph_pbc` | Fold positions into unit cell; clamp forces to `[-1, 1]` |
| 6h | `distnnp/experiments/synthetic/main.py` | `torch.randn` gives negative epsilon → `sqrt(neg)` = NaN | Changed to `torch.abs(torch.randn(…))` |
| 7 | `run_test.sh` (new file) | No SLURM script existed | Created with correct account/partition/GPU settings |
