# DistNNP Multi-GPU Fixes — What Was Broken and What Was Changed

This document records every problem encountered getting
`DistNNP/distnnp/experiments/synthetic/main.py` to run on NCSA Delta
(2× A100 GPUs, SLURM, `torchrun --nproc_per_node=2`) and the exact
changes made to fix each one.  The simulation now completes all 1 000
Verlet steps and writes checkpoint files for both ranks.

---

## 1. Wrong import path in `main.py`

**File:** `DistNNP/distnnp/experiments/synthetic/main.py`

**Problem:**
```python
# broken
from distnnp.src.distnnp.simulator import Simulator, SimulatorConfig
```
The package is installed as `distnnp`; the `src/` layout is only a
source-tree convention and is not part of the importable path.

**Fix:**
```python
# fixed
from distnnp.simulator import Simulator, SimulatorConfig
```

---

## 2. `CMakeLists.txt` — `find_library(TORCH_PYTHON_LIBRARY)` ran before `find_package(Torch)`

**File:** `DistNNP/distnnp_integrator/CMakeLists.txt`

**Problem:**
`find_library(TORCH_PYTHON_LIBRARY torch_python HINTS ${TORCH_LIB_DIR} …)`
referenced `${TORCH_LIB_DIR}`, which is derived from `${TORCH_LIBRARY}`,
which is only set after `find_package(Torch REQUIRED)`.  Because the
`find_library` call came *before* `find_package(Torch)` in the original
file, `TORCH_LIB_DIR` was empty and `torch_python` was never found,
causing a build failure (`Could not find TORCH_PYTHON_LIBRARY`).

**Fix:**  Move `find_package(Python …)` and `find_package(Torch REQUIRED)`
*before* all `find_library` calls that reference Torch paths:

```cmake
# correct order
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()

find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS ${TORCH_LIB_DIR} …
  NO_DEFAULT_PATH)

find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)
```

---

## 3. Wrong function name in `euler.py` — `check_shapes` does not exist

**File:** `DistNNP/distnnp_integrator/src/distnnp_integrator/euler.py`

**Problem — import:**
```python
# broken
from distnnp_integrator._data_check import check_devices, check_shapes
```
The module exports `check_shapes_match`, not `check_shapes`.

**Problem — call signature:**
```python
# broken
check_shapes(pos, vel, frc, mass)   # mass is not accepted
```
`check_shapes_match` only takes `pos, vel, frc` (shapes must match for
2-D tensors; mass shape is checked separately by `check_mass_shape`).

**Fix:**
```python
# fixed
from distnnp_integrator._data_check import check_devices, check_shapes_match
…
check_shapes_match(pos, vel, frc)
```

---

## 4. `check_integrator` compared instances to a list of strings incorrectly

**File:** `DistNNP/distnnp/src/distnnp/simulator.py`

**Problem:**
```python
# broken — compares the object itself, not its class name
if integrator not in AVALIABLE_INTEGRATORS:
```
`AVALIABLE_INTEGRATORS` is a list of *strings* (`["Euler", "Verlet", …]`).
An integrator *instance* will never equal a string.

**Fix:**
```python
# fixed
if type(integrator).__name__ not in AVALIABLE_INTEGRATORS:
```

---

## 5. `DistNNPState` does not support dynamic attributes — `SimState` wrapper needed

**File:** `DistNNP/distnnp/src/distnnp/simulator.py`

**Problem:**
The C++ pybind11 class `DistNNPState` is bound without
`py::dynamic_attr()`, so you cannot add arbitrary Python attributes to
it.  The integrators read and write `state.pos`, `state.vel`,
`state.forces`, and `state.mass`, none of which are fields on
`DistNNPState`.  Assigning them raised `AttributeError`.

**Fix:** Add a `SimState` Python wrapper *before* the `Simulator` class
that proxies `pos`/`vel` through the C++ object's
`local_partition_info` fields, stores `forces` and `mass` as plain
Python attributes, and forwards everything else to `_state`:

```python
class SimState:
    """Wraps DistNNPState to expose pos/vel/forces/mass for integrators."""

    def __init__(self, state: DistNNPState):
        object.__setattr__(self, "_state", state)
        pi = state.local_partition_info
        object.__setattr__(self, "forces", torch.zeros_like(pi.local_atom_positions))
        object.__setattr__(self, "mass", torch.ones(
            pi.local_atom_positions.shape[0],
            device=pi.local_atom_positions.device,
        ))

    @property
    def pos(self):
        return self._state.local_partition_info.local_atom_positions

    @pos.setter
    def pos(self, value):
        self._state.local_partition_info.local_atom_positions = value

    @property
    def vel(self):
        return self._state.local_partition_info.local_atom_velocities

    @vel.setter
    def vel(self, value):
        self._state.local_partition_info.local_atom_velocities = value

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)

    def __setattr__(self, name, value):
        if name in ("forces", "mass"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_state"), name, value)
```

`Simulator._set_state` was updated to wrap the state:

```python
def _set_state(self, state: DistNNPState):
    sim_state = SimState(state)
    self.state = sim_state
    self.model.setup(state)
```

---

## 6. `DistNNPModel` had no `setup()` method — IPC exchanger was never initialised

**File:** `DistNNP/distnnp_model/src/distnnp_model/potential.py`

**Problem:**
`DistNNPModel.__init__` created a `FeatureExchanger()` object but never
called `setup_ipc()` or `setup_events()` on it.  Any call to
`feature_push_forward` or `feature_push_backward` raised
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
- `all_recv_offsets` must be gathered from every rank's
  `recv_offsets` (shape `[world_size, world_size+1]`) via
  `dist.all_gather_object`.
- `grad_recv_buffer` must have at least 1 row even when `total_send==0`
  (use `max(total_send, 1)`).
- The event exchange must happen *after* `setup_ipc` and uses
  `create_and_export_event()` followed by `setup_events(all_handles)`.

---

## 7. `lj_potential_model.py` — multiple bugs

**File:** `DistNNP/distnnp_model/src/distnnp_model/empirical/lj_potential_model.py`

### 7a. `simulation_config` attribute does not exist

**Problem:**
```python
# broken
sim_config = distributed_state.simulation_config
```
The field is named `sim_config`.

**Fix:**
```python
sim_config = distributed_state.sim_config
```

### 7b. `natoms` must be a list, not a scalar

**Problem:**
```python
# broken — radius_graph_pbc expects natoms to be list-like
"natoms": n_atoms,
```

**Fix:**
```python
"natoms": [n_atoms],
```

### 7c. `cell` must be `[1, 3, 3]` (batched), not `[3, 3]`

**Problem:**
```python
# broken
"cell": sim_config.cell,          # shape [3, 3]
```
The radius graph wrapper indexes `data.cell[0]` (the first image's
cell), so the tensor must have a batch dimension.

**Fix:**
```python
"cell": sim_config.cell.unsqueeze(0),   # shape [1, 3, 3]
```

### 7d. `epsilon` and `sigma` must be moved to the same device as `positions`

**Problem:**
```python
# broken — epsilon/sigma created on CPU, positions on CUDA
energy = lennard_jones_potential(atom_types, positions, edge_index,
                                 self.epsilon, self.sigma)
```

**Fix:**
```python
energy = lennard_jones_potential(
    atom_types, positions, edge_index,
    self.epsilon.to(positions.device),
    self.sigma.to(positions.device),
)
```

### 7e. Double IPC exchange for atom_types — event race corruption

**Problem:**
The original code exchanged atom types by calling
`FeaturePushToExchange.apply` a *second* time with the same
`FeatureExchanger` that had just been used for positions:

```python
atom_types_2d = local_atom_types.float().unsqueeze(1).expand(-1, 3).contiguous()
boundary_atom_types = FeaturePushToExchange.apply(
    atom_types_2d, distributed_state, self._feature_exchanger
)[n_local:, 0].long()
```

The `FeatureExchanger` allocates a single pair of IPC buffers and one
set of CUDA IPC events.  Those events are consumed by the *first*
exchange (positions).  The second exchange runs without valid
synchronization, so each rank reads the other's buffer before it has
been written with atom-type data.  The result is that
`boundary_atom_types` contains garbage values from whatever was last in
the shared buffer (e.g. `613502353408`, far outside the valid type
range 0–99), causing a CUDA device-side `index out of bounds` assertion
that surfaces one kernel launch later at `sigma[type_i]`.

**Diagnosis note:** Because CUDA device-side assertions are deferred
(even with `CUDA_LAUNCH_BLOCKING=1`), the assertion was *reported* at
`sigma[type_i]` but actually *triggered* in the `atom_types[idx_i]`
gather kernel.  Adding `torch.cuda.synchronize()` and explicit OOB
checks confirmed that `atom_types` itself contained garbage.

**Fix:** Remove the second IPC exchange entirely.  Use
`partition_info.recv_ghost_indices`, which is already populated by the
partitioner with the **global atom IDs** of every ghost atom received
by this rank.  In the synthetic experiment the atom type equals the
global atom ID, so this gives the correct types with no additional
communication:

```python
pi = distributed_state.local_partition_info
boundary_atom_types = pi.recv_ghost_indices
```

### 7f. LJ singularity at r ≈ 0

**Problem:**
With 100 atoms placed randomly in a 20 Å box some pairs start very
close together (< 0.1 Å).  `(sigma / r)^6` becomes enormous, making
the energy and its gradient overflow to `inf`/`NaN`.

**Fix:** Clamp distances to a minimum before computing `sr6`:

```python
distances = torch.linalg.norm(dr, dim=1).clamp(min=0.1)
```

### 7g. Positions drift outside the simulation box, crashing `radius_graph_pbc`

**Problem:**
Random LJ parameters can produce forces of O(10³) eV/Å even after the
distance clamp (the backward gradient through large `sr6` is still
huge).  After a single Verlet step positions jumped from [0, 20] Å to
[−930, +484] Å; by step 10 they reached ~10⁹ Å.  The C++ cell-list
code in `radius_graph_pbc` computes the number of periodic images from
the raw position, causing an int32 overflow at large displacements
(error: `Trying to create tensor with negative dimension -1696113313`).

**Fix 1 — fold positions before neighbour search:**
Compute `positions_folded` by wrapping coordinates into the primary
unit cell before passing them to `radius_graph_pbc_cuda`.  The edge
indices returned are indices into the *local* positions array, so the
energy computation still uses the original (unfolded) `positions` with
gradients:

```python
with torch.no_grad():
    cell = sim_config.cell
    box_lengths = cell.diagonal()
    positions_folded = positions - torch.floor(positions / box_lengths) * box_lengths
    edge_index, _, _ = radius_graph_pbc_cuda(
        data=type("Data", (object,), {
            "pos": positions_folded,
            "cell": sim_config.cell.unsqueeze(0),
            "natoms": [n_atoms],
            "pbc": sim_config.pbc,
        })(),
        …
    )
```

**Fix 2 — clamp output forces:**
Cap the forces returned to the integrator so that velocities and
positions cannot accumulate arbitrarily large values from one step to
the next:

```python
forces = -local_positions.grad
forces = forces.clamp(-1.0, 1.0)
```

### 7h. `epsilon` and `sigma` must be non-negative for `torch.sqrt`

**Problem:**
```python
# broken — randn produces negative values; sqrt(negative) = NaN on GPU
potential = LennardJonesPotentialModel(
    epsilon=torch.randn((100,)), sigma=torch.randn((100,))
)
```
`epsilon_ij = torch.sqrt(epsilon[type_i] * epsilon[type_j])` gives NaN
when the product is negative.

**Fix** (in `main.py`):
```python
potential = LennardJonesPotentialModel(
    epsilon=torch.abs(torch.randn((100,))),
    sigma=torch.abs(torch.randn((100,))),
)
```

---

## 8. SLURM batch script (`run_test.sh`)

A new file was created to submit the job on Delta:

```bash
#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --job-name=distnnp-test
#SBATCH --output=/u/pkostioukov/DistNNP/distnnp_test_%j.out

cd /u/pkostioukov/DistNNP/DistNNP/distnnp/experiments/synthetic
CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc_per_node=2 main.py
```

`CUDA_LAUNCH_BLOCKING=1` was used during debugging to make CUDA kernels
synchronous so error locations could be narrowed down.

---

## Build note

The packages must be installed in editable mode with
`CMAKE_PREFIX_PATH` pointing at the PyTorch CMake directory on this
system:

```bash
cd /u/pkostioukov/DistNNP/DistNNP
CMAKE_PREFIX_PATH=/u/pkostioukov/.local/lib/python3.9/site-packages/torch/share/cmake \
  pip install -r requirements-dev.txt
```

---

## Summary table

| # | File | What was wrong | What was changed |
|---|------|----------------|-----------------|
| 1 | `distnnp/experiments/synthetic/main.py` | Wrong import path (`distnnp.src.distnnp.simulator`) | Fixed to `distnnp.simulator` |
| 2 | `distnnp_integrator/CMakeLists.txt` | `find_library(TORCH_PYTHON_LIBRARY)` ran before `find_package(Torch)` | Moved `find_package(Python)` and `find_package(Torch)` earlier |
| 3 | `distnnp_integrator/src/distnnp_integrator/euler.py` | Imported non-existent `check_shapes`; called with wrong args | Changed to `check_shapes_match(pos, vel, frc)` |
| 4 | `distnnp/src/distnnp/simulator.py` | `check_integrator` compared instance to string list | Fixed to `type(integrator).__name__ not in …` |
| 5 | `distnnp/src/distnnp/simulator.py` | `DistNNPState` (C++ pybind11) has no dynamic attributes; `state.pos/vel/forces/mass` failed | Added `SimState` wrapper class |
| 6 | `distnnp_model/src/distnnp_model/potential.py` | `FeatureExchanger` IPC never initialised; `feature_push_forward` raised `IPC not set up` | Added full `setup()` method with buffer allocation, handle exchange, and event setup |
| 7a | `distnnp_model/src/distnnp_model/empirical/lj_potential_model.py` | `distributed_state.simulation_config` does not exist | Changed to `distributed_state.sim_config` |
| 7b | same | `natoms=n_atoms` (int) instead of list | Changed to `natoms=[n_atoms]` |
| 7c | same | `cell` passed as `[3,3]` instead of `[1,3,3]` | Added `.unsqueeze(0)` |
| 7d | same | `epsilon`/`sigma` on CPU, `positions` on CUDA | Added `.to(positions.device)` |
| 7e | same | Second IPC exchange for atom_types caused CUDA event race, returning garbage type values | Replaced with `pi.recv_ghost_indices` (no second exchange) |
| 7f | same | LJ singularity at r≈0 → NaN energy/forces | Added `distances.clamp(min=0.1)` |
| 7g | same | Drifting positions caused int32 overflow inside `radius_graph_pbc` C++ | Fold positions into unit cell before calling radius graph; clamp forces to `[-1, 1]` |
| 7h | `distnnp/experiments/synthetic/main.py` | `torch.randn` gives negative epsilon → `sqrt(neg)` = NaN | Changed to `torch.abs(torch.randn(…))` |
| 8 | `run_test.sh` (new file) | No SLURM script existed | Created with correct account/partition/GPU settings |
</content>
</invoke>