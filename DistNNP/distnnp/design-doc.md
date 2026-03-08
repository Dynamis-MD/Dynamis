# `distnnp` Simulator Design Document

`distnnp` is the top-level entry point for the DistNNP framework. Its primary export is the `Simulator` class, which orchestrates distributed molecular dynamics by composing the three sub-packages:

- **`distnnp_state`** — atom partitioning, halo map construction, and the shared state struct (`DistNNPState`)
- **`distnnp_model`** — force and energy computation via a user-supplied `DistNNPModel`
- **`distnnp_integrator`** — time integration (Euler, Velocity Verlet, Langevin)

---

## Goals

- Provide a single, minimal entry point that a user can call to run a distributed MD simulation.
- Hide the details of process-group setup, partitioning, halo map construction, and repartitioning.
- Be composable: the model and integrator are injected by the user, not hardcoded.
- Match or exceed the baseline throughput (~7.2 steps/s on 4× A100, ~4,000 atoms).

---

## Configuration

All simulation parameters are expressed through a set of plain dataclasses. These can be constructed manually or loaded from a config file (e.g. via Hydra or YAML). The dataclasses are intentionally flat and do not contain framework objects.

```python
@dataclass
class SimulatorConfig:
    # Physical parameters
    timestep_fs: float = 0.5          # Integration timestep in femtoseconds
    total_steps: int = 1000           # Total number of MD steps to run

    # Spatial decomposition
    cutoff_angstrom: float = 6.0      # NNP neighbour cutoff radius (Å)
    ghost_buffer_angstrom: float = 0.5  # Extra radius for ghost atom selection (Å)
    partition_algorithm: str = "sfc_morton"
    # Valid values: "sfc_morton" | "sfc_hilbert" | "kdtree" | "simple"
    # "simple": index-order split, cheapest, good only for pre-sorted structures
    # "sfc_morton" / "sfc_hilbert": space-filling curve (recommended)
    # "kdtree": recursive KD-tree bisection, best for anisotropic geometries
    edge_weighted_partition: bool = True
    # When True, partition by dst-edge count (balanced GNN compute).
    # When False, partition by atom count.

    # Repartitioning
    rebalance_interval: int = 100     # Repartition every N steps (0 = never)

    # Output
    output_dir: str = "./results"
    traj_interval: int = 100          # Enqueue a trajectory snapshot every N steps (0 = disabled)
    log_interval: int = 10            # Log energy/timing every N steps
    writer_buffer_size: int = 10      # Max snapshots buffered in memory before a forced flush
    writer_flush_interval: int = 500  # Flush writer to disk every N steps regardless of buffer size

    # Distributed backend
    backend: str = "nccl"             # PyTorch distributed backend ("nccl" | "gloo")
```

---

## `Simulator` Class

### Responsibilities

1. **Distributed initialisation** — detect rank/world-size from environment (SLURM, torchrun, or single-process), set the GPU device, initialise the PyTorch process group.
2. **Structure loading** — read the input file on all ranks and broadcast atom data.
3. **Initial partitioning** — call the selected `AbstractPartitioner` to distribute atoms across ranks and build halo maps, populating a `DistNNPState`.
4. **MD loop** — advance the simulation `total_steps` steps; each step:
   a. Call `integrator.step(state, model)` — the integrator is responsible for calling `model(state)` at the right point(s) within the step (once for Euler, twice for Velocity Verlet and Langevin).
   b. Reduce energy across ranks for logging (on logging steps only).
   c. Enqueue a trajectory snapshot with the `DistributedBufferedWriter` (on output steps only).
5. **Periodic repartitioning** — every `rebalance_interval` steps, re-run partition + halo map build.
6. **Output** — delegate all I/O to a `DistributedBufferedWriter`; each rank writes independently to its own file.

### Constructor

```python
class Simulator:
    def __init__(
        self,
        model: DistNNPModel,
        integrator: DistNNPIntegrator,
        config: SimulatorConfig,
    ) -> None:
```

`model` and `integrator` are user-supplied instances. `Simulator` does not create them. This keeps the class composable and testable: the user picks the NNP and the time-integration scheme independently.

`__init__` does **not** perform any distributed communication. It only validates the config and stores the injected objects. All distributed setup is deferred to `setup()`.

---

### Lifecycle Methods

#### `Simulator.setup(input_file: str) -> None`

Performs all one-time initialisation. Must be called once before `run()` or `step()`.

Steps performed (in order):

1. **Detect distributed environment**

   Reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from the process environment (set by `torchrun` or SLURM). Calls `torch.distributed.init_process_group(backend=config.backend)`. Binds the process to `cuda:{local_rank}`.

   On a single process (`world_size == 1`), all distributed calls are skipped; the simulation runs in single-GPU mode without code changes.

   Populates a `DistEnvironment` struct.

2. **Load structure**

   Reads `input_file` (`.xyz` or compatible ASE-readable format) on all ranks. Returns a `MolecularInfo` containing positions, velocities, atom indices, and optionally edge counts. Velocities are initialised from a Maxwell–Boltzmann distribution at a user-specified temperature and broadcast from rank 0 so all ranks start from the same state.

   The cell matrix and PBC flags are also read and stored in a `SimulationConfig`.

   > **Assumed function**: `load_molecular_info(path, temperature_K, device) -> (MolecularInfo, SimulationConfig)` — to be implemented in `distnnp`.

3. **Partition atoms**

   Instantiates the chosen `AbstractPartitioner` subclass and calls:

   ```
   partition_info = partitioner.partition(global_mol_info, dist_env, sim_config)
   partitioner.build_halo_maps(partition_info, global_mol_info, dist_env, sim_config)
   ```

   Assembles the initial `DistNNPState`:

   ```
   state = DistNNPState()
   state.dist_env              = dist_env
   state.global_molecular_info = global_mol_info
   state.local_partition_info  = partition_info
   ```

4. **Atom masses**

   Looks up atomic masses (by atomic number) from a standard table and stores them as a per-rank tensor `[num_local]` float32 for use by the integrator.

   > **Assumed function**: `masses_from_atomic_numbers(atom_indices: Tensor) -> Tensor` — to be implemented in `distnnp`.

5. **Output directory setup**

   Creates `config.output_dir` on rank 0 (with a barrier so other ranks do not proceed before it exists).

6. **Writer initialisation**

   Constructs a `DistributedBufferedWriter` bound to this rank's output file (see below). The writer's background thread is started here and runs for the lifetime of the simulation.

---

#### `Simulator.run() -> None`

Runs the full MD simulation for `config.total_steps` steps. Equivalent to calling `step()` in a loop with repartitioning and output logic woven in.

```
while total_done < config.total_steps:
    if should_rebalance(total_done):
        _repartition()
    energy = _step()
    if should_log(total_done):
        _log_energy(total_done, energy)
    if should_write_traj(total_done):
        writer.submit(total_done, state)
    total_done += 1

writer.flush()   # drain any remaining buffered snapshots at end of run
```

Repartitioning (`should_rebalance`) triggers when `rebalance_interval > 0` and `total_done % rebalance_interval == 0`. The first repartition is at step 0 (i.e., before any integration), which means `setup()` performs the initial partition and `run()` re-checks immediately on the first step before calling the integrator. This avoids duplicating partition logic between `setup()` and `run()`.

> **Design note:** `setup()` does the first partition so that a user calling `step()` manually (without `run()`) has a valid state immediately after `setup()`.

---

#### `Simulator.step() -> float`

Advances the simulation by one timestep. Returns `local_energy` (the rank-local energy scalar, before any cross-rank reduction). Intended for users who want manual control of the loop (e.g., custom logging, adaptive timestep, mid-loop interventions).

Calling `step()` does **not** trigger repartitioning or output — those are the caller's responsibility when using the manual step interface.

Internal sequence:

```
integrator.step(state, model, dt=config.timestep_fs)
```

The integrator is fully responsible for the model call(s) and state mutation. The `Simulator` does not call `model(state)` directly. The energy returned to `Simulator` is retrieved from `state` after the integrator has completed (the integrator stores the last computed energy on `state.energy`).

**How each integrator handles the model call:**

| Integrator | Model call pattern |
|---|---|
| `Euler` | `model(state)` once at the start of the step; then advances pos and vel |
| `Verlet` | `halfstep1_()` → `model(state)` → `halfstep2_()` (two-half-step) |
| `Langevin` | `halfstep1_(noise1)` → `model(state)` → `halfstep2_(noise2)` (BBK) |

**Forces vs halo atoms:** Each integrator only updates `state.pos` and `state.vel` for the `num_local` owned atoms. The model is expected to return forces shaped `[num_local, 3]` with ghost contributions already accumulated via autograd on the owned atoms, as guaranteed by the `energy.backward()` design in `DistNNPModel.__call__`.

**Noise generation for Langevin:** The `Langevin` integrator generates its own `torch.randn_like(state.vel)` noise tensors internally at each half-step. The `Simulator` does not supply noise.

---

### Internal Methods

#### `Simulator._repartition() -> None`

Re-runs the full partition + halo map pipeline on the current atom positions. Performs an `all_to_all` to redistribute atoms to their new owning ranks and updates `state.local_partition_info` and `self._masses` in place.

The sequence is:
1. Gather current positions/velocities from `state.local_partition_info`.
2. Call `partitioner.partition()` → `partitioner.build_halo_maps()`.
3. Redistribute owned-atom tensors (positions, velocities, atom indices, masses) via `torch.distributed.all_to_all_single` according to the new `partition_id`.
4. Update `state.local_partition_info` with the redistributed data and new halo maps.

> **Assumed function**: `redistribute_atoms(local_data, partition_id, dist_env) -> redistributed_data` — to be implemented in `distnnp`.

#### `Simulator._log_energy(step: int, local_energy: float) -> None`

Performs an `allreduce` (sum) of `local_energy` across all ranks to obtain the total system energy, then prints step, total energy, wall-clock time per step, and steps/second. Only rank 0 prints. The `allreduce` is skipped on non-logging steps; `step()` only returns the local energy and the reduction is deferred to logging time.

Timing uses CUDA events for accurate GPU-side measurement. The measured interval is from the start of one `integrator.step()` call to the start of the next, capturing both compute and any synchronisation cost.

---

## `DistributedBufferedWriter`

### Overview

`DistributedBufferedWriter` handles all trajectory and diagnostic output for a single rank. Each rank owns one writer instance that writes to its own file — there is no cross-rank coordination during writes, which eliminates lock contention on shared file handles.

Snapshots are produced by the simulation loop at GPU speed (~millions of floats per step) but written to disk at storage speed. The writer bridges this gap by accumulating snapshots in an in-memory queue and draining it on a background thread, so the simulation thread is never blocked by I/O.

### Design

```
Simulation thread                  Background I/O thread
──────────────────                 ─────────────────────
integrator.step(...)         ─┐
writer.submit(step, state)    │    queue.get() → serialize → write → flush (periodic)
integrator.step(...)         ─┘
writer.submit(step, state)    │    queue.get() → ...
...                           │
writer.flush()  ──────────────┘    drain remaining items, close file
```

The queue has a bounded capacity (`writer_buffer_size`). If the simulation produces snapshots faster than the background thread can write them, `submit()` blocks until space is available. This provides back-pressure and prevents unbounded memory growth, at the cost of stalling the simulation — in practice this only happens when storage I/O is significantly slower than one step per `traj_interval` steps.

### API

```python
class DistributedBufferedWriter:
    def __init__(
        self,
        output_path: str,          # e.g. "results/rank_2.traj"
        buffer_size: int = 10,     # max snapshots held in memory before blocking
        flush_interval: int = 500, # force a disk flush every N submitted snapshots
    ) -> None:
        ...

    def submit(self, step: int, state: DistNNPState) -> None:
        """
        Enqueue a snapshot of the current state for writing.
        Copies the relevant tensors to CPU immediately (on the simulation thread)
        so that subsequent in-place updates by the integrator do not corrupt the
        snapshot. The actual disk write happens on the background thread.

        Blocks if the internal queue is full (back-pressure).
        """
        ...

    def flush(self) -> None:
        """
        Block until all enqueued snapshots have been written and the output
        file has been flushed to disk. Must be called before process exit
        to guarantee no data is lost. Called automatically by run().
        """
        ...

    def close(self) -> None:
        """
        Flush and shut down the background thread. After close(), submit()
        must not be called. Implicitly called by flush() when used as a
        context manager.
        """
        ...
```

### Snapshot content

Each submitted snapshot captures, at minimum:

| Field | Shape | dtype |
|---|---|---|
| `step` | scalar | int64 |
| `positions` | `[num_local, 3]` | float32 |
| `velocities` | `[num_local, 3]` | float32 |
| `atom_indices` | `[num_local]` | int64 |
| `local_energy` | scalar | float32 |

Fields are copied from `state.local_partition_info` to CPU tensors on the simulation thread before being enqueued, so the background thread works entirely in CPU memory and never touches GPU tensors.

### Output format

The output format has not been decided. A custom binary format is the likely direction, as it gives full control over layout, header structure, append semantics, and alignment — without the overhead or constraints of general-purpose container formats.

The key requirements any format must satisfy:

- **Append without rewrite**: each new snapshot is appended sequentially; the file does not need to be re-read or re-written to add a record.
- **Fixed-size records**: a record header (step index, atom count, field count, byte offsets) followed by raw float32/int64 payloads. Fixed-size records allow O(1) seeking to any step by index.
- **No cross-rank synchronisation**: each rank writes its own file independently. There is no shared index, no collective operation, and no file locking.
- **Crash recovery**: a partially written record at the end of a file (e.g., after a job preemption) must be detectable. A magic footer or record-length field written after the payload enables this.
- **Post-processing**: per-rank files can be merged offline (step-aligned, by global atom index) to reconstruct the full trajectory. The merge step does not need to be real-time.

The `DistributedBufferedWriter` interface is intentionally format-agnostic: `submit()` enqueues a CPU-side snapshot dict and the background thread calls a serialiser. Swapping the serialiser (e.g., from a prototype NPZ writer to the production binary format) does not change the writer's threading model or the caller API.

### Flush policy

The background thread flushes (calls the file's `flush()` / `fsync()` equivalent) in two situations:
1. Every `flush_interval` snapshots written, to limit data loss on a crash.
2. When `flush()` is called by the simulation (end of run, or checkpoint).

Individual `submit()` calls do not trigger a flush. This keeps per-step I/O cost to a queue append + CPU tensor copy, with no syscall on the simulation hot path.

### Contention model

Because each rank writes to its own file, there is no write contention between ranks on a shared filesystem. The only contention is between the simulation thread and the background I/O thread for the queue lock, which is a lightweight `threading.Queue` operation and not a bottleneck. On NFS or Lustre filesystems, per-rank files also avoid the metadata bottleneck that would arise from all ranks writing to a single shared file simultaneously.

---

## Usage Example

```python
from distnnp import Simulator, SimulatorConfig
from distnnp_model import LennardJonesPotentialModel
from distnnp_integrator import Langevin

config = SimulatorConfig(
    timestep_fs=0.5,
    total_steps=5000,
    cutoff_angstrom=6.0,
    partition_algorithm="sfc_morton",
    edge_weighted_partition=True,
    rebalance_interval=100,
    traj_interval=100,
    log_interval=10,
    output_dir="./md_results",
    writer_buffer_size=10,
    writer_flush_interval=500,
)

model      = LennardJonesPotentialModel(epsilon=0.01, sigma=3.4)
integrator = Langevin(dt=0.5, gamma=0.01, kT=0.02585)  # kT in eV at 300 K

sim = Simulator(model, integrator, config)
sim.setup("./system.xyz")
sim.run()
```

For manual control:

```python
sim.setup("./system.xyz")
for step in range(5000):
    local_energy = sim.step()
    if step % 100 == 0:
        total_energy = sim.reduce_energy(local_energy)
        if sim.rank == 0:
            print(f"Step {step}: E = {total_energy:.4f} eV")
    if step % 50 == 0:
        sim._repartition()
    if step % 100 == 0:
        sim.writer.submit(step, sim.state)
sim.writer.flush()
```

---

## Data Flow Through a Single Step

The integrator is the outermost coordinator. `Simulator.step()` delegates entirely to `integrator.step(state, model)`.

```
Simulator.step()
  │
  └── integrator.step(state, model, dt)
        │
        │   [Euler]
        ├── model.gather_halo(state)           # GPU-to-GPU halo exchange
        │     Appends ghost atoms to state positions: [num_local + num_ghost, 3]
        ├── energy, forces = model(state)      # NNP forward pass
        │     forces: [num_local, 3] (owned atoms only, ghost grads accumulated by autograd)
        ├── state.energy = energy
        └── CUDA kernel: pos += vel*dt + 0.5*(F/m)*dt²
                         vel += (F/m)*dt

        │   [Verlet]
        ├── halfstep1_(pos, vel, forces, mass, dt)    # v += F/m * dt/2;  pos += v * dt
        ├── model.gather_halo(state)
        ├── energy, forces = model(state)
        ├── state.energy = energy
        └── halfstep2_(vel, forces, mass, dt)         # v += F/m * dt/2

        │   [Langevin (BBK)]
        ├── noise1 = randn_like(vel)
        ├── halfstep1_(pos, vel, forces, mass, noise1, gamma, kT, dt)
        ├── model.gather_halo(state)
        ├── energy, forces = model(state)
        ├── state.energy = energy
        ├── noise2 = randn_like(vel)
        └── halfstep2_(vel, forces, mass, noise2, gamma, kT, dt)

  └── returns state.energy   (local, not yet reduced)


(on logging steps only)
  allreduce(state.energy) → total_energy printed by rank 0

(on traj_interval steps)
  writer.submit(step, state)
    │  CPU copy of positions/velocities/indices/energy (simulation thread)
    └► background thread: serialize → append to rank_{rank} output file
```

---

## Repartition Decision Logic

The default policy is time-based: repartition every `rebalance_interval` steps unconditionally. A threshold-based policy (repartition only when imbalance exceeds a configurable fraction) is a planned future extension.

The cost of repartitioning is one `all_to_all` for atom redistribution plus one `radius_graph_pbc` call for halo map construction. At ~4,000 atoms on 4 GPUs this is negligible at an interval of 100 steps. At larger atom counts or very small intervals this cost grows and the threshold policy becomes preferable.

---

## Distributed Initialisation Details

The `Simulator` supports three launch modes, detected automatically from environment variables:

| Mode | Environment | Detection |
|---|---|---|
| `torchrun` | `RANK`, `LOCAL_RANK`, `WORLD_SIZE` set by launcher | `LOCAL_RANK` present in env |
| SLURM | `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID` | `SLURM_PROCID` present in env |
| Single-process | None of the above | `world_size` defaults to 1 |

NCCL init uses a file-based rendezvous (`file://`) for SLURM jobs and the default `env://` init method for `torchrun`. In both cases, `torch.cuda.set_device(local_rank)` is called before `init_process_group` to ensure NCCL binds to the correct GPU.

---

## Assumed Future Interfaces

The following functions and methods are referenced in this document but are not yet implemented. They are treated as contracts to be fulfilled during implementation.

| Symbol | Package | Description |
|---|---|---|
| `load_molecular_info(path, temperature_K, device)` | `distnnp` | Reads structure file, initialises velocities, returns `(MolecularInfo, SimulationConfig)` |
| `masses_from_atomic_numbers(atom_indices)` | `distnnp` | Returns float32 mass tensor from a lookup table |
| `redistribute_atoms(local_data, partition_id, dist_env)` | `distnnp` | `all_to_all`-based atom redistribution after repartition |
| `DistNNPModel.gather_halo(state)` | `distnnp_model` | Executes GPU-to-GPU halo exchange; appends ghost atoms to position tensors in `state` |
| `state.energy` | `distnnp_state` | Field on `DistNNPState` for the integrator to store the last computed local energy |
| `state.pos`, `state.vel`, `state.forces`, `state.mass` | `distnnp_state` | Convenience aliases (or properties) on `DistNNPState` for the fields in `local_partition_info` |

**Integrators (implemented):**

| Class | Package | Notes |
|---|---|---|
| `Euler` | `distnnp_integrator` | `step(state, model, dt=None)` — single model call, first-order |
| `Verlet` | `distnnp_integrator` | `step(state, model, dt=None)` + `halfstep1_()` / `halfstep2_()` — NVE |
| `Langevin` | `distnnp_integrator` | `step(state, model, dt=None)` + `halfstep1_(noise)` / `halfstep2_(noise)`, `gamma`, `kT` — NVT (BBK) |

---

## Open Design Questions

1. **Masses and energy on `DistNNPState`**: Currently `DistNNPState` does not carry `masses` or `energy`. The integrators reference `state.mass` and `state.energy` directly. Options: (a) add these fields to `MolecularInfo` / `DistNNPState` in `distnnp_state`, or (b) define them as properties on a thin Python wrapper around `DistNNPState` in `distnnp`. Option (a) is cleaner for checkpointing; option (b) avoids touching the C++ struct.

2. **`state.pos` / `state.vel` aliases**: The integrators access `state.pos`, `state.vel`, `state.forces`, `state.mass` directly. These must resolve to `state.local_partition_info.local_atom_positions` etc. Whether these are added to the C++ `DistNNPState` struct or provided as Python properties needs to be decided in coordination with `distnnp_state`.

3. **Checkpoint / restart**: `run()` has no restart support yet. A checkpoint mechanism would write `state.local_partition_info` (positions, velocities, atom indices) plus the step counter to disk via the `DistributedBufferedWriter`, so a run can be resumed without re-reading the input file from scratch.

4. **Trajectory output format**: The binary format has not been specified. The key design decisions are: record layout (fixed-size vs length-prefixed), header fields (step, atom count, field descriptors), alignment for direct memory mapping, and crash-recovery sentinel values. The `DistributedBufferedWriter` interface is format-agnostic and the serialiser is a swappable implementation detail.

5. **Writer back-pressure vs drop**: When `buffer_size` is full, `submit()` currently blocks. An alternative is to drop the snapshot and emit a warning. Blocking is safer for correctness (no data loss) but can stall the simulation on a slow filesystem. A configurable `on_full: "block" | "drop"` policy could cover both use cases.
