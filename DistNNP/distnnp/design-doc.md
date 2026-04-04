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
    traj_file: str = "trajectory.traj"  # Single shared output file (relative to output_dir)
    traj_interval: int = 100            # Enqueue a trajectory snapshot every N steps (0 = disabled)
    log_interval: int = 10              # Log energy/timing every N steps
    writer_buffer_size: int = 10        # Max snapshots held in memory before submit() blocks

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
6. **Output** — delegate all I/O to a `DistributedBufferedWriter`; all ranks write concurrently to a single shared trajectory file using `pwrite()` with vertex-ID-based byte offsets, so no cross-rank coordination or file locking is required.

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

   Constructs a `DistributedBufferedWriter` bound to the shared trajectory file (see below). On rank 0, writes the file front matter before any simulation steps. All ranks open the same file path for `pwrite()`. The writer's background thread is started here and runs for the lifetime of the simulation.

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

`DistributedBufferedWriter` handles all trajectory output for a single rank. All ranks write to a **single shared file** using the `pwrite()` system call with vertex-ID-based byte offsets, so no cross-rank locking or coordination is required during writes.

Snapshots are produced by the simulation loop at GPU speed but written to disk at storage speed. The writer bridges this gap by accumulating snapshots in an in-memory queue and draining it on a background thread, so the simulation thread is never blocked by I/O.

### Design

```
Simulation thread                   Background I/O thread
──────────────────                  ─────────────────────
integrator.step(...)          ─┐
writer.submit(step, state)     │    queue.get() → compute pwrite offsets → pwrite()
integrator.step(...)          ─┘
writer.submit(step, state)     │    queue.get() → ...
...                            │
writer.flush()  ───────────────┘    drain remaining items, close fd
```

The queue has a bounded capacity (`writer_buffer_size`). If the simulation produces snapshots faster than the background thread can write them, `submit()` blocks until space is available. For the initial implementation this is not expected to be a concern — the target deployment uses high-speed NFS where write throughput is sufficient.



## Overview of the I/O Requirements
- Random access of atom data by atom number at any frame, without reading the entire frame or trajectory.
- A **single** file for the entire trajectory 
- All data types aligned to a multiple of their size (64 bit alignment for 64-bit types, 32-bit alignment for 32-bit types)
- Endianess: little-endian (standard on x86-64) 


### API
### Trajectory file format

The trajectory uses a custom binary format designed for random access, concurrent multi-rank writes, and self-description without external metadata.

#### Front matter

Rank 0 writes a fixed-size front matter block at byte offset 0 before any simulation steps. No other rank writes to this region. Fields:

| Field | Type | Description |
|---|---|---|
| `magic` | `uint8[8]` | Magic string `DISTNNP\x00` identifying the file type |
| `version` | `uint32` | Format version (currently `1`) |
| `run_id` | `uint8[16]` | Unique run identifier (128-bit UUID, set at `setup()` time) |
| `author` | `uint8[64]` | Null-terminated metadata string (hostname, user, or job ID) |
| `total_atoms` | `uint64` | Global atom count; fixed for the lifetime of the simulation |
| `time_per_step` | `float32`| The amount of time elapsed per step |
| `steps_per_write` | `uint32` | The frequency of logging. Frame i is the `steps_per_write` * i-th step in the simulation |
| `bytes_per_atom` | `uint32` | Size of one atom record in bytes (see per-atom layout below) |
| `field_count` | `uint32` | Number of fields stored per atom |
| `field_descriptors` | `uint8[field_count × 32]` | Null-padded field names (e.g. `"position\0"`, `"velocity\0"`) |
| `front_matter_size` | `uint64` | Total byte size of this header block; used to locate frame 0 |


`front_matter_size` is always a multiple of 64 bytes for alignment.

#### Per-frame layout


Since `total_atoms` is constant throughout the simulation, every frame is the same size:

```
frame_size = total_atoms × 8 + total_atoms × 12 + total_atoms × 12
           = total_atoms × 32
```

Frame `i` starts at byte offset:

```
frame_offset(i) = front_matter_size + i × frame_size
```

Each frame starts with `vertex_number_list`, an array of `total_atoms` values of `uint64` vertex IDs (8 bytes/atom) for random access, followed by the per-atom records. Each process calculates a starting offset, which is the sum of previous processes' local atoms, and writes its `PartitionMessagePassingInfo.partition_indices`.

> Reasoning: This allows a reader to quickly locate the data for any given atom in any frame by computing the byte offset from the vertex ID, without needing to read the entire frame or maintain an external index. An `offset_map` can be computed per frame by mapping the vertex IDs to the index in this array.

***Note: Assuming all processes are numbered from [0, world_size). This requires all processes to know how many `local_atoms` each previous process has, to calculate the appropriate offset.***

Python code representation of the `local_frame_offset` calculation:

```python
local_frame_offset = 0
for prev_process in range(current_process):
    local_frame_offset += num_local_atoms[prev_process]
```
In practice this should be calculated and stored during initial partitioning and subsequent repartitionings.

For frame `i`, a process writes its contiguous array representing the `vertex_number_list` at byte offset:

```
frame_offset[i] + local_frame_offset * 8
```


Within each frame, the atoms are stored in arbitrary order. Each process writes its local data in sequential order.


Each frame therefore has the following layout:

| Field | Type | Description |
|---|---|---|
| `vertex_number_list` | `uint64[total_atoms]` | The ordering of the vertex IDs on this frame |
| `positions` | `float32[total_atoms x 3]` | The positions of the atoms on this frame. The atoms are ordered as per `vertex_number_list` |
| `velocities` | `float32[total_atoms x 3]` | The velocities of the atoms on this frame. The atoms are ordered as per `vertex_number_list` |



#### Per-atom record

Within a frame, the positions of atom `v` (by global vertex ID) are stored at:

```
pos_atom_offset(i, v) = frame_offset(i) + total_atoms × 8 + offset_map[v] × 12
```

The velocities for a given atom `v` (by global vertex ID) are stored at:

```
vel_atom_offset(i, v) = frame_offset(i) + total_atoms × 8 + total_atoms × 12 + offset_map[v] × 12
```

> Reasoning: The `positions` and `velocities` are stored in contiguous memory when updated by the `integrator`. Matching the ordering in memory and on disk will allow for reading and writing with a single system call.

From frame `i`, each process writes its `vertex_number_list`, `positions`, and `velocities` at the offset based on `frame_offset[i]` and `local_frame_offset`. The offsets are:
```
vertex_id_write_offset   = frame_offset(i) + local_frame_offset × 8
position_write_offset    = frame_offset(i) + total_atoms × 8 + local_frame_offset × 12
velocity_write_offset    = frame_offset(i) + total_atoms × 8 + total_atoms × 12 + local_frame_offset × 12
```


#### Random access cost

Reading a single atom from a given frame requires first reading the
`vertex_number_list` for that frame (total_atoms × 8 bytes) to build the
`offset_map`. For a 1M-atom system this is ~8 MB per frame. This is
considered acceptable for post-processing workflows, which will typically
build the map once per frame and reuse it across many atom lookups.

For analysis tools that need to track a small subset of atoms across many
frames, the recommended pattern is:
  1. Read the `vertex_number_list` for each frame of interest.
  2. Cache the offset_map per frame.
  3. Issue targeted reads for only the atoms needed.


#### Repartitioning and write offset recalculation

After a repartition event, the number of local atoms on each rank changes.
Since all per-process write offsets depend on local_frame_offset (the
cumulative atom count of all processes with rank less than the current process), this value must be
recalculated before the next frame write.

Required steps after repartitioning:

  1. Each process computes its new num_local_atoms.
  2. An allgather collects num_local_atoms from every process into a
     shared array of length world_size.
  3. Each process recomputes local_frame_offset as:
       local_frame_offset = sum(num_local_atoms[0 : process])
  4. The writer is updated with the new local_frame_offset and
     num_local_atoms before the next submit() call.

This is the same allgather that the initial partitioning performs, so no
new communication primitive is needed.
> Note on cross-frame consistency: repartitioning changes the vertex ordering between frames. Analysis tools that diff consecutive frames (e.g., displacement calculations) should be aware that the underlying rank layout may have changed.


#### Reader validation requirements

Before interpreting any data, a reader must validate the front matter
in the following order:

  1. Read the first 8 bytes and verify they match the magic string
     DISTNNP\0. If they do not match, reject the file.
  2. Read the version field (uint32). If the version is not recognized,
     the reader should refuse to parse the file rather than guess at
     the layout, unless it is a generic introspection tool using
     field_descriptors.
  3. Validate bytes_per_atom against the expected value for the format
     version. For v1 this is 24 (3×float32 position + 3×float32
     velocity). A mismatch indicates either corruption or a version
     the reader does not support.
  4. Validate that front_matter_size is a multiple of 64 bytes.
  5. Validate that total_atoms > 0.

Endianness: the format is little-endian throughout. Readers running on
big-endian architectures must byte-swap all multi-byte fields. Since the
target platform is x86-64 (natively little-endian), the reference reader
implementation does not perform byte-swapping but should include a
compile-time or runtime check:

    assert(system_is_little_endian(),
           "Big-endian host detected; byte-swapping not implemented");

A portable reader should check the first bytes of the magic string and
infer endianness from the version field: if reading version as
little-endian yields a recognized version number, the file is
little-endian. If not, try byte-swapping. This avoids adding an explicit
endianness marker to the header.

<!-- 
Simple per-atom layout (32 bytes):

| Field | Type | Bytes |
|---|---|---|
| `px, py, pz` | `float32 × 3` | 12 |
| `vx, vy, vz` | `float32 × 3` | 12 |

Total: **24 bytes/atom**. The field descriptors in the front matter reflect the actual fields stored. This requires combining the ids, positions, and velocities into a single CPU tensor before enqueuing for writing, but it keeps the on-disk format simple and compact. -->


<!-- #### Concurrent writes with `pwrite()`

Each rank computes the `atom_offset` for each of its owned atoms using the global vertex IDs assigned at simulation setup. It then calls `pwrite(fd, buf, nbytes, atom_offset)` — a single syscall that writes to an arbitrary file position without seeking, making it safe to call from multiple ranks simultaneously on the same open file descriptor.

Because atom records do not overlap (each vertex ID maps to a unique byte range), no file locking is needed. On POSIX-compliant NFS and Lustre filesystems, concurrent `pwrite()` calls to non-overlapping regions are atomic and do not require coordination. -->

#### C++ writer implementation

The writer is implemented in C++ and exposed to Python via pybind11. 
<!-- It uses `open(O_WRONLY | O_CREAT)` and `pwrite()` directly — no `fwrite`, no `fflush`, and no explicit `fsync`. The OS page cache handles write coalescing and durability. This avoids the overhead of `FILE*` buffering and eliminates the need for explicit flush commands in the Python simulation loop. -->

> **Assumed symbol**: `distnnp_model_impl.TrajectoryWriter` — C++ class with `write_front_matter(total_atoms, run_id, author)` and `write_frame(fd, frame_idx, vertex_ids, positions, velocities)` methods. To be implemented in `distnnp_model` or a new `distnnp_io` extension.

### Write bandwidth requirements

Per-atom record size: 32 bytes (vertex ID + positions + velocities).

| System size | Steps/s | Frame size | Required write bandwidth |
|---|---|---|---|
| 100k atoms | 10 | 3.2 MB | 32 MB/s |
| 100k atoms | 20 | 3.2 MB | 64 MB/s |
| 100k atoms | 50 | 3.2 MB | 160 MB/s |
| 1M atoms | 10 | 32 MB | 320 MB/s |
| 1M atoms | 20 | 32 MB | 640 MB/s |
| 1M atoms | 50 | 32 MB | 1.6 GB/s |

For reference: typical NFS throughput on HPC systems is 1–10 GB/s aggregate across clients. At 1M atoms / 50 steps/s each process writes `1.6 GB/s ÷ world_size`, which is comfortably within range for 4–16 GPU jobs. At larger scales or higher step rates, storage bandwidth becomes the binding constraint.

### Snapshot content

Each submitted snapshot captures, at minimum:

| Field | Shape | dtype |
|---|---|---|
| `vertex_ids` | `[num_local]` | int64 |
| `positions` | `[num_local, 3]` | float32 |
| `velocities` | `[num_local, 3]` | float32 |

Fields are copied from `state.local_partition_info` to CPU tensors on the simulation thread before being enqueued, so the background thread works entirely in CPU memory and never touches GPU tensors.

### Contention model

Because all ranks use `pwrite()` to non-overlapping byte regions, there is no write contention between ranks on a POSIX-compliant filesystem. The only lock is the Python-side queue between the simulation thread and the background I/O thread, which is a lightweight `threading.Queue` operation. Back-pressure (a full queue stalling the simulation thread) is not expected to be an issue on high-speed NFS systems in the initial deployment.

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
    traj_file="trajectory.traj",
    writer_buffer_size=10,
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
    │  CPU copy of vertex_ids/positions/velocities (simulation thread)
    └► background thread: compute pwrite() offsets by vertex_id → pwrite() to shared traj file
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
| `state.pos`, `state.vel`, `state.forces`, `state.mass`, `state.vertex_ids` | `distnnp_state` | Convenience aliases (or properties) on `DistNNPState` for the fields in `local_partition_info` |

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

4. **Trajectory format extensibility**: The current per-atom record is fixed at 32 bytes (vertex ID + positions + velocities). Adding fields (e.g. forces, per-atom energies) requires bumping `bytes_per_atom` and `version`, making old readers incompatible unless a migration path is defined. The `field_descriptors` array in the front matter provides enough metadata for readers to introspect fields, but a versioned field-extension scheme has not been designed yet.

5. **Writer back-pressure**: For the initial implementation on high-speed NFS systems, `submit()` blocks when the queue is full. This is the safe default (no data loss) and is not expected to stall the simulation in practice. A future `on_full: "block" | "drop"` policy option can be added if slow-storage deployments become a concern.

6. **Vertex ID assignment**: Each atom needs a globally unique, stable vertex number assigned once at `setup()` time and preserved across repartitioning. The simplest scheme is to number atoms 0…N-1 in the order they appear in the input file. This must be stored in `MolecularInfo` and tracked through `all_to_all` redistribution so each rank always knows the vertex IDs of its currently owned atoms.

