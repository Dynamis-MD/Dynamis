# Synthetic End-to-End Experiment

This experiment runs DistNNP on a randomly generated atom system to verify the full distributed pipeline without requiring a real molecular structure file. It is the primary integration test for the `distnnp` package.

## What it tests

- Atom partitioning (SFC Morton) and halo map construction via `dynamis_state`
- Force and energy evaluation via a Lennard-Jones empirical potential (`distnnp_model`)
- Time integration and simulation loop via `distnnp` `Simulator`
- Correct single-GPU and multi-GPU operation

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — constructs `SimulatorConfig`, calls `create_dist_nnp_state`, and runs the simulation |
| `setup_system.py` | Helpers to build a synthetic `MolecularInfo` and a fully initialised `DistNNPState` |

## Setup

Install all sub-packages in the `distnnp-env` conda environment (from the repo root):

```bash
conda activate distnnp-env
pip install -e ./dynamis_state -e ./distnnp_integrator -e ./distnnp_model -e ./distnnp
```

## Running

### Single GPU

```bash
conda activate distnnp-env
cd distnnp/experiments/synthetic
python main.py
```

No launcher is required. `setup_system.py` detects the absence of `RANK` / `SLURM_PROCID` environment variables and skips `init_process_group`, running as a single-rank world.

### Multi-GPU (torchrun)

```bash
conda activate distnnp-env
cd distnnp/experiments/synthetic
torchrun --nproc_per_node=<NUM_GPUS> main.py
```

### Multi-GPU (SLURM)

```bash
srun --ntasks=<NUM_GPUS> --gpus-per-task=1 python main.py
```

## `setup_system.py` API

### `generate_synthetic_dataset(num_atoms) -> MolecularInfo`

Creates a random atom system of size `num_atoms`. Uses a fixed random seed (42) so every rank generates the identical positions and velocities — no broadcast needed. Positions are drawn uniformly from a 20 × 20 × 20 Å box; velocities are drawn from a zero-mean Gaussian with σ = 0.1 Å/fs. Tensors are placed on `cuda:0` when a GPU is available, otherwise on CPU.

### `create_dist_nnp_state(num_atoms) -> DistNNPState`

Builds a fully initialised `DistNNPState` ready to hand to `Simulator`. Steps performed:

1. **Distributed init** — calls `dist.init_process_group` only when launched via `torchrun` (`RANK` env var present) or SLURM (`SLURM_PROCID` env var present). Single-process runs skip this step and use `rank=0, world_size=1` directly.
2. **Synthetic structure** — calls `generate_synthetic_dataset` to produce a `MolecularInfo` that is identical on all ranks.
3. **Partitioning** — runs `SFCPartitioner(Morton)` to assign atoms to ranks, then calls `build_halo_maps` to construct the CSR send/recv tables for ghost-atom communication. Uses a 6.0 Å cutoff with a 0.5 Å ghost buffer, matching the `SimulatorConfig` defaults in `main.py`.
4. **State assembly** — populates and returns a `DistNNPState` containing the `DistEnvironment`, `SimulationConfig`, global `MolecularInfo`, and rank-local `PartitionMessagePassingInfo`.

### `sync_and_repartition(state) -> DistNNPState`

Gathers updated atom positions and velocities from all ranks, re-partitions the global system, and returns a new `DistNNPState` with atoms redistributed to their new owning ranks. Intended to be called periodically from the simulation loop:

```python
if step_num % sim_config.rebalance_interval == 0:
    dist_nnp_state = simulator.state
    repartitioned_state = sync_and_repartition(dist_nnp_state)
    simulator._set_state(repartitioned_state)
```

Steps performed:

1. **Single GPU** (`world_size=1`): writes updated local positions/velocities directly back into `global_molecular_info` by index — no communication.
2. **Multi-GPU**: uses `dist.all_gather_object` to collect each rank's `(partition_indices, positions, velocities)`, concatenates them, and sorts back to canonical atom order (0…N-1).
3. **Both paths**: re-runs `SFCPartitioner(Morton).partition` + `build_halo_maps` on the refreshed global positions and returns a new `DistNNPState`.
