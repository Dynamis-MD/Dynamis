# HIV Capsid MD Simulation

Runs a Lennard-Jones molecular dynamics simulation of the full HIV-1 capsid (PDB: 3J3Q, ~2.4 M atoms) distributed across one or more GPUs, using an NVT Langevin thermostat.

## What it tests

- End-to-end DistNNP pipeline at large scale (~2.4 M atoms) on a real, non-synthetic structure
- Atom partitioning (SFC Morton, edge-weighted) and halo map construction via `distnnp_state`
- Force and energy evaluation via a Lennard-Jones empirical potential (`distnnp_model`), with per-element ε/σ parameters
- NVT time integration via the Langevin (BBK) integrator in `distnnp_integrator`

## Files

| File | Purpose |
|---|---|
| `retrieve_data.py` | Downloads the 3J3Q mmCIF structure from RCSB and converts it to a heavy-atom-only `.xyz` file |
| `setup_system.py` | Loads the `.xyz` file into a `MolecularInfo`, sizing the simulation box and drawing Maxwell-Boltzmann velocities at 310 K |
| `main.py` | Entry point — builds the LJ potential and Langevin integrator, constructs `SimulatorConfig`, and runs the simulation loop |

## Setup

1. Download and convert the structure (one-time, run from this directory):

   ```bash
   python retrieve_data.py
   ```

   Downloads `3J3Q.cif.gz` (~40 MB compressed) from RCSB and writes `3J3Q.xyz` (heavy atoms only — hydrogens are absent from the cryo-EM structure and are not added).

2. Install all sub-packages (from the `DistNNP/` repo root):

   ```bash
   pip install -e ./distnnp_state -e ./distnnp_integrator -e ./distnnp_model -e ./distnnp
   ```

## Running

### Single GPU

```bash
cd distnnp/experiments/proteins
python main.py
```

### Multi-GPU (torchrun)

```bash
cd distnnp/experiments/proteins
torchrun --nproc_per_node=<NUM_GPUS> main.py
```

### Multi-GPU (SLURM)

```bash
srun --ntasks=<NUM_GPUS> --gpus-per-task=1 python main.py
```

Results (trajectory / logs) are written to `./results`.

## Configuration

Key parameters in `main.py`:

- `TOTAL_STEPS = 100`, `timestep_fs = 1.0`, `cutoff_angstrom = 10.0`, `ghost_buffer_angstrom = 0.5`
- `partition_algorithm = "sfc_morton"` with `edge_weighted_partition=True`
- LJ `epsilon`/`sigma` tables keyed by atomic number Z (source: [Rappe et al. 1992, UFF](https://pubs.acs.org/doi/10.1021/ja00051a040))
- Langevin NVT thermostat: `KT = 2.577e-4` amu·Å²·fs⁻² (310 K) and `GAMMA = 0.01` fs⁻¹ — must match the `kT` used for velocity initialisation in `setup_system.py`

Every `log_interval` steps (default 5), rank 0 prints instantaneous kinetic temperature, max velocity, max force, and GPU memory usage.

## Limitations

- Hydrogen atoms are absent (heavy-atom only; see `retrieve_data.py`).
- No equilibration phase; velocities are initialised from a Maxwell-Boltzmann distribution at 310 K with centre-of-mass drift removed. A production simulation would require energy minimization, ~1 ns NVT equilibration, and ~1 ns NPT equilibration before collecting data (Perilla et al. 2017).
- No barostat; simulation runs NVT Langevin only.
- UFF LJ (Rappe et al. 1992) is used as a placeholder potential. Bonded pairs are excluded by distance threshold (< 2.2 Å).
- The Langevin friction (`GAMMA = 0.01` fs⁻¹) has not been tuned/validated against a reference thermostat response.
