# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DistNNP** is a distributed neural network potential (NNP) framework for large-scale molecular dynamics (MD) simulations on multi-GPU systems. It computes forces and energies via ML potentials, partitions atoms across GPUs, communicates boundary atoms via halo exchange over NCCL, and integrates equations of motion using CUDA kernels.

This repository is a monorepo of four installable sub-packages, built from a working reference implementation (`baseline/`, not tracked in this repo).

## Development Setup

From the repository root:

```bash
uv pip install torch triton pytest scipy scikit-build-core pybind11
uv pip install --no-build-isolation -e ./dynamis_state -e ./dynamis_integrator -e ./dynamis_model -e ./dynamis
```

See `README.md` for details (plain pip works too). Each sub-package must be individually reinstalled after C++ or CUDA changes:

```bash
pip install -e ./dynamis_state
pip install -e ./dynamis_integrator
pip install -e ./dynamis_model
pip install -e ./dynamis
```

### CUDA location

Set `CUDA_HOME` (HPC/SLURM) or `CUDAToolkit_ROOT` if CUDA is not found automatically. On conda environments neither is usually needed.

## Running Tests

```bash
pytest                              # all tests
pytest dynamis_integrator/tests/    # single sub-package
pytest dynamis_integrator/tests/test_euler.py -v  # single file
```

Tests that require CUDA are skipped automatically when no GPU is present.

## Repository Structure

```
Dynamis/                        # repository root
├── dynamis_state/              # C++/CUDA: atom state, partitioning, halo maps
├── dynamis_integrator/         # C++/CUDA + Triton: MD time integration kernels
├── dynamis_model/              # Python + C++/CUDA: distributed NNP model interface, exchangers
├── dynamis/                    # Python: main user-facing entry point
├── benchmarks/                 # standalone kernel/communication benchmarks
├── experiments/                # paper experiments and dataset generation
├── post_processing/            # trajectory rendering, timing summaries
├── refs/                       # git submodules: nvalchemi, batmobile
└── pytest.ini
```

## Sub-package Status and Architecture

### `dynamis_state`
C++/CUDA extension. Defines the core data structures shared by all other packages.

Key types (in `include/dynamis_state/dynamis_state.hpp`):
- `MolecularInfo` — positions, velocities, atom indices for a partition
- `DistEnvironment` — rank, world size, GPU topology
- `PartitionMessagePassingInfo` — partition indices + halo map + send/recv matrix
- `DistNNPState` — aggregates the above

Partitioner interface (`include/dynamis_state/partitioner.hpp`): `AbstractPartitioner` with concrete subclasses `SimpleSpatialPartitioner`, `KDTreePartitioner`, `SFCPartitioner`.

Python classes are re-exported at the top level: `from dynamis_state import MolecularInfo, DistNNPState, ...`

Stubs are generated via `scripts/stub_gen.sh` (runs `pybind11-stubgen`).

### `dynamis_integrator`
Advances positions and velocities by one timestep. Implemented: Euler, Verlet, Langevin (BBK).

Two kernel backends with identical class/method names (`EulerIntegrator.step`, `VerletIntegrator.halfstep1/halfstep2`, `LangevinIntegrator.halfstep1/halfstep2`):
- **CUDA** — C++ extension `dynamis_integrator_impl` (`src/csrc/*.cu`, bound in `src/dynamis_integrator_bindings.cpp`). Only built when CMake finds a CUDA compiler.
- **Triton** — `src/dynamis_integrator/_kernels.py` (`@triton.jit`, compiled on first use).

`src/dynamis_integrator/_backend.py` picks CUDA if `dynamis_integrator_impl` imports, else Triton; `dynamis_integrator.BACKEND` reports which. The Python integrators (`euler.py`, `verlet.py`, `langevin.py`) import their kernel classes from `_backend`.

To add a new integrator, implement it in both backends with the same class/method names (see `dynamis_integrator/README.md`).

Tests live in `dynamis_integrator/tests/` and run against whichever backend is installed.

### `dynamis_model`
Pure Python (C++ extension stub present but empty). Defines the model interface.

`DynamisModel` ABC (`src/dynamis_model/potential.py`):
```python
__call__(distributed_state: DistNNPState) -> Tuple[energy, forces]
```

Design: each GPU (1) fetches halo atoms from neighbours according to `DistNNPState`, (2) evaluates the NNP locally to get a scalar energy, (3) calls `energy.backward()` to obtain forces via autograd (`F = -dE/dx`). For GNN-based NNPs, halo atoms must be re-fetched once per message-passing layer since updated embeddings on one partition are inputs to the next layer on a neighbouring partition.

### `dynamis` (stub)
Empty entry point. Will orchestrate the other sub-packages.

## Build System

All C++ sub-packages use the same pattern:
- **Backend**: `scikit-build-core` (PEP 517), build artefacts in `skbuild/`
- **Extension**: `python_add_library(... MODULE WITH_SOABI ...)` from CMake `FindPython`
- **Editable mode**: `editable.mode = "inplace"` — `.so` is placed in `src/<pkg>/` alongside `__init__.py`
- **Torch linking**: `${TORCH_LIBRARIES}` + explicit `${TORCH_PYTHON_LIBRARY}` (found via `find_library`). Do **not** link standalone `pybind11::module` — torch bundles its own pybind11 and mixing them causes undefined `type_caster<at::Tensor>` symbols at import time.
- **CUDA discovery**: `find_package(CUDAToolkit)` runs before `find_package(Torch)` and caches `CUDA_TOOLKIT_ROOT_DIR` / `CUDA_INCLUDE_DIRS` with `FORCE` so PyTorch's legacy `FindCUDA` does not overwrite them.

## Reference Implementation

`baseline/` (not tracked in this repo) contains the full working implementation (~4,700 lines). Treat it as a specification, not code to be modified. Key files:
- `dist_md.py` — distributed MD loop, Hydra config, NCCL init, Langevin thermostat
- `distributed_model.py` — `DistributedESCNMDBackbone`, equivariant message passing
- `halo_exchange.py` — `HaloExchanger`, GPU-to-GPU boundary atom communication
- `partition.py` / `sfc.py` — atom partitioning (SFC Morton/Hilbert, METIS)
- `local_fairchem/` — embedded FAIR Chemistry (UMA/ESCN, Wigner/Triton kernels)

Partition balance metric to optimize: **edge imbalance** (not atom count). METIS edge-weighted gives best results (see `docs/partition_benchmark.md`, not tracked in this repo).

## Performance Notes
- Baseline: ~7.2 steps/s on 4× A100 GPUs, ~4,000 atoms
- CUDA graphs, `torch.compile`, Triton kernels used for kernel fusion in baseline
- Halo exchange volume dominates at scale; partition quality directly impacts communication cost
