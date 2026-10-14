# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dynamis** is a distributed neural network potential (NNP) framework for large-scale molecular dynamics (MD) simulations on multi-GPU systems. It computes forces and energies via ML potentials, partitions atoms across GPUs, communicates boundary atoms via halo exchange over NCCL, and integrates equations of motion using CUDA kernels.

This repository is a monorepo of four installable sub-packages, built from a working reference implementation (`baseline/`, not tracked in this repo).

## Development Setup

From the repository root:

```bash
pip install -r requirements-dev.txt
```

This installs PyTorch, pytest, and all sub-packages in editable mode. Each sub-package must be individually reinstalled after C++ or CUDA changes:

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
├── dynamis_integrator/         # C++/CUDA: MD time integration kernels
├── dynamis_model/              # Python + C++/CUDA: distributed NNP model interface, exchangers
├── dynamis/                    # Python: main user-facing entry point
├── benchmarks/                 # standalone kernel/communication benchmarks
├── experiments/                # paper experiments and dataset generation
├── post_processing/            # trajectory rendering, timing summaries
├── refs/                       # git submodules: nvalchemi, batmobile
├── requirements-dev.txt        # installs all sub-packages in editable mode
└── pytest.ini
```

## Sub-package Status and Architecture

### `dynamis_state`
C++/CUDA extension. Defines the core data structures shared by all other packages.

Key types (in `include/dynamis_state/dynamis_state.hpp`):
- `MolecularInfo` — positions, velocities, atom indices for a partition
- `DistEnvironment` — rank, world size, GPU topology
- `PartitionMessagePassingInfo` — partition indices + halo map + send/recv matrix
- `DynamisState` — aggregates the above

Partitioner interface (`include/dynamis_state/partitioner.hpp`): `AbstractPartitioner` with concrete subclasses `SimpleSpatialPartitioner`, `KDTreePartitioner`, `SFCPartitioner`.

Python classes are re-exported at the top level: `from dynamis_state import MolecularInfo, DynamisState, ...`

Stubs are generated via `scripts/stub_gen.sh` (runs `pybind11-stubgen`).

### `dynamis_integrator`
C++/CUDA extension. Advances positions and velocities by one timestep.

Interface (`include/dynamis_integrator/integrator.hpp`):
```
AbstractDynamisIntegrator.step(positions, velocities, forces, masses, dt)
```
Currently implemented: `EulerIntegrator` (CUDA kernel in `src/csrc/euler_integrator.cu`).

To add a new integrator: subclass `AbstractDynamisIntegrator` in the header → implement `.cu` kernel → add source to `python_add_library` in `CMakeLists.txt` → bind in `src/dynamis_integrator_bindings.cpp` → re-export from `__init__.py`. See `dynamis_integrator/README.md` for the full walkthrough.

Tests live in `dynamis_integrator/tests/test_euler.py`.

### `dynamis_model`
Pure Python (C++ extension stub present but empty). Defines the model interface.

`DynamisModel` ABC (`src/dynamis_model/potential.py`):
```python
__call__(distributed_state: DynamisState) -> Tuple[energy, forces]
```

Design: each GPU (1) fetches halo atoms from neighbours according to `DynamisState`, (2) evaluates the NNP locally to get a scalar energy, (3) calls `energy.backward()` to obtain forces via autograd (`F = -dE/dx`). For GNN-based NNPs, halo atoms must be re-fetched once per message-passing layer since updated embeddings on one partition are inputs to the next layer on a neighbouring partition.

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
