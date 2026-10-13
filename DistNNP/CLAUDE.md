# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DistNNP** is a distributed neural network potential (NNP) framework for large-scale molecular dynamics (MD) simulations on multi-GPU systems. It computes forces and energies via ML potentials, partitions atoms across GPUs, communicates boundary atoms via halo exchange over NCCL, and integrates equations of motion using CUDA kernels.

The monorepo under `DistNNP/` contains four installable sub-packages being built from a working reference implementation in `../baseline/`.

## Development Setup

From the `DistNNP/` directory:

```bash
pip install -r requirements-dev.txt
```

This installs PyTorch, pytest, and all sub-packages in editable mode. Each sub-package must be individually reinstalled after C++ or CUDA changes:

```bash
pip install -e ./dynamis_state
pip install -e ./distnnp_integrator
pip install -e ./distnnp_model
pip install -e ./distnnp
```

### CUDA location

Set `CUDA_HOME` (HPC/SLURM) or `CUDAToolkit_ROOT` if CUDA is not found automatically. On conda environments neither is usually needed.

## Running Tests

```bash
pytest                              # all tests
pytest distnnp_integrator/tests/    # single sub-package
pytest distnnp_integrator/tests/test_euler.py -v  # single file
```

Tests that require CUDA are skipped automatically when no GPU is present.

## Repository Structure

```
DistNNP/                        # working directory (this repo)
├── dynamis_state/              # C++/CUDA: atom state, partitioning, halo maps
├── distnnp_integrator/         # C++/CUDA: MD time integration kernels
├── distnnp_model/              # Python: distributed NNP model interface
├── distnnp/                    # Python: main user-facing entry point (stub)
├── requirements-dev.txt        # installs all sub-packages in editable mode
../baseline/                    # reference production implementation (read-only)
../refs/                        # git submodules: fairchem, nvalchemi, DGraph, batmobile
../docs/                        # design docs and benchmarks
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

### `distnnp_integrator`
C++/CUDA extension. Advances positions and velocities by one timestep.

Interface (`include/distnnp_integrator/integrator.hpp`):
```
AbstractDistNNPIntegrator.step(positions, velocities, forces, masses, dt)
```
Currently implemented: `EulerIntegrator` (CUDA kernel in `src/csrc/euler_integrator.cu`).

To add a new integrator: subclass `AbstractDistNNPIntegrator` in the header → implement `.cu` kernel → add source to `python_add_library` in `CMakeLists.txt` → bind in `src/distnnp_integrator_bindings.cpp` → re-export from `__init__.py`. See `distnnp_integrator/README.md` for the full walkthrough.

Tests live in `distnnp_integrator/tests/test_euler.py`.

### `distnnp_model`
Pure Python (C++ extension stub present but empty). Defines the model interface.

`DistNNPModel` ABC (`src/distnnp_model/potential.py`):
```python
__call__(distributed_state: DistNNPState) -> Tuple[energy, forces]
```

Design: each GPU (1) fetches halo atoms from neighbours according to `DistNNPState`, (2) evaluates the NNP locally to get a scalar energy, (3) calls `energy.backward()` to obtain forces via autograd (`F = -dE/dx`). For GNN-based NNPs, halo atoms must be re-fetched once per message-passing layer since updated embeddings on one partition are inputs to the next layer on a neighbouring partition.

### `distnnp` (stub)
Empty entry point. Will orchestrate the other sub-packages.

## Build System

All C++ sub-packages use the same pattern:
- **Backend**: `scikit-build-core` (PEP 517), build artefacts in `skbuild/`
- **Extension**: `python_add_library(... MODULE WITH_SOABI ...)` from CMake `FindPython`
- **Editable mode**: `editable.mode = "inplace"` — `.so` is placed in `src/<pkg>/` alongside `__init__.py`
- **Torch linking**: `${TORCH_LIBRARIES}` + explicit `${TORCH_PYTHON_LIBRARY}` (found via `find_library`). Do **not** link standalone `pybind11::module` — torch bundles its own pybind11 and mixing them causes undefined `type_caster<at::Tensor>` symbols at import time.
- **CUDA discovery**: `find_package(CUDAToolkit)` runs before `find_package(Torch)` and caches `CUDA_TOOLKIT_ROOT_DIR` / `CUDA_INCLUDE_DIRS` with `FORCE` so PyTorch's legacy `FindCUDA` does not overwrite them.

## Reference Implementation

`../baseline/` contains the full working implementation (~4,700 lines). Treat it as a specification, not code to be modified. Key files:
- `dist_md.py` — distributed MD loop, Hydra config, NCCL init, Langevin thermostat
- `distributed_model.py` — `DistributedESCNMDBackbone`, equivariant message passing
- `halo_exchange.py` — `HaloExchanger`, GPU-to-GPU boundary atom communication
- `partition.py` / `sfc.py` — atom partitioning (SFC Morton/Hilbert, METIS)
- `local_fairchem/` — embedded FAIR Chemistry (UMA/ESCN, Wigner/Triton kernels)

Partition balance metric to optimize: **edge imbalance** (not atom count). METIS edge-weighted gives best results (see `../docs/partition_benchmark.md`).

## Performance Notes
- Baseline: ~7.2 steps/s on 4× A100 GPUs, ~4,000 atoms
- CUDA graphs, `torch.compile`, Triton kernels used for kernel fusion in baseline
- Halo exchange volume dominates at scale; partition quality directly impacts communication cost
