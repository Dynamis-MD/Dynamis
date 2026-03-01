# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DistNNP** is a distributed neural network potential (NNP) framework for large-scale molecular dynamics (MD) simulations on multi-GPU systems. It computes forces and energies via ML potentials (e.g., UMA/ESCN from FAIR Chemistry), partitions atoms across GPUs using space-filling curves or METIS, and communicates boundary atoms via halo exchange over NCCL.

The current branch (`proj-setup`) is refactoring a working baseline (`../baseline/`) into a clean monorepo of installable sub-packages.

## Development Setup

From the `DistNNP/` directory (where `requirements-dev.txt` lives):

```bash
pip install -r requirements-dev.txt
```

This installs PyTorch, pytest, and all sub-packages in editable mode (`-e`). Sub-packages with C++ extensions (`distnnp_state`) require CMake and pybind11 and are built via scikit-build-core.

## Running Tests

```bash
pytest                          # run all tests
pytest distnnp_state/tests/     # run tests for a specific sub-package
```

## Repository Structure

```
DistNNP/                        # working directory (this repo)
├── distnnp/                    # main user-facing entry point
├── distnnp_state/              # atom state, partitioning, halo exchange
├── distnnp_forces/             # force/energy computation (planned)
├── distnnp_integrator/         # MD integration algorithms (planned)
├── requirements-dev.txt        # installs all sub-packages in editable mode
../baseline/                    # reference production implementation (read-only)
../refs/                        # git submodules: fairchem, nvalchemi, DGraph, batmobile
../docs/                        # design docs and benchmarks
```

Each sub-package follows the `src/` layout (`src/<pkg>/__init__.py`) and has its own `pyproject.toml`. Sub-packages with C++ extensions also have a `CMakeLists.txt` using pybind11.

## Architecture

### Sub-package Dependencies
```
distnnp
  └── distnnp_state
  └── distnnp_forces
        └── distnnp_state
  └── distnnp_integrator
        └── distnnp_state
        └── distnnp_forces
```

### Distributed MD Simulation Flow (from baseline)
```
Config (Hydra) → NCCL distributed init
  → Load ML model (UMA/ESCN) + atom structure (.xyz)
  → Partition atoms across GPUs (SFC Morton/Hilbert or METIS edge-weighted)
  → Each GPU: build local neighbor graph + identify halo atoms
  → Halo exchange: communicate boundary atom positions via NCCL
  → Forward pass: graph → forces/energies
  → Langevin/Velocity-Verlet integration step
  → Repeat
```

### Key Concepts
- **Partitioning** (`partition.py`, `sfc.py`): Distributes atoms across GPUs. Space-filling curves (Morton/Hilbert) preserve spatial locality; METIS edge-weighted balances graph edge count per partition. Benchmarks show METIS edge-weighted is best overall (see `../docs/partition_benchmark.md`).
- **Halo exchange** (`halo_exchange.py`): Atoms near partition boundaries are duplicated as "halo" atoms on neighboring GPUs so each GPU has the full cutoff neighborhood. `HaloExchanger` builds halo maps and performs GPU-to-GPU position communication.
- **Distributed model** (`distributed_model.py`): `DistributedESCNMDBackbone` uses E(n)-equivariant message passing (SO(3), Wigner coefficients) for force predictions. Custom Triton kernels accelerate Wigner transforms.
- **Force calculator** (`distributed_calculator.py`): Wraps the distributed model in an ASE-compatible `Calculator` interface.
- **Profiling** (`profiler.py`): `CUDAEventProfiler`/`StepProfiler` measure wall time and per-step GPU timing for benchmarking.

### Build System
Sub-packages with C++ (e.g., `distnnp_state`) use:
- `scikit-build-core` as the PEP 517 build backend
- `pybind11` for Python-C++ bindings
- CMake `Release` build type
- `.pyi` stubs + `py.typed` marker for type checking support

### Reference Implementation
`../baseline/` contains the full working implementation (~4,700 lines) that the sub-packages are being refactored from. Treat it as a specification and reference, not as code to be modified. `../baseline/local_fairchem/` is an embedded copy of FAIR Chemistry (UMA/ESCN models, SO(3) utilities, Triton-optimized Wigner kernels).

## Performance Notes
- Baseline achieves ~7.2 steps/s on 4× A100 GPUs for ~4,000 atoms
- CUDA graphs, `torch.compile`, and Triton kernels are used for kernel fusion
- Activation checkpointing is available in the backbone
- Partition balance metric to optimize: edge imbalance (not just atom count)
