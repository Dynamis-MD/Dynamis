# dynamis_state

`dynamis_state` defines the state representation for the DistNNP framework. It provides C++ structs exposed to Python via pybind11 that describe the molecular system (`MolecularInfo`), the distributed compute environment (`DistEnvironment`), and the per-partition message-passing data (`PartitionMessagePassingInfo`). The top-level `DistNNPState` aggregates all three.

This sub-package is a dependency of every other DistNNP component: `distnnp_forces` reads state to compute forces, and `dynamis_integrator` writes state to evolve the system over time.

---

## Building

### Prerequisites

- Python >= 3.8 with PyTorch >= 2.0.0 installed
- CUDA Toolkit (12.x recommended)
- CMake >= 3.25
- A C++17/CUDA 17-capable compiler
- `scikit-build-core >= 0.5` (installed automatically by pip)

### CUDA location

The build system checks `CUDA_HOME` first (common on HPC/SLURM clusters), then `CUDAToolkit_ROOT`. Set whichever is appropriate before calling pip:

```bash
export CUDA_HOME=/path/to/cuda   # e.g. /usr/local/cuda or $CONDA_PREFIX/targets/x86_64-linux
```

On conda-managed environments neither variable typically needs to be set — the build system detects the conda CUDA layout automatically.

### Development (editable) install

```bash
pip install -e .
```

scikit-build-core runs CMake in `inplace` editable mode, compiling the extension and placing the `.so` directly in `src/dynamis_state/` alongside `__init__.py`. CMake build artefacts are written to `skbuild/`. After this, changes to Python files in `src/dynamis_state/` are picked up immediately. Changes to C++ files require re-running `pip install -e .`.

### Wheel build

```bash
pip install build
python -m build --wheel
```

The resulting wheel is written to `dist/`.

---

## Contributing: Adding a New C++ Function

The C++ → Python boundary is defined in two places:

| File | Role |
|---|---|
| `include/dynamis_state/*.hpp` | C++ declarations (structs, free functions) |
| `src/dynamis_state_bindings.cpp` | pybind11 bindings (what Python sees) |
| `src/dynamis_state/__init__.py` | Python-level re-exports |
| `src/dynamis_state/dynamis_state_impl.pyi` | Type stubs (auto-generated) |

### Step 1 — Declare in the header

Add your function or struct to the appropriate header in `include/dynamis_state/`. For example, to add a free function to `dynamis_state.hpp`:

```cpp
namespace dynamis_state {
    torch::Tensor compute_distances(const MolecularInfo& info);
}
```

### Step 2 — Implement in C++

For a free function, implement it in a new `.cpp` file under `src/` (e.g. `src/distances.cpp`) and add it to the `python_add_library` source list in `CMakeLists.txt`:

```cmake
python_add_library(dynamis_state_impl
    MODULE WITH_SOABI
    src/dynamis_state_bindings.cpp
    src/distances.cpp          # <-- new file
)
```

### Step 3 — Bind in `dynamis_state_bindings.cpp`

Inside the `PYBIND11_MODULE` block, register the function:

```cpp
m.def("compute_distances", &dynamis_state::compute_distances,
      py::arg("info"),
      "Returns a [N, N] tensor of pairwise distances for the atoms in info.");
```

For a new struct, use `py::class_<>` following the pattern of the existing bindings.

### Step 4 — Rebuild

```bash
pip install -e .
```

### Step 5 — Regenerate type stubs

```bash
./scripts/stub_gen.sh
```

This runs `pybind11-stubgen` against the installed module and writes updated `.pyi` stubs to `src/dynamis_state/dynamis_state_impl.pyi`. Commit the updated stub alongside your C++ and binding changes.

### Step 6 — Re-export from `__init__.py` (optional)

If the new symbol should be importable directly as `dynamis_state.compute_distances` (rather than `dynamis_state.dynamis_state_impl.compute_distances`), add it to `src/dynamis_state/__init__.py`:

```python
from dynamis_state.dynamis_state_impl import compute_distances
__all__ = [..., "compute_distances"]
```

