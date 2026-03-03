# distnnp_integrator

`distnnp_integrator` provides time integration algorithms for the DistNNP molecular dynamics framework. Each integrator advances atomic positions and velocities by one timestep given the forces computed by `distnnp_forces`. Integrators are written as CUDA kernels for GPU execution and exposed to Python via pybind11.

Currently implemented: **`EulerIntegrator`** — a first-order explicit integrator that updates velocities and positions according to:

```
v(t + dt) = v(t) + (F(t) / m) * dt
x(t + dt) = x(t) + v(t + dt) * dt
```

---

## Installation

### Prerequisites

- Python >= 3.8 with PyTorch >= 2.0.0 installed
- CUDA Toolkit (12.x recommended)
- CMake >= 3.25
- A C++17 / CUDA 17-capable compiler
- `scikit-build-core >= 0.5` (installed automatically by pip)

### CUDA location

Set `CUDA_HOME` (common on HPC/SLURM) or `CUDAToolkit_ROOT` before calling pip if the build system cannot find CUDA automatically:

```bash
export CUDA_HOME=/path/to/cuda   # e.g. /usr/local/cuda
```

On conda-managed environments neither variable typically needs to be set.

### Development (editable) install

From the `distnnp_integrator/` directory:

```bash
pip install -e .
```

The C++ extension is compiled and the `.so` placed in `src/distnnp_integrator/`. CMake build artefacts go to `build/`. Re-run `pip install -e .` after any C++ or CUDA change.

### Running tests

```bash
pip install -e ".[test]"
pytest tests/
```

### Wheel build

```bash
pip install build
python -m build --wheel
```

---

## Package layout

| Path | Role |
|---|---|
| `include/distnnp_integrator/integrator.hpp` | `AbstractDistNNPIntegrator` ABC and all concrete integrator declarations |
| `include/distnnp_integrator/macros.hpp` | CUDA error-checking macros (`CUDACHECK`, `CHECK_INPUT`, …) |
| `src/csrc/<name>.cu` | CUDA kernel implementation for each integrator |
| `src/distnnp_integrator_bindings.cpp` | pybind11 bindings (`PYBIND11_MODULE`) |
| `src/distnnp_integrator/__init__.py` | Python re-exports |

---

## Contributing: Adding a New Integrator

All integrators inherit from `AbstractDistNNPIntegrator` and override a single `step()` method. The steps below walk through adding a new one end-to-end.

### Step 1 — Declare in `integrator.hpp`

Subclass `AbstractDistNNPIntegrator` and declare `step()`:

```cpp
// include/distnnp_integrator/integrator.hpp
namespace distnnp_integrator {

  class VelocityVerletIntegrator : public AbstractDistNNPIntegrator
  {
  public:
    void step(torch::Tensor &positions,
              torch::Tensor &velocities,
              const torch::Tensor &forces,
              const torch::Tensor &masses,
              const float dt) override;
  };

} // namespace distnnp_integrator
```

The `step()` signature is fixed by the ABC and must not change.

### Step 2 — Implement the CUDA kernel in `src/csrc/`

Create `src/csrc/velocity_verlet_integrator.cu`. Use `macros.hpp` for input validation and `at::cuda::getCurrentCUDAStream` to submit work on the correct stream. Follow the pattern in `euler_integrator.cu`:

```cpp
#include "distnnp_integrator/integrator.hpp"
#include "distnnp_integrator/macros.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {
  __global__ void velocity_verlet_kernel(...) { /* kernel body */ }
}

namespace distnnp_integrator {
  void VelocityVerletIntegrator::step(...) {
    CHECK_INPUT(positions);
    // ... launch kernel on getCurrentCUDAStream ...
    CUDACHECK(cudaGetLastError());
  }
}
```

### Step 3 — Register the source file in `CMakeLists.txt`

Add the new `.cu` file to the `python_add_library` source list:

```cmake
python_add_library(distnnp_integrator_impl
    MODULE WITH_SOABI
    src/distnnp_integrator_bindings.cpp
    src/csrc/euler_integrator.cu
    src/csrc/velocity_verlet_integrator.cu   # <-- new file
)
```

### Step 4 — Bind in `distnnp_integrator_bindings.cpp`

Register the new class inside the existing `PYBIND11_MODULE` block:

```cpp
py::class_<distnnp_integrator::VelocityVerletIntegrator>(m, "VelocityVerletIntegrator")
    .def(py::init<>())
    .def("step", &distnnp_integrator::VelocityVerletIntegrator::step);
```

### Step 5 — Rebuild

```bash
pip install -e .
```

### Step 6 — Re-export from `__init__.py`

Make the new class importable directly from `distnnp_integrator`:

```python
from distnnp_integrator.distnnp_integrator_impl import VelocityVerletIntegrator
__all__ = [..., "VelocityVerletIntegrator"]
```

### Step 7 — Add tests

Add a test file under `tests/` that at minimum constructs the integrator and calls `step()` with small GPU tensors to verify it runs without error.
