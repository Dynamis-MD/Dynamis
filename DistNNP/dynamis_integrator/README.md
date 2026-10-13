# dynamis_integrator

`dynamis_integrator` provides time integration algorithms for the DistNNP molecular dynamics framework. Each integrator advances atomic positions and velocities by one timestep given the forces computed by `distnnp_forces`. Integrators are written as CUDA kernels for GPU execution and exposed to Python via pybind11.

Currently implemented: **`EulerIntegrator`** — a first-order explicit integrator that updates velocities and positions according to:

```
v(t + dt) = v(t) + (F(t) / m) * dt
x(t + dt) = x(t) + v(t + dt) * dt
```

**`VerletIntegrator`** — a first-order explicit integrator that updates velocities over a half timestep, the positions over a full timestep, and then the velocities over a second half timestep according to:

```
v(t + dt/2) = v(t) + (F(t) / 2m) * dt
x(t + dt) = x(t) + v(t + dt/2) * dt

v(t + dt) = v(t + dt/2) + (F(t + dt) / 2m) * dt
```

**`LangevinIntegrator`** — a first-order explicit integrator that updates velocities over a half timestep, the positions over a full timestep, and then the velocities over a second half timestep, while also incorporating friction and random noise to simulate a controlled heat bath according to the BBK formulation of Langevin mechanics.

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

From the `dynamis_integrator/` directory:

```bash
pip install -e .
```

The C++ extension is compiled and the `.so` placed in `build/`. CMake build artefacts go to `build/` as well. Re-run `pip install -e .` after any C++ or CUDA change.

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
| `include/dynamis_integrator/integrator.hpp` | `AbstractDistNNPIntegrator` ABC and all concrete integrator declarations |
| `include/dynamis_integrator/macros.hpp` | CUDA error-checking macros (`CUDACHECK`, `CHECK_INPUT`, …) |
| `src/csrc/<name>.cu` | CUDA kernel implementation for each integrator |
| `src/dynamis_integrator_bindings.cpp` | pybind11 bindings (`PYBIND11_MODULE`) |
| `src/dynamis_integrator/__init__.py` | Python re-exports |

---

## Contributing: Adding a New Integrator

All integrators inherit from `AbstractDistNNPIntegrator` and override a single `step()` method. The steps below walk through adding a new one end-to-end.

### Step 1 — Declare in `integrator.hpp`

Subclass `AbstractDistNNPIntegrator` and declare `step()`:

```cpp
// include/dynamis_integrator/integrator.hpp
namespace dynamis_integrator {

  class VelocityVerletIntegrator : public AbstractDistNNPIntegrator
  {
  public:
    void step(torch::Tensor &positions,
              torch::Tensor &velocities,
              const torch::Tensor &forces,
              const torch::Tensor &masses,
              const float dt) override;
  };

} // namespace dynamis_integrator
```

The `step()` signature is fixed by the ABC and must not change.

### Step 2 — Implement the CUDA kernel in `src/csrc/`

Create `src/csrc/velocity_verlet_integrator.cu`. Use `macros.hpp` for input validation and `at::cuda::getCurrentCUDAStream` to submit work on the correct stream. Follow the pattern in `euler_integrator.cu`:

```cpp
#include "dynamis_integrator/integrator.hpp"
#include "dynamis_integrator/macros.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {
  __global__ void velocity_verlet_kernel(...) { /* kernel body */ }
}

namespace dynamis_integrator {
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
python_add_library(dynamis_integrator_impl
    MODULE WITH_SOABI
    src/dynamis_integrator_bindings.cpp
    src/csrc/euler_integrator.cu
    src/csrc/velocity_verlet_integrator.cu   # <-- new file
)
```

### Step 4 — Bind in `dynamis_integrator_bindings.cpp`

Register the new class inside the existing `PYBIND11_MODULE` block:

```cpp
py::class_<dynamis_integrator::VelocityVerletIntegrator>(m, "VelocityVerletIntegrator")
    .def(py::init<>())
    .def("step", &dynamis_integrator::VelocityVerletIntegrator::step);
```

### Step 5 — Rebuild

```bash
pip install -e .
```

### Step 6 — Re-export from `__init__.py`

Make the new class importable directly from `dynamis_integrator`:

```python
from dynamis_integrator.dynamis_integrator_impl import VelocityVerletIntegrator
__all__ = [..., "VelocityVerletIntegrator"]
```

### Step 7 — Add tests

Add a test file under `tests/` that at minimum constructs the integrator and calls `step()` with small GPU tensors to verify it runs without error.
