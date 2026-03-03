# distnnp_model

`distnnp_model` wraps neural network potentials (NNPs) with the distributed primitives needed to evaluate them across multiple GPUs using the state described by `distnnp_state`.

---

## Design

Evaluating an NNP in a distributed setting requires three steps per timestep:

```
1. Gather boundary atoms
   Each GPU uses the halo map in DistNNPState to fetch the positions of
   atoms that belong to neighbouring partitions but fall within the cutoff
   radius of its own atoms. After this step every GPU has a complete local
   neighbourhood graph.

   Note: for GNN-based NNPs, boundary vertices must be fetched once per
   message-passing layer, not just once per timestep. After each layer,
   updated atom embeddings on one partition may be needed by a neighbouring
   partition as inputs to the next layer. The number of halo exchanges per
   timestep therefore scales with the number of GNN layers.

2. Compute energy locally
   The NNP is evaluated on the local graph. Each GPU produces a scalar
   local energy contribution. No inter-GPU communication is required at
   this stage.

3. Compute forces via autograd
   Positions are tracked with requires_grad=True. Calling energy.backward()
   lets PyTorch propagate gradients back through the model, yielding forces
   as F = -dE/dx without any hand-written force expressions.
```

The `DistNNPModel` abstract base class (`src/distnnp_model/potential.py`) formalises this contract. Every concrete model must implement `__call__`, which receives a `DistNNPState` and returns `(energy, forces)`:

```python
class DistNNPModel(ABC):
    @abstractmethod
    def __call__(
        self, distributed_state: DistNNPState, *args, **kwargs
    ) -> Tuple[Union[float, torch.Tensor], torch.Tensor]:
        ...
```

The C++ extension (`distnnp_model_impl`) will house the low-level distributed communication primitives for step 1. The energy evaluation (step 2) and autograd (step 3) are pure Python / PyTorch.

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- `distnnp_state` installed (see `../distnnp_state/README.md`)
- CMake >= 3.25 and a CUDA 12.x toolkit (for the C++ extension)
- `scikit-build-core >= 0.5` (installed automatically by pip)

### CUDA location

Set `CUDA_HOME` (HPC/SLURM) or `CUDAToolkit_ROOT` if CUDA is not found automatically:

```bash
export CUDA_HOME=/path/to/cuda
```

### Development (editable) install

From the `distnnp_model/` directory:

```bash
pip install -e .
```

CMake build artefacts go to `skbuild/`. Re-run `pip install -e .` after any C++ change.

### Wheel build

```bash
pip install build
python -m build --wheel
```

---

## Package layout

| Path | Role |
|---|---|
| `src/distnnp_model/potential.py` | `DistNNPModel` abstract base class |
| `src/distnnp_model_bindings.cpp` | pybind11 bindings for C++ distributed primitives |
| `src/distnnp_model/__init__.py` | Python re-exports |

---

## Implementing a New Model

Subclass `DistNNPModel` and implement `__call__`. The implementation is responsible for:

1. Reading the partition and halo information from `distributed_state`
2. Building the local neighbourhood graph
3. Running the NNP forward pass with `positions.requires_grad_(True)`
4. Calling `energy.backward()` and reading `positions.grad` as forces

```python
import torch
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState


class MyNNP(DistNNPModel):
    def __init__(self, ...):
        # load model weights, set cutoff, etc.
        ...

    def __call__(self, distributed_state: DistNNPState, **kwargs):
        positions = distributed_state.local_partition_info.atom_positions
        positions = positions.detach().requires_grad_(True)

        # --- step 1: gather halo atoms (distributed primitive, TBD) ---
        # full_positions = gather_halo(positions, distributed_state)

        # --- step 2: compute local energy ---
        energy = self._forward(positions, ...)

        # --- step 3: forces via autograd ---
        energy.backward()
        forces = -positions.grad

        return energy, forces

    def _forward(self, positions, ...):
        # NNP graph construction and message passing
        ...
```

The model does not need to concern itself with how atoms were partitioned — that is handled by `distnnp_state`. It only sees its own local atoms plus whatever halo atoms the distributed primitives deliver.
