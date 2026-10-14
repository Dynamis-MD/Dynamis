# DistNNP

`DistNNP` is a distributed neural network potential framework designed to efficiently compute forces and energies for large-scale molecular simulations. It is built on top of PyTorch and provides a modular architecture for defining state representations, force computations, and integration algorithms on multi-GPU systems. The framework uses custom CUDA kernels to optimize performance and scalability, making it suitable for simulating complex systems with millions of atoms.



## Installation

To install with editable mode and install all dependencies:

```bash
pip install -r requirements-dev.txt
```


## Sub-packages

The dynamis project is organized into several sub-packages, each responsible for a specific aspect of the overall functionality. Below is a brief overview of each sub-package:

- `dynamis_state`: This sub-package contains modules and classes related to the state representation of the system. It defines how the state of the system is represented, including positions, velocities, and other relevant properties related to distribution.
- `dynamis_model`: This sub-package focuses on the computation of forces within the system. It includes modules that calculate forces based on the state representation and other relevant parameters, specifically tailored for distribution-based neural network potentials.
- `dynamis_integrator`: This sub-package is responsible for the integration of the equations of motion. It includes modules that implement various integration algorithms to evolve the state of the system over time, taking into account the forces computed by the `dynamis_model` sub-package.
- `dynamis`: This is the main sub-package that serves as the entry point for users.

## Samples
The `samples` directory contains example scripts that demonstrate how to use the DistNNP framework for a large-scale molecular simulation.
