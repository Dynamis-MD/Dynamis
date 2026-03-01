## DistNNP State Representation

The `distnnp_state` sub-package defines the state representation for the DistNNP framework. It includes classes and functions that read, write, and represent the positions, velocities, and other relevant properties of the system. The state representaiton also crucially handles the distribution of the system across multiple GPUs, including partitioning the system and managing the communication between different parts of the system. The distnnp_state is used by all other sub-packes. `distnnp_forces` uses the state representation to compute forces, while `distnnp_integrator` uses it to evolve the state over time. 

