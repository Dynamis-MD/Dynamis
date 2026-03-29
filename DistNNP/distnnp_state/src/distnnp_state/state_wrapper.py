from dataclasses import dataclass
from distnnp_state import DistNNPState
from distnnp_state.masses import TORCH_ATOMIC_MASSES
import torch


class SimState:
    def __init__(self, dist_nnp_state: DistNNPState):
        self._state = dist_nnp_state
        pi = dist_nnp_state.local_partition_info
        self.atomic_numbers = pi.partition_indices.reshape(-1)
        self.mass = TORCH_ATOMIC_MASSES[self.atomic_numbers].to(
            device=pi.local_atom_positions.device
        )

        # Zero-iniitlaize forces for now
        self.forces = torch.zeros_like(pi.local_atom_positions)

    @property
    def pos(self):
        return self._state.local_partition_info.local_atom_positions

    @property
    def vel(self):
        return self._state.local_partition_info.local_atom_velocities

    # Pass on all other attribute accesses to the underlying DistNNPState
    def __getattr__(self, name):
        return getattr(self._state, name)
