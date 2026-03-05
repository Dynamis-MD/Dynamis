from distnnp_model.potential import DistNNPModel
import torch
import torch.nn.functional as F
from distnnp_state import DistNNPState


def lennard_jones_potential(positions, epsilon=1.0, sigma=1.0):
    """
    Calculates the total Lennard-Jones potential energy for a set of particles.

    Args:
        positions (torch.Tensor): An (N, 3) tensor of particle coordinates.
        epsilon (float): Depth of the potential well.
        sigma (float): Distance at which the potential is zero.

    Returns:
        torch.Tensor: A scalar tensor representing the total potential energy.
    """
    # 1. Compute pairwise distances for all unique pairs
    # pdist returns a 1D tensor of length N*(N-1)/2, completely avoiding
    # the diagonal (self-interactions) and duplicate pairs.
    distances = F.pdist(positions)

    # 2. Compute the (sigma / r)^6 term
    sr6 = (sigma / distances) ** 6

    # 3. Compute the full LJ potential for all pairs
    # (sigma / r)^12 is just sr6 squared, which is faster to compute
    pairwise_potentials = 4 * epsilon * (sr6**2 - sr6)

    # 4. Sum the potentials of all unique pairs to get the total system energy
    total_energy = pairwise_potentials.sum()

    return total_energy


class LennardJonesPotentialModel(DistNNPModel):
    def __init__(self, epsilon=1.0, sigma=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        prev_local_positions = (
            distributed_state.local_partition_info.local_atom_positions
        )
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_partition_info.local_atom_types
        # TODO: Not implemented yet
        # boundary_positions = ...
        # local_positions = torch.cat([local_positions, boundary_positions], dim=0)

        energy = lennard_jones_potential(local_positions, self.epsilon, self.sigma)

        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()
