from distnnp_model.potential import DistNNPModel
import torch
import torch.nn.functional as F
from distnnp_state import DistNNPState
from distnnp_model.comm.feature_push import FeaturePushToExchange


def lennard_jones_potential(atom_types, positions, epsilon, sigma):
    """
    Calculates the total Lennard-Jones potential energy for a set of particles
    using Lorentz-Berthelot mixing rules for different atom types.

    Args:
        atom_types (torch.Tensor): An (N,) integer tensor of atom types.
        positions (torch.Tensor): An (N, 3) tensor of particle coordinates.
        epsilon (torch.Tensor): A (num_types,) 1D tensor of well depths for each type.
        sigma (torch.Tensor): A (num_types,) 1D tensor of zero-potential distances for each type.

    Returns:
        torch.Tensor: A scalar tensor representing the total potential energy.
    """
    N = positions.shape[0]

    # 1. Compute pairwise distances for all unique pairs
    # pdist returns a 1D tensor of length N*(N-1)/2
    distances = F.pdist(positions)

    # 2. Generate the index pairs matching the pdist output order
    # Ensure the indices are created on the same device as the positions (e.g., GPU)
    pair_indices = torch.combinations(torch.arange(N, device=positions.device))
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]

    # 3. Retrieve atom types for each pair
    type_i = atom_types[idx_i]
    type_j = atom_types[idx_j]

    # 4. Apply Lorentz-Berthelot mixing rules
    sigma_ij = (sigma[type_i] + sigma[type_j]) / 2.0
    epsilon_ij = torch.sqrt(epsilon[type_i] * epsilon[type_j])

    # 5. Compute the (sigma / r)^6 term
    sr6 = (sigma_ij / distances) ** 6

    # 6. Compute the full LJ potential for all pairs
    pairwise_potentials = 4 * epsilon_ij * (sr6**2 - sr6)

    # 7. Sum the potentials of all unique pairs
    total_energy = pairwise_potentials.sum()

    return total_energy


class LennardJonesPotentialModel(DistNNPModel):
    def __init__(self, epsilon: torch.Tensor, sigma: torch.Tensor):
        """Initializes the Lennard-Jones potential model with given epsilon and sigma parameters.

        Args:
            epsilon (torch.Tensor): A (num_types,) 1D tensor of well depths for each type.
            sigma (torch.Tensor): A (num_types,) 1D tensor of zero-potential distances for each type.
        """
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        prev_local_positions = (
            distributed_state.local_partition_info.local_atom_positions
        )
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_partition_info.partition_indices

        boundary_positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self._feature_exchanger
        )

        boundary_atom_types = FeaturePushToExchange.apply(
            local_atom_types.float(), distributed_state, self._feature_exchanger
        ).long()  # type: ignore

        local_positions = torch.cat([local_positions, boundary_positions], dim=0)  # type: ignore
        local_atom_types = torch.cat([local_atom_types, boundary_atom_types], dim=0)  # type: ignore

        energy = lennard_jones_potential(
            local_atom_types, local_positions, self.epsilon, self.sigma
        )

        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()
