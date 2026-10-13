from distnnp_model.potential import DistNNPModel
import torch
import torch.nn.functional as F
from dynamis_state import (
    radius_graph_pbc_cuda,
    DistNNPState,
    PartitionMessagePassingInfo,
)
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.comm.exchanger import Exchanger, GPUDirectExchanger
from typing import Optional


def lennard_jones_potential(atom_types, positions, edge_index, epsilon, sigma, n_local):
    """
    Calculates the total Lennard-Jones potential energy for a set of particles
    using Lorentz-Berthelot mixing rules and a cutoff radius graph.

    Args:
        atom_types (torch.Tensor): An (N,) integer tensor of atom types.
        positions (torch.Tensor): An (N, 3) tensor of particle coordinates (requires_grad=True).
        edge_index (torch.Tensor): A (2, E) tensor of edge indices from the radius graph.
        epsilon (torch.Tensor): A (num_types,) 1D tensor of well depths for each type.
        sigma (torch.Tensor): A (num_types,) 1D tensor of zero-potential distances for each type.
        n_local (int): Number of local (non-halo) atoms in the buffer.

    Returns:
        torch.Tensor: A scalar tensor representing the total potential energy.
    """

    # Extract source and target node indices
    idx_i, idx_j = edge_index[0], edge_index[1]

    # Count every edge whose SOURCE atom is local. radius_graph_pbc emits each pair
    # as two directed edges, so an intra-rank pair is counted twice on its rank, and
    # a cross-rank pair is counted once on each of the two ranks it straddles -- i.e.
    # every pair is double-counted uniformly. The /2 on total_energy below cancels it.
    mask = idx_i < n_local
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    dr = positions[idx_i] - positions[idx_j]
    distances = torch.linalg.norm(dr, dim=1).clamp(min=1e-6)  # Avoid division by zero

    # Exclude bonded pairs: anything closer than 2.5 Å is covalently bonded
    # and sits deep inside the LJ repulsive wall
    nonbonded = distances > 2.5
    idx_i = idx_i[nonbonded]
    idx_j = idx_j[nonbonded]
    distances = distances[nonbonded]

    # 2. Retrieve atom types for each pair
    type_i = atom_types[idx_i]
    type_j = atom_types[idx_j]

    # 3. Apply Lorentz-Berthelot mixing rules
    sigma_ij = (sigma[type_i] + sigma[type_j]) / 2.0
    epsilon_ij = torch.sqrt(epsilon[type_i] * epsilon[type_j])

    # 4. Compute the (sigma / r)^6 term
    sr6 = (sigma_ij / distances) ** 6

    # 5. Compute the full LJ potential for all pairs
    pairwise_potentials = 4 * epsilon_ij * (sr6**2 - sr6)

    # 6. Sum the potentials. Every pair is counted twice (see mask above), so halve.
    total_energy = pairwise_potentials.sum() / 2.0

    return total_energy


class LennardJonesPotentialModel(DistNNPModel):
    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        exchanger: Optional[Exchanger] = None,
    ):
        """Initializes the Lennard-Jones potential model with given epsilon and sigma parameters.

        Args:
            epsilon (torch.Tensor): A (num_types,) 1D tensor of well depths for each type.
            sigma (torch.Tensor): A (num_types,) 1D tensor of zero-potential distances for each type.
        """
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        # Always use independent exchangers for different dimensionalities (3D pos vs 1D types)
        # to prevent IPC buffer shape mismatches.
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()

    def initialize_buffers(
        self, partition_info: PartitionMessagePassingInfo, device
    ) -> None:
        # For this simple LJ model, we don't have any learnable parameters or buffers that depend on the partitioning
        # However, if we wanted to, we could initialize any necessary buffers here based on the partition_info
        self.position_exchanger.setup_exchanger(
            feature_dim=3, partition_info=partition_info, device=device
        )
        self.atom_type_exchanger.setup_exchanger(
            feature_dim=1, partition_info=partition_info, device=device
        )

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        prev_local_positions = distributed_state.local_molecular_info.atom_positions
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        positions: torch.Tensor = FeaturePushToExchange.apply(
            local_positions, distributed_state, self.position_exchanger
        )  # type: ignore

        local_atom_types = distributed_state.local_molecular_info.atom_types
        # The unsqueeze/reshape pair below assumes 1-D [N] type indices. If this
        # ever arrives as [N, 1], unsqueeze(1) makes it [N, 1, 1] and the pushed
        # feature dim silently stops being 1.
        assert local_atom_types.dim() == 1, (
            "local_atom_types must be 1-D [n_local], got shape "
            f"{tuple(local_atom_types.shape)}"
        )

        with torch.no_grad():
            # exchanger needs a 2D [N, 1] feature; the LJ energy needs 1D [N]
            # atom-type indices. Without the reshape(-1), atom_types stays [N, 1],
            # making sigma_ij [E, 1] and sigma_ij / distances broadcast to [E, E]
            # -> energy (and forces) inflated ~E x, which blows positions up on the
            # next integrator step and overflows the radius-graph grid size.
            atom_types = (
                FeaturePushToExchange.apply(
                    local_atom_types.float().unsqueeze(1), distributed_state, self.atom_type_exchanger
                )
                .long()
                .reshape(-1)
            )  # type: ignore

        # Construct the radius graph with PBC using the local positions (including boundary atoms)

        sim_config = distributed_state.sim_config
        # Ensure autograd is not tracking the radius graph construction
        n_atoms = positions.shape[0]
        with torch.no_grad():
            edge_index, _, _ = radius_graph_pbc_cuda(
                data=type(
                    "Data",
                    (object,),
                    {
                        "pos": positions,
                        "cell": sim_config.cell.unsqueeze(0),  # [1, 3, 3]
                        "natoms": [n_atoms],
                        "pbc": sim_config.pbc,
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        # Compute the Lennard-Jones potential energy using the local atom types, positions, and edge index
        energy = lennard_jones_potential(
            atom_types, positions, edge_index, self.epsilon, self.sigma,
            n_local=local_positions.shape[0],
        )

        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()
