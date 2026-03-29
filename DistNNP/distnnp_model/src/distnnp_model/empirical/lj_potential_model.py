from distnnp_model.potential import DistNNPModel
import torch
import torch.nn.functional as F
from distnnp_state import DistNNPState, radius_graph_pbc_cuda, SimState
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model import FeatureExchanger


def lennard_jones_potential(atom_types, positions, edge_index, epsilon, sigma):
    """
    Calculates the total Lennard-Jones potential energy for a set of particles
    using Lorentz-Berthelot mixing rules and a cutoff radius graph.

    Args:
        atom_types (torch.Tensor): An (N,) integer tensor of atom types.
        positions (torch.Tensor): An (N, 3) tensor of particle coordinates (requires_grad=True).
        edge_index (torch.Tensor): A (2, E) tensor of edge indices from the radius graph.
        epsilon (torch.Tensor): A (num_types,) 1D tensor of well depths for each type.
        sigma (torch.Tensor): A (num_types,) 1D tensor of zero-potential distances for each type.

    Returns:
        torch.Tensor: A scalar tensor representing the total potential energy.
    """

    # Extract source and target node indices
    idx_i, idx_j = edge_index[0], edge_index[1]

    mask = idx_i < idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    dr = positions[idx_i] - positions[idx_j]
    distances = torch.linalg.norm(dr, dim=1).clamp(min=1e-6)  # Avoid division by zero

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

    # 6. Sum the potentials of all unique pairs
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

    def __call__(self, distributed_state: SimState, *args, **kwargs):
        prev_local_positions = (
            distributed_state.local_partition_info.local_atom_positions
        )
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_partition_info.partition_indices

        if not self._feature_exchanger.is_ipc_setup:
            n_local = local_positions.shape[0]
            rank = distributed_state.dist_env.global_rank

            world_size = distributed_state.dist_env.world_size
            total_recv = (
                distributed_state.local_partition_info.global_send_recv_matrix.sum(
                    dim=0
                )[rank].item()
            )
            total_send = (
                distributed_state.local_partition_info.global_send_recv_matrix.sum(
                    dim=1
                )[rank].item()
            )
            F = local_positions.shape[1]
            x_buffer = FeatureExchanger.allocate_ipc_buffer(
                n_local + total_recv, F, rank
            )
            grad_recv_buffer = FeatureExchanger.allocate_ipc_buffer(total_send, F, rank)

            x_handle = self._feature_exchanger.export_ipc_handle(x_buffer)
            grad_handle = self._feature_exchanger.export_ipc_handle(grad_recv_buffer)

            handle_size = len(x_handle)
            x_handle_tensor = torch.tensor(
                x_handle, dtype=torch.uint8, device=local_positions.device
            )
            grad_handle_tensor = torch.tensor(
                grad_handle, dtype=torch.uint8, device=local_positions.device
            )
            all_x_handles_tensor = [
                torch.zeros(
                    handle_size, dtype=torch.uint8, device=local_positions.device
                )
                for _ in range(world_size)
            ]
            all_grad_handles_tensor = [
                torch.zeros(
                    handle_size, dtype=torch.uint8, device=local_positions.device
                )
                for _ in range(world_size)
            ]

            # self._feature_exchanger.setup_ipc()

        boundary_positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self._feature_exchanger
        )

        boundary_atom_types = FeaturePushToExchange.apply(
            local_atom_types.float(), distributed_state, self._feature_exchanger
        ).long()  # type: ignore

        positions = torch.cat([local_positions, boundary_positions], dim=0)  # type: ignore
        atom_types = torch.cat([local_atom_types, boundary_atom_types], dim=0)  # type: ignore

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
            atom_types, positions, edge_index, self.epsilon, self.sigma
        )

        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()
