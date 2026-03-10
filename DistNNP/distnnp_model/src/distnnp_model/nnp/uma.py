import torch
from baseline.local_fairchem.models.uma.escn_moe import eSCNMDMoeBackbone
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, radius_graph_pbc_cuda
from distnnp_model.comm.feature_push import FeaturePushToExchange

class UMAModel(eSCNMDMoeBackbone, DistNNPModel):
    def __init__(self, **kwargs):
        # Initialize the underlying eSCN MOE model
        eSCNMDMoeBackbone.__init__(self, **kwargs)
        DistNNPModel.__init__(self)
        
        # Save parent kwargs to support the merge_experts logic
        self.parent_kwargs = kwargs

    def forward(
        self, 
        local_positions: torch.Tensor, 
        local_atom_types: torch.Tensor, 
        boundary_positions: torch.Tensor, 
        boundary_atom_types: torch.Tensor, 
        distributed_state: DistNNPState
    ):
        sim_config = distributed_state.simulation_config
        n_local_atoms = local_positions.shape[0]

        # 1. Combine local and boundary inputs
        # local_positions must require gradients to compute forces later.
        positions = torch.cat([local_positions, boundary_positions], dim=0)
        atom_types = torch.cat([local_atom_types, boundary_atom_types], dim=0)

        # 2. Construct the radius graph (Autograd must not track graph construction)
        with torch.no_grad():
            edge_index, _, _ = radius_graph_pbc_cuda(
                data=type("Data", (object,), {
                    "pos": positions,
                    "cell": sim_config.cell,
                    "natoms": positions.shape[0],
                    "pbc": sim_config.pbc,
                })(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        # 3. Initial Embedding layer
        x = self.embedding(atom_types)

        # 4. Message Passing Layers with Direct-Push IPC
        # Initialize the push operator for this forward pass
        feature_exchange = FeaturePushToExchange(distributed_state)

        for i, layer in enumerate(self.interactions):
            # Compute interactions (convolution updates both local and boundary, 
            # but boundary updates are mathematically "incomplete" here)
            x = layer(x, positions, edge_index)

            # Synchronize intermediate features using the Push API
            # We skip pushing after the final interaction if the output head only uses local atoms
            if i < len(self.interactions) - 1:
                # We push the valid local features. 
                # The wrapper should handle the C++ feature_push_forward, wait_for_pushes, 
                # and return the fully concatenated [local_features, updated_boundary_features] tensor.
                x = feature_exchange(x[:n_local_atoms])

        # 5. Energy Computation
        # The output head aggregates only the local atoms to compute total local energy
        energy = self.output_head(x[:n_local_atoms])
        
        # 6. Force computation via Autograd
        energy.backward()
        
        assert local_positions.grad is not None, "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()

    def merge_experts(self):
        """
        Preserved from the original implementation to support MoE merging.
        """
        from baseline.local_fairchem.models.uma.nn.mole_utils import (
            model_search_and_replace,
            recursive_replace_so2_MOLE,
            replace_MOLE_with_linear,
        )

        if getattr(self, "mole_type", "so2") != "so2":
            raise ValueError("Only mole_type=so2 supported for merging")

        model_search_and_replace(
            self, recursive_replace_so2_MOLE, replace_MOLE_with_linear
        )

        self.routing_mlp = None
        self.composition_embedding = None
        self.num_experts = 0

        new_kwargs = self.parent_kwargs.copy()
        new_kwargs["num_experts"] = 0
        new_kwargs["use_composition_embedding"] = False 
        
        new_model = DistributedESCNMDBackbone(**new_kwargs)
        new_model.load_state_dict(self.state_dict())
        
        return new_model
