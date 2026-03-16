import torch
from baseline.local_fairchem.models.uma.escn_moe import eSCNMDMoeBackbone
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, radius_graph_pbc_cuda
from distnnp_model.comm.feature_push import FeaturePushToExchange

class UMAModel(eSCNMDMoeBackbone, DistNNPModel):
    def __init__(
        self, 
        num_experts: int = 8,
        moe_dropout: float = 0.0,
        use_global_embedding: bool = False,  # obsolete
        use_composition_embedding: bool = False,
        moe_expert_coefficient_norm: str = "softmax",
        act=torch.nn.SiLU,
        layers_moe=None,
        moe_layer_type: str = "pytorch",
        moe_single: bool = False,
        moe_type: str = "so2",
        model_version: float = 1.0,
        **kwargs
    ):
        eSCNMDMoeBackbone.__init__(
            self,
            num_experts,
            moe_dropout,
            use_global_embedding,
            use_composition_embedding,
            moe_expert_coefficient_norm,
            act,
            layers_moe,
            moe_layer_type,
            moe_single,
            moe_type,
            model_version,
            **kwargs
        )
        DistNNPModel.__init__(self)

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        # 1. Setup initial local tensors
        prev_local_positions = distributed_state.local_partition_info.local_atom_positions
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_partition_info.partition_indices

        n_local_atoms = local_positions.shape[0]

        # 2. Push to get boundary positions and types via Autograd Function
        boundary_positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self._feature_exchanger
        )
        
        boundary_atom_types = FeaturePushToExchange.apply(
            local_atom_types.float(), distributed_state, self._feature_exchanger
        ).long()

        # 3. Concatenate local and boundary to form the full graph inputs
        positions = torch.cat([local_positions, boundary_positions], dim=0)
        n_atoms = positions.shape[0]
        atom_types = torch.cat([local_atom_types, boundary_atom_types], dim=0)

        # 4. Construct the radius graph
        sim_config = distributed_state.simulation_config
        with torch.no_grad():
            edge_index, _, _ = radius_graph_pbc_cuda(
                data=type(
                    "Data",
                    (object,),
                    {
                        "pos": positions,
                        "cell": sim_config.cell,
                        "natoms": n_atoms,
                        "pbc": sim_config.pbc,
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        # 5. Initial Embedding layer
        x = self.embedding(atom_types)

        # 6. Message Passing Layers with Intermediate IPC Exchange
        for i, layer in enumerate(self.interactions):
            # Compute interactions over the entire graph
            x = layer(x, positions, edge_index)

            # Synchronize intermediate features via Direct-Push IPC
            if i < len(self.interactions) - 1:
                # Fetch updated boundary features from neighbors
                boundary_features = FeaturePushToExchange.apply(
                    x, distributed_state, self._feature_exchanger
                )
                
                # Re-concatenate the synchronized features for the next layer
                x = torch.cat([x, boundary_features], dim=0)

        # Compute total local energy using only the local atoms
        energy = self.output_head(x[:n_local_atoms])
        
        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        return energy.detach(), forces.detach()
