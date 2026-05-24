from distnnp_model.potential import DistNNPModel
from distnnp_model.comm.exchanger import Exchanger, GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_state import (
    DistNNPState,
    PartitionMessagePassingInfo,
    radius_graph_pbc_cuda,
)

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces_and_stress,
    compute_forces,
    compute_hessian,
)
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings
import torch
import torch.distributed as dist
from typing import Literal, Optional, List

class UMAModel(DistNNPModel):
    def __init__(
        self, 
        model_name: str,
        # I don't think we can use turbo inference settings because that compiles the graphs, which change when we repartition?
        inference_settings: InferenceSettings | str = "default", #"turbo",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        exchanger: Optional[Exchanger] = None,
    ):
        DistNNPModel.__init__(self)
        self.position_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )

        self.atom_type_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.feature_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.predict_unit = pretrained_mlip.get_predict_unit(model_name, inference_settings, overrides, device)

    def initialize_buffers(
        self, partition_info: PartitionMessagePassingInfo, device
    ) -> None:
        self.position_exchanger.setup_exchanger(
            feature_dim=3, partition_info=partition_info, device=device
        )
        self.atom_type_exchanger.setup_exchanger(
            feature_dim=1, partition_info=partition_info, device=device
        )

        backbone = self.predict_unit.model.module.backbone
        total_feature_dim = backbone.sph_feature_size * backbone.sphere_channels
        self.feature_exchanger.setup_exchanger(
            feature_dim=total_feature_dim, partition_info=partition_info, device=device
        )

    def __call__(
        self,
        distributed_state: DistNNPState,
        batch: torch.Tensor,
        batch_full: torch.Tensor,     # ?
        datasets: str | List[str], # ?
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        device: Literal["cuda", "cpu"] | None = None,
    ):
        if isinstance(datasets, str):
            datasets = [datasets]

        #ESCN FORWARD
        mol_info = distributed_state.local_molecular_info
        sim_config = distributed_state.sim_config
        part_info = distributed_state.local_partition_info

        prev_local_positions = mol_info.atom_positions
        local_positions = prev_local_positions.clone().requires_grad_(True)
        local_atom_types = mol_info.atom_types
        n_local_atoms = local_positions.shape[0]

        device = prev_local_positions.device if device is None else device
        self.predict_unit.model.to(device)
        main_model = self.predict_unit.model.module if hasattr(self.predict_unit.model, "module") else self.predict_unit.model
        backbone = main_model.backbone

        positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self.position_exchanger
        )

        atom_types = FeaturePushToExchange.apply(
            local_atom_types.float().unsqueeze(-1), distributed_state, self.atom_type_exchanger
        ).squeeze(-1).long()

        n_atoms = positions.shape[0]
        pbc = sim_config.pbc.unsqueeze(0)

        cell = sim_config.cell
        with torch.no_grad():
            edge_index, _, _ = radius_graph_pbc_cuda(
                data=type(
                    "Data",
                    (object,),
                    {
                        "pos": positions,
                        "cell": cell.unsqueeze(0),
                        "natoms": torch.tensor([n_atoms], dtype=torch.long, device=device),
                        "pbc": pbc,
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        src, tgt = edge_index[0], edge_index[1]
        edge_distance_vec = positions[tgt] - positions[src]
        edge_distance = torch.norm(edge_distance_vec, p=2, dim=-1)
        # gp_node_offset = 0

        num_systems = int(batch.max().item()) + 1 # Not everything supports batched systems, TODO
        if charge is None:
            charge = torch.zeros(num_systems, dtype=torch.long, device=device)
        if spin is None:
            spin = torch.zeros(num_systems, dtype=torch.long, device=device)

        n_edges = edge_index.shape[1]
        data = AtomicData(
            positions,  # (num_node, 3)
            atom_types,  # (num_node,)
            cell.unsqueeze(0),  # (num_graph, 3, 3)
            pbc,  # (num_graph, 3)
            torch.tensor([n_atoms], dtype=torch.long, device=device),  # (1,)
            edge_index,  # (2, num_edge)
            torch.zeros((edge_index.shape[1], 3), device=device), # cell_offsets, (num_edge, 3)
            torch.tensor([n_edges], dtype=torch.long, device=device),  # (1,)
            charge,  # (num_graph,)
            spin,  # (num_graph,)
            torch.zeros(n_atoms, dtype=torch.long, device=device), # fixed, (num_node,)
            torch.zeros(n_atoms, dtype=torch.long, device=device), # tags, (num_node,)
        )
        self.predict_unit._lazy_init(data)
        

        csd_mixed_emb = backbone.csd_embedding(
            charge=charge,
            spin=spin,
            dataset=datasets,
        )

        backbone.set_MOLE_coefficients(
            atomic_numbers_full=atom_types, # NOT float, using nn.Embedding
            batch_full=batch_full,
            csd_mixed_emb=csd_mixed_emb,
        )

        # Enable gradients for autograd-based force/stress computation.
        # Must be set before graph generation so the computation graph
        # SHOULD THESE BE USED?
        # tracks positions and cell through edge distance calculations.
        # if not backbone.regress_config.direct_forces:
        #     if backbone.regress_config.forces or backbone.regress_config.stress:
        #         positions.requires_grad_(True)
        #     if backbone.regress_config.stress:
        #         cell.requires_grad_(True)

        wigner, wigner_inv = backbone._get_rotmat_and_wigner(
            edge_distance_vecs = edge_distance_vec
        )
        coefficient_index = (
            backbone.coefficient_index if backbone.mmax != backbone.lmax else None
        )
        wigner, wigner_inv = backbone.backend.prepare_wigner(
            wigner,
            wigner_inv,
            backbone.mappingReduced,
            coefficient_index,
        )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x_message = torch.zeros(
            n_atoms,
            backbone.sph_feature_size,
            backbone.sphere_channels,
            device=device,
            dtype=positions.dtype,
        )
        x_message[:, 0, :] = backbone.sphere_embedding(atom_types)

        sys_node_embedding = csd_mixed_emb[batch_full]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        ###
        # Hook to allow MOLE
        ###
        backbone.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=batch_full,
            edge_index=edge_index,
        )
        backbone.log_MOLE_stats()

        # edge degree embedding
        dist_scaled = edge_distance / backbone.cutoff
        edge_envelope = backbone.envelope(dist_scaled).reshape(-1, 1, 1)
        edge_distance_embedding = backbone.distance_expansion(
            edge_distance
        )
        source_embedding = backbone.source_embedding(
            atom_types.long()[edge_index[0]]
        )
        target_embedding = backbone.target_embedding(
            atom_types.long()[edge_index[1]]
        )
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding),
            dim=1,
        )

        # Pre-fuse envelope into wigner_inv
        wigner_inv_envelope = wigner_inv * edge_envelope

        x_message = backbone.edge_degree_embedding(
            x_message,
            x_edge,
            edge_index,
            wigner_inv_envelope,
            # gp_node_offset,
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        # Get edge embeddings for each layer
        # General backend: raw x_edge (rad_func computed inside SO2_Convolution)
        # Fast backends: precomputed radials
        x_edge_per_layer = backbone.backend.get_layer_radial_emb(x_edge, backbone)

        for i in range(backbone.num_layers):
            x_message = backbone.blocks[i](
                x_message,
                x_edge_per_layer[i],
                edge_index,
                wigner,
                wigner_inv_envelope,
                total_atoms_across_gp_ranks=atom_types.shape[
                    0
                ],
                sys_node_embedding=sys_node_embedding,
                # node_offset=gp_node_offset,
            )
            # balance any channels requested
            x_message = backbone.balance_channels(
                x_message,
                charge=charge,
                spin=spin,
                natoms=torch.tensor([atom_types.shape[0]], device=device),
                batch=batch_full,
            )

            if i < backbone.num_layers - 1: #Maybe we should push this even on the last layer? Does it cause test failures?
                # Fetch updated features from neighbors (only send local features)
                # UMA features are shape [(lmax + 1)^2, sphere_dim],
                # we need 1D (lmax+1)^2 * sphere_dim
                sph_size = x_message.shape[1]
                channels = x_message.shape[2]

                x_message_local_flat = x_message[:n_local_atoms].flatten(start_dim=1)
                x_message_flat = FeaturePushToExchange.apply(
                        x_message_local_flat, distributed_state, self.feature_exchanger
                )
                x_message = x_message_flat.view(x_message_flat.shape[0], sph_size, channels)

        # Final layer norm
        x_message = backbone.norm(x_message)

        out = {
            "node_embedding": x_message,
            "batch": batch_full,
        }

        # Compute local energy and forces
        # energy = backbone.output_head(x_message)
        # energy.backward()
        # assert (
        #     local_positions.grad is not None
        # ), "Gradient should not be None after backward()"
        # forces = -local_positions.grad

        #MLP_EFS_Head FORWARD
        output_head = main_model.output_heads["energyandforcehead"].head

        energy_key = f"{output_head.prefix}_energy" if output_head.prefix else "energy"
        forces_key = f"{output_head.prefix}_forces" if output_head.prefix else "forces"
        stress_key = f"{output_head.prefix}_stress" if output_head.prefix else "stress"
        hessian_key = f"{output_head.prefix}_hessian" if output_head.prefix else "hessian"

        outputs = {}

        # Use shared energy computation from parent class
        if hasattr(self.predict_unit.model, "module"):
            self.predict_unit.model.module.prepare_for_inference(
                data, self.predict_unit.inference_settings
            )
            working_model = self.predict_unit.model.module
        else:
            self.predict_unit.model.prepare_for_inference(
                data, self.predict_unit.inference_settings
            )
            working_model = self.predict_unit.model

        if not self.predict_unit.inference_settings.merge_mole:
            active_backbone_coefficients = None
            container = None
            
            for m in working_model.modules():
                if hasattr(m, "global_mole_tensors") and m.global_mole_tensors is not None:
                    coefs = getattr(m.global_mole_tensors, "expert_mixing_coefficients", None)
                    sizes = getattr(m.global_mole_tensors, "mole_sizes", None)
                    if coefs is not None and coefs.shape[-1] == 64 and sizes is not None:
                        active_backbone_coefficients = coefs
                        container = m.global_mole_tensors
                        break
            
            if container is not None:
                for m in working_model.modules():
                    if type(m).__name__ == "MOLE":
                        layer_experts = getattr(m, "num_experts", None) or (m.weight.shape[0] if hasattr(m, "weight") else None)

                        import copy
                        m.global_mole_tensors = copy.copy(container)

                        if layer_experts == 64 and active_backbone_coefficients is not None:
                            m.global_mole_tensors.expert_mixing_coefficients = active_backbone_coefficients

                        elif layer_experts == 7:
                            dataset_name = getattr(data, "dataset_name", "omol")
                            dataset_idx = 0

                            for attr_name in ["dataset_id_map", "dataset_mapping", "datasets", "tasks"]:
                                attr = getattr(output_head, attr_name, None)
                                if isinstance(attr, dict) and dataset_name in attr:
                                    dataset_idx = attr[dataset_name]
                                    break
                                elif isinstance(attr, (list, tuple)) and dataset_name in attr:
                                    dataset_idx = attr.index(dataset_name)
                                    break

                            batch_size = data.natoms.shape[0] if hasattr(data, "natoms") else 1
                            head_coefs = torch.zeros(batch_size, 7, device=device)
                            head_coefs[:, dataset_idx] = 1.0

                            m.global_mole_tensors.expert_mixing_coefficients = head_coefs

        energy, energy_part = compute_energy(
            out,
            output_head.energy_block,
            batch_full,
            num_systems=1,
            natoms=n_atoms,
            reduce=output_head.reduce,
        )

        outputs[energy_key] = {"energy": energy} if output_head.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = out["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if output_head.wrap_property else embeddings
            )

        # Determine if we need create_graph for higher-order derivatives
        # Hessian computation requires second derivatives, so we need create_graph=True
        create_graph = output_head.training or output_head.regress_config.hessian

        # if output_head.regress_config.stress and not output_head.regress_config.direct_stress:
        #     forces, stress = compute_forces_and_stress(
        #         energy_part,
        #         positions,
        #         cell,
        #         batch=batch_full,  # use batch_full to work with GP reduction
        #         training=create_graph,
        #     )
        #     # TODO should we assume gradient forces always when stress is requested?
        #     outputs[forces_key] = {"forces": forces} if output_head.wrap_property else forces
        #     outputs[stress_key] = {"stress": stress} if output_head.wrap_property else stress
        # elif output_head.regress_config.forces and not output_head.regress_config.direct_forces:
        #     forces = compute_forces(energy_part, positions, training=output_head.training)
        #     outputs[forces_key] = {"forces": forces} if output_head.wrap_property else forces
        # else:
        #     energy.backward()
        #     forces = -local_positions.grad

        energy.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        # if output_head.regress_config.hessian:
        #     if forces is None:
        #         raise ValueError(
        #             "Hessian computation requires forces. "
        #             "Please enable regress_forces or regress_stress."
        #         )

        #     hessian = compute_hessian(
        #         forces,
        #         positions,
        #         vmap=output_head.regress_config.hessian_vmap,
        #         training=create_graph,
        #     )
        #     outputs[hessian_key] = (
        #         {"hessian": hessian} if output_head.wrap_property else hessian
        #     )

        total_energy = dist.reduce(energy.detach(), op=dist.ReduceOp.SUM)
        local_forces = forces[:n_local_atoms]
        return total_energy, local_forces
