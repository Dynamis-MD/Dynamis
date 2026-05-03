from distnnp_model.potential import DistNNPModel
from distnnp_model.comm.exchanger import Exchanger, GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_state import DistNNPState, radius_graph_pbc_cuda, edge_distances

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces_and_stress,
    compute_forces,
    compute_hessian,
)
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit import InferenceSettings
import torch
import torch.distributed as dist
from typing import Literal, Optional

class UMAModel(DistNNPModel):
    def __init__(
        self, 
        model_name: str,
        inference_settings: InferenceSettings | str = "turbo",
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
        self.predict_unit = pretrained_mlip.get_predict_unit(model_name, inference_settings, overrides, device)

    def __call__(
        self,
        distributed_state: DistNNPState,
        batch: torch.Tensor,
        batch_full: torch.Tensor,   # ?
        dataset: str | None = None, # ?
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
    ):
        #ESCN FORWARD
        backbone = self.predict_unit.model.module.backbone

        mol_info = distributed_state.global_molecular_info
        sim_config = distributed_state.sim_config
        part_info = distributed_state.local_partition_info

        prev_local_positions = part_info.local_atom_positions
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = part_info.partition_indices
        n_local_atoms = local_positions.shape[0]

        positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self.position_exchanger
        )

        atom_types = FeaturePushToExchange.apply(
            local_atom_types.float(), distributed_state, self.atom_type_exchanger
        ).long()

        n_atoms = positions.shape[0]

        cell = sim_config.cell
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
        edge_distance_vec, edge_distance = edge_distances(positions, edge_index)
        # gp_node_offset = 0

        csd_mixed_emb = backbone.csd_embedding(
            charge=charge,
            spin=spin,
            dataset=dataset,
        )

        backbone.set_MOLE_coefficients(
            atomic_numbers_full=atom_types.float(),
            batch_full=batch_full,
            csd_mixed_emb=csd_mixed_emb,
        )

        # Enable gradients for autograd-based force/stress computation.
        # Must be set before graph generation so the computation graph
        # tracks positions and cell through edge distance calculations.
        if not backbone.regress_config.direct_forces:
            if backbone.regress_config.forces or backbone.regress_config.stress:
                positions.requires_grad_(True)
            if backbone.regress_config.stress:
                cell.requires_grad_(True)

        wigner, wigner_inv = backbone._get_rotmat_and_wigner(
            edge_distance_vec = edge_distance_vec
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
            device=positions.device,
            dtype=positions.dtype,
        )
        x_message[:, 0, :] = backbone.sphere_embedding(atom_types)

        sys_node_embedding = csd_mixed_emb[batch]
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
                natoms=atom_types.shape[0],
                batch=batch,
            )

            if i < len(backbone.interactions) - 1:
                # Fetch updated features from neighbors (only send local features)
                x_message = FeaturePushToExchange.apply(
                        x_message[:n_local_atoms], distributed_state, self._feature_exchanger
                )

        # Final layer norm
        x_message = backbone.norm(x_message)

        out = {
            "node_embedding": x_message,
            "batch": batch,
        }

        # Compute local energy and forces
        # energy = backbone.output_head(x_message)
        # energy.backward()
        # assert (
        #     local_positions.grad is not None
        # ), "Gradient should not be None after backward()"
        # forces = -local_positions.grad

        #MLP_EFS_Head FORWARD
        output_head = self.predict_unit.model.module.output_heads["energyandforcehead"]

        energy_key = f"{output_head.prefix}_energy" if output_head.prefix else "energy"
        forces_key = f"{output_head.prefix}_forces" if output_head.prefix else "forces"
        stress_key = f"{output_head.prefix}_stress" if output_head.prefix else "stress"
        hessian_key = f"{output_head.prefix}_hessian" if output_head.prefix else "hessian"

        outputs = {}

        # Use shared energy computation from parent class
        energy, energy_part = compute_energy(
            out,
            output_head.energy_block,
            batch,
            num_systems=1,
            natoms=n_atoms,
            reduce=output_head.reduce,
        )

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = out["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )

        # Determine if we need create_graph for higher-order derivatives
        # Hessian computation requires second derivatives, so we need create_graph=True
        create_graph = self.training or self.regress_config.hessian

        if self.regress_config.stress and not self.regress_config.direct_stress:
            forces, stress = compute_forces_and_stress(
                energy_part,
                positions,
                cell,
                batch=batch_full,  # use batch_full to work with GP reduction
                training=create_graph,
            )
            # TODO should we assume gradient forces always when stress is requested?
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
        elif self.regress_config.forces and not self.regress_config.direct_forces:
            forces = compute_forces(energy_part, positions, training=self.training)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        else:
            energy.backward()
            forces = -local_positions.grad

        if self.regress_config.hessian:
            if forces is None:
                raise ValueError(
                    "Hessian computation requires forces. "
                    "Please enable regress_forces or regress_stress."
                )

            hessian = compute_hessian(
                forces,
                positions,
                vmap=self.regress_config.hessian_vmap,
                training=create_graph,
            )
            outputs[hessian_key] = (
                {"hessian": hessian} if self.wrap_property else hessian
            )

        total_energy = dist.reduce(energy.detach(), op=dist.ReduceOp.SUM)
        local_forces = forces[:n_local_atoms]
        return total_energy, local_forces
