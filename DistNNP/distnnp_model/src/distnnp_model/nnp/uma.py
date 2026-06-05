import sys
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
from fairchem.core.units.mlip_unit import (
    InferenceSettings,
    guess_inference_settings,
)

from fairchem.core.models.uma.nn.mole_utils import (
    model_search_and_replace,
    recursive_replace_so2_MOLE,
    replace_MOLE_with_linear,
)
from fairchem.core.models.uma import escn_moe

import torch
import torch.distributed as dist
from typing import Literal, Optional, List

# original_balance = escn_moe.balance_channels_batched

def balance_channels_batched(
    emb: torch.Tensor,
    target: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    target_offset: float = 0.0,
    n_local_atoms: int = None,
) -> torch.Tensor:
    out_emb = emb.clone()
    num_systems = len(natoms)
    n_channels = end_idx - start_idx

    # Extract all channels (local + ghosts)
    channels_to_balance = emb[:, 0, start_idx:end_idx]

    system_sums = torch.zeros(
        num_systems, n_channels, device=emb.device, dtype=emb.dtype
    )
    
    # --- CRITICAL FIX: NUMERATOR ---
    # Only sum the local atoms to prevent double-counting ghosts across ranks
    if n_local_atoms is not None:
        local_batch = batch[:n_local_atoms]
        local_channels = channels_to_balance[:n_local_atoms]
        system_sums.index_add_(0, local_batch, local_channels)
    else:
        system_sums.index_add_(0, batch, channels_to_balance)

    # Reduce partial sums across all ranks so every rank has the exact global sum
    if dist.is_initialized():
        dist.all_reduce(system_sums, op=dist.ReduceOp.SUM)
        # Note: If you want to use Fairchem's graph-parallel backend instead of raw NCCL:
        # if gp_utils.initialized():
        #     system_sums = gp_utils.all_reduce_with_grad(system_sums, group=gp_utils.get_gp_group())

    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    
    # --- CRITICAL FIX: DENOMINATOR ---
    # 'natoms' here is n_global_atoms, ensuring the per-atom correction scales correctly
    corrections = (system_sums - target_sums) / natoms.unsqueeze(1)

    # Apply the correction to ALL atoms in the rank (local + ghosts)
    out_emb[:, 0, start_idx:end_idx] = channels_to_balance - corrections[batch]
    
    return out_emb

class UMAModel(DistNNPModel):
    def __init__(
        self, 
        model_name: str,
        inference_settings: InferenceSettings | str = "default", #"turbo",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        exchanger: Optional[Exchanger] = None,
        merge_mole_experts: bool = False,
        rank: int = 0,
    ):
        DistNNPModel.__init__(self)
        self.log = open(f"/p/lustre1/smolenski1/rank_{rank}.log", "w", buffering=1)
        self.log.write(f"rank {rank} start\n")
        inference_settings = guess_inference_settings(inference_settings)
        inference_settings.merge_mole = False # we do it manually, don't have fairchem try too
        self.predict_unit = pretrained_mlip.get_predict_unit(model_name, inference_settings, overrides, device)
        self.merge_mole_experts = merge_mole_experts
        self._experts_merged = False

        main_model = self.predict_unit.model.module if hasattr(self.predict_unit.model, "module") else self.predict_unit.model
        backbone = main_model.backbone
        self.position_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.atom_type_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.feature_exchangers = [(
            GPUDirectExchanger() if exchanger is None else exchanger
        ) for i in range(backbone.num_layers)]

    def initialize_buffers(
        self, partition_info: PartitionMessagePassingInfo, device
    ) -> None:
        self.position_exchanger.setup_exchanger(
            feature_dim=3, partition_info=partition_info, device=device
        )
        self.atom_type_exchanger.setup_exchanger(
            feature_dim=1, partition_info=partition_info, device=device
        )

        backbone = self.predict_unit.model.module.backbone if hasattr(self.predict_unit.model, "module") else self.predict_unit.model.backbone
        total_feature_dim = backbone.sph_feature_size * backbone.sphere_channels
        for feature_exchanger in self.feature_exchangers:
            feature_exchanger.setup_exchanger(
                feature_dim=total_feature_dim, partition_info=partition_info, device=device
            )

    def balance_channels(
        self,
        x_message_prime: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        natoms: torch.Tensor,
        batch: torch.Tensor,
        n_local_atoms: int,
    ) -> torch.Tensor:
        backbone = self.predict_unit.model.module.backbone if hasattr(self.predict_unit.model, "module") else self.predict_unit.model.backbone
        if backbone.charge_channel_end > backbone.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms=natoms,
                batch=batch,
                start_idx=backbone.charge_channel_start,
                end_idx=backbone.charge_channel_end,
                target_offset=0.0,
                n_local_atoms=n_local_atoms,
            )
        if backbone.spin_channel_end > backbone.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms=natoms,
                batch=batch,
                start_idx=backbone.spin_channel_start,
                end_idx=backbone.spin_channel_end,
                target_offset=1.0,
                n_local_atoms=n_local_atoms,
            )
        return x_message_prime

    def __call__(
        self,
        distributed_state: DistNNPState,
        batch: torch.Tensor,
        batch_full: torch.Tensor,     
        datasets: str | List[str], 
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        device: Literal["cuda", "cpu"] | None = None,
    ):
        if isinstance(datasets, str):
            datasets = [datasets]

        #ESCN FORWARD
        local_mol_info = distributed_state.local_molecular_info
        global_mol_info = distributed_state.global_molecular_info
        sim_config = distributed_state.sim_config
        partition_info = distributed_state.local_partition_info

        prev_local_positions = local_mol_info.atom_positions
        local_positions = prev_local_positions.clone().requires_grad_(True)
        local_atom_types = local_mol_info.atom_types
        global n_local_atoms
        n_local_atoms = local_mol_info.num_atoms
        n_global_atoms = global_mol_info.num_atoms
        global_atom_types = global_mol_info.atom_types
        global_positions = global_mol_info.atom_positions

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
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                data=type(
                    "Data",
                    (object,),
                    {
                        # "pos": positions,
                        "pos": global_positions,
                        "cell": cell.unsqueeze(0),
                        # "natoms": torch.tensor([n_atoms], dtype=torch.long, device=device),
                        "natoms": torch.tensor([n_global_atoms], dtype=torch.long, device=device),
                        "pbc": pbc,
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        local_global_indices = partition_info.partition_indices[:n_local_atoms]  # [n_local]
        ghost_positions = positions[n_local_atoms:]
        ghost_global_indices = torch.cdist(
            ghost_positions.float(), 
            global_mol_info.atom_positions.float()
        ).argmin(dim=1)
        all_global_indices = torch.cat([local_global_indices, ghost_global_indices])  # [n_local+n_ghosts]
        print(f"[rank {distributed_state.dist_env.global_rank}] "
              f"local_global_indices: {local_global_indices}, "
              f"ghost_global_indices: {ghost_global_indices}",
              f"all_global_indices: {all_global_indices}",
              file=self.log)

        # Invert: global_idx -> local position index (-1 if not present on this rank)
        global_to_local = torch.full((n_global_atoms,), -1, dtype=torch.long, device=device)
        global_to_local[all_global_indices] = torch.arange(
            all_global_indices.shape[0], dtype=torch.long, device=device
        )

        # Keep only edges where both src and tgt are present on this rank
        global_src, global_tgt = edge_index[0], edge_index[1]
        src_local = global_to_local[global_src]
        tgt_local = global_to_local[global_tgt]
        edge_mask = (src_local >= 0) & (tgt_local >= 0)

        edge_index = torch.stack([src_local[edge_mask], tgt_local[edge_mask]], dim=0)
        cell_offsets = cell_offsets[edge_mask]

        print(f"edge_mask.sum(): {edge_mask.sum()}",
              f"edge_index.shape: {edge_index.shape}",
              file=self.log)

        print(f"FROM DISTUMA: edge_index: {edge_index}", file=self.log)

        src, tgt = edge_index[0], edge_index[1]
        edge_distance_vec = positions[tgt] - positions[src] + torch.einsum("ni,nij->nj", cell_offsets, cell.unsqueeze(0))
        edge_distance = torch.norm(edge_distance_vec, p=2, dim=-1)

        num_systems = int(batch.max().item()) + 1 
        if charge is None:
            charge = torch.zeros(num_systems, dtype=torch.long, device=device)
        if spin is None:
            spin = torch.ones(num_systems, dtype=torch.long, device=device)

        n_edges = edge_index.shape[1]
        data = AtomicData(
            positions,
            atom_types,
            cell.unsqueeze(0),
            pbc,
            torch.tensor([n_atoms], dtype=torch.long, device=device),
            edge_index,
            cell_offsets,
            torch.tensor([n_edges], dtype=torch.long, device=device),
            charge,
            spin,
            torch.zeros(n_atoms, dtype=torch.long, device=device),
            torch.zeros(n_atoms, dtype=torch.long, device=device),
        )
        if hasattr(self.predict_unit.model, "module"):
            self.predict_unit.model.module.prepare_for_inference(
                data, self.predict_unit.inference_settings
            )
        else:
            self.predict_unit.model.prepare_for_inference(
                data, self.predict_unit.inference_settings
            )

        csd_mixed_emb = backbone.csd_embedding(
            charge=charge,
            spin=spin,
            dataset=datasets,
        )
        print(f"FROM DISTUMA: csd_mixed_emb: {csd_mixed_emb}", file=self.log)
        known_be_list = []
        def routing_hook(module, inputs, output):
            known_be_list.append(output.detach().clone())

        if getattr(backbone, "num_experts", 0) > 0:
            hook_handle = None
            if hasattr(backbone, "routing_mlp") and backbone.routing_mlp is not None:
                hook_handle = backbone.routing_mlp.register_forward_hook(routing_hook)
                
            backbone.set_MOLE_coefficients(
                atomic_numbers_full=global_atom_types,
                batch_full=torch.zeros_like(global_atom_types),
                csd_mixed_emb=csd_mixed_emb,
            )
            
            if hook_handle is not None:
                hook_handle.remove()

        known_be = known_be_list[0] if len(known_be_list) > 0 else None

        if self.merge_mole_experts and not self._experts_merged:
            model_search_and_replace(
                backbone, recursive_replace_so2_MOLE, replace_MOLE_with_linear
            )
            
            if hasattr(main_model, "output_heads"):
                for head_key, head_wrapper in main_model.output_heads.items():
                    actual_head = head_wrapper.head if hasattr(head_wrapper, "head") else head_wrapper
                    if hasattr(actual_head, "merge_mole"):
                        actual_head.merge_mole(data)
                        
            backbone.routing_mlp = None
            backbone.composition_embedding = None
            backbone.num_experts = 0
            if hasattr(main_model, "num_experts"):
                main_model.num_experts = 0
            
            if hasattr(backbone, "use_composition_embedding"):
                backbone.use_composition_embedding = False
                
            self._experts_merged = True

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

        x_message = torch.zeros(
            n_atoms,
            backbone.sph_feature_size,
            backbone.sphere_channels,
            device=device,
            dtype=positions.dtype,
        )
        x_message[:, 0, :] = backbone.sphere_embedding(atom_types)

        batch_full = torch.zeros(n_atoms, dtype=batch.dtype, device=batch.device)
        sys_node_embedding = csd_mixed_emb[batch_full]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        if getattr(backbone, "num_experts", 0) > 0:
            backbone.set_MOLE_sizes(
                nsystems=csd_mixed_emb.shape[0],
                batch_full=batch_full,
                edge_index=edge_index,
            )
            backbone.log_MOLE_stats()

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

        wigner_inv_envelope = wigner_inv * edge_envelope

        local_tgt_mask = edge_index[1] < n_local_atoms
        edge_index_local = edge_index[:, local_tgt_mask]
        print(f"FROM DISTUMA: edge_index_local: {edge_index_local}", file=self.log)
        x_edge_local = x_edge[local_tgt_mask]
        wigner_inv_envelope_local = wigner_inv_envelope[local_tgt_mask]

        x_message = backbone.edge_degree_embedding(
            x_message,
            x_edge_local,
            edge_index_local,
            wigner_inv_envelope_local,
            node_offset=0,
        )
        print(f"FROM DISTUMA: x_message after edge_degree_embedding: {x_message}", file=self.log)

        sph_size = x_message.shape[1]
        channels = x_message.shape[2]

        x_message_local_flat = x_message[:n_local_atoms].flatten(start_dim=1)
        x_message_flat = FeaturePushToExchange.apply(
            x_message_local_flat, distributed_state, self.feature_exchangers[0]
        )
        x_message = x_message_flat.view(x_message_flat.shape[0], sph_size, channels)

        x_edge_per_layer = backbone.backend.get_layer_radial_emb(x_edge, backbone)

        for i in range(backbone.num_layers):
            print(f"FROM DISTUMA: layer {i} x_message[:n_local_atoms]: {x_message[:n_local_atoms]}", file=self.log)
            print(f"FROM DISTUMA: layer {i} x_message[n_local_atoms:]: {x_message[n_local_atoms:]}", file=self.log)
            sys.stdout.flush()
            x_message = backbone.blocks[i](
                x_message,
                x_edge_per_layer[i],
                edge_index,
                wigner,
                wigner_inv_envelope,
                total_atoms_across_gp_ranks=n_global_atoms,
                sys_node_embedding=sys_node_embedding,
            )
            x_message = self.balance_channels(
                x_message,
                charge=charge,
                spin=spin,
                natoms=torch.tensor([n_global_atoms], device=device), 
                batch=batch_full,
                n_local_atoms=n_local_atoms,
            )

            if i < backbone.num_layers - 1:
                sph_size = x_message.shape[1]
                channels = x_message.shape[2]

                x_message_local_flat = x_message[:n_local_atoms].flatten(start_dim=1)
                x_message_flat = FeaturePushToExchange.apply(
                        x_message_local_flat, distributed_state, self.feature_exchangers[i + 1]
                )
                x_message = x_message_flat.view(x_message_flat.shape[0], sph_size, channels)

        x_message = backbone.norm(x_message)

        out = {
            "node_embedding": x_message,
            "batch": batch_full,
        }

        head_wrapper = main_model.output_heads["energyandforcehead"]
        output_head = head_wrapper.head

        energy_key = f"{output_head.prefix}_energy" if output_head.prefix else "energy"
        forces_key = f"{output_head.prefix}_forces" if output_head.prefix else "forces"
        stress_key = f"{output_head.prefix}_stress" if output_head.prefix else "stress"
        hessian_key = f"{output_head.prefix}_hessian" if output_head.prefix else "hessian"

        outputs = {}

        local_out = {
            "node_embedding": out["node_embedding"][:n_local_atoms],
            "batch": batch,
        }

        expert_idx = head_wrapper.dataset_name_to_exp[backbone.dataset_mapping["omol"]]
        head_wrapper.global_mole_tensors.expert_mixing_coefficients = torch.zeros(
            1,
            len(head_wrapper.dataset_name_to_exp),
            dtype=positions.dtype,
            device=device,
        )
        head_wrapper.global_mole_tensors.expert_mixing_coefficients[0, expert_idx] = 1.0
        head_wrapper.global_mole_tensors.mole_sizes = torch.tensor(
            [n_global_atoms], dtype=torch.long, device=device
        )

        energy, energy_part = compute_energy(
            local_out,
            output_head.energy_block,
            batch,
            num_systems=1,
            natoms=n_local_atoms,
            reduce=output_head.reduce,
        )

        outputs[energy_key] = {"energy": energy} if output_head.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = out["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if output_head.wrap_property else embeddings
            )

        energy_part.backward()
        assert (
            local_positions.grad is not None
        ), "Gradient should not be None after backward()"
        forces = -local_positions.grad

        local_energy = energy.detach().clone()
        local_forces = forces[:n_local_atoms]
        return local_energy, local_forces
