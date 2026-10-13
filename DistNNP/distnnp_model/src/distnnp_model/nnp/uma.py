from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import numpy as np
from torch.profiler import record_function
from typing import List, Literal
try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets.atomic_data import AtomicData, warn_if_upcasting
    from fairchem.core.common.utils import conditional_grad
    from fairchem.core.common.registry import registry
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
    from fairchem.core.models.uma.nn.mole_utils import (
        model_search_and_replace,
        recursive_replace_so2_MOLE,
        replace_MOLE_with_linear,
    )
    from fairchem.core.units.mlip_unit import (
        InferenceSettings,
        guess_inference_settings,
    )
    from fairchem.core.units.mlip_unit.utils import tf32_context_manager
    from fairchem.core.units.mlip_unit.single_atom_patch import single_atom_prediction_from_lookup
except ImportError as e:
    raise ImportError("fairchem-core must be installed to use UMAModel") from e

from distnnp_model.potential import DistNNPModel
from distnnp_model.comm.exchanger import Exchanger, GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from dynamis_state import (
    DistNNPState,
    PartitionMessagePassingInfo,
    radius_graph_pbc_cuda,
)

# Shared across every backend; re-exported under the historical private name so
# existing `from distnnp_model.nnp.uma import _EV_PER_A_TO_AMU_A_PER_FS2` keeps
# resolving to the one definition in distnnp_model.units.
from distnnp_model.units import EV_PER_A_TO_AMU_A_PER_FS2

_EV_PER_A_TO_AMU_A_PER_FS2 = EV_PER_A_TO_AMU_A_PER_FS2


def balance_channels_batched(
    emb: torch.Tensor,
    target: torch.Tensor,
    natoms_global: torch.Tensor,
    batch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    num_local: int,
    target_offset: float = 0.0,
) -> torch.Tensor:
    out_emb = emb.clone()
    num_systems = len(natoms_global)
    n_channels = end_idx - start_idx

    channels_to_balance_local = emb[:num_local, 0, start_idx:end_idx]
    
    system_sums = torch.zeros(
        num_systems, n_channels, device=emb.device, dtype=emb.dtype
    )
    system_sums.index_add_(0, batch[:num_local], channels_to_balance_local)

    if dist.is_initialized():
        system_sums = dist_nn.all_reduce(system_sums, op=torch.distributed.ReduceOp.SUM)

    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    
    corrections = (system_sums - target_sums) / natoms_global.unsqueeze(1)

    channels_to_balance_all = emb[:, 0, start_idx:end_idx]
    out_emb[:, 0, start_idx:end_idx] = channels_to_balance_all - corrections[batch]
    
    return out_emb

@registry.register_model("distributed_escnmd_backbone")
class DistributedESCNMDBackbone(eSCNMDMoeBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize exchanger with dummy values, will be updated later
        self._feature_exchangers = []
        self._distributed_state = None
        self.num_local = 0
        self.profiler = None  # CUDAEventProfiler for detailed timing
        self._cached_mole_coefficients = None  # MOLE cache for first-call pattern
        self.differentiable_exchange = False  # Differentiable feature exchange for correct force gradients
    
    def set_profiler(self, profiler):
        """Set profiler for detailed model timing."""
        self.profiler = profiler

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        if self.num_experts == 0:
            return
        
        # OPTIMIZATION: Cache MOLE coefficients - compute once on first call, reuse forever.
        # In MD simulations with merge_mole=True, composition is static.
        cached = getattr(self, "_cached_mole_coefficients", None)
        if cached is not None:
            self.global_mole_tensors.expert_mixing_coefficients = cached
            return
        
        
        # Ensure inputs are on correct device
        if self.use_composition_embedding:
             device = self.composition_embedding.weight.device
             atomic_numbers_full = atomic_numbers_full.to(device)
             batch_full = batch_full.to(device)

        with torch.autocast(device_type=atomic_numbers_full.device.type, enabled=False):
            embeddings = []
            if self.use_composition_embedding:
                # Distributed Logic: Compute Global Mean Composition
                if dist.is_initialized():
                    num_local = self.num_local
                    if num_local is None:
                        num_local = atomic_numbers_full.shape[0]
                    
                    if num_local > atomic_numbers_full.shape[0]:
                         num_local = atomic_numbers_full.shape[0]

                    local_an = atomic_numbers_full[:num_local]
                    local_comp = self.composition_embedding(local_an)
                    local_sum = local_comp.sum(0)
                    local_count = torch.tensor([num_local], device=device, dtype=local_sum.dtype)
                    
                    # Handle CPU tensors with NCCL backend (which doesn't support CPU all_reduce)
                    is_cpu = local_sum.device.type == "cpu"
                    if is_cpu and dist.get_backend() == "nccl":
                        local_sum_cuda = local_sum.cuda()
                        local_count_cuda = local_count.cuda()
                        
                        local_sum_cuda = dist_nn.all_reduce(local_sum_cuda)
                        dist.all_reduce(local_count_cuda)
                        
                        local_sum = local_sum_cuda.cpu()
                        local_count = local_count_cuda.cpu()
                    else:
                        local_sum = dist_nn.all_reduce(local_sum)
                        dist.all_reduce(local_count)
                    
                    # Match eSCNMDMoeBackbone behavior for include_self
                    # Default version 1.0 uses include_self=True
                    if np.isclose(self.model_version, 1.0).item():
                        local_count = local_count + 1.0 # Denominator adds 1 (implicit 0-vector self)
                        # Numerator adds 0 (self value) -> no change
                        
                    global_mean = local_sum / local_count
                    
                    # Assume 1 system for MD inference usually
                    # If multiple systems, this logic needs batch-awareness?
                    # For now, replicate to match csd_mixed_emb shape [B, C]
                    # But index_reduce reduces to unique batch indices.
                    # In MD inference, batch is usually all 0s.
                    composition = global_mean.unsqueeze(0) 
                    
                else:
                    # Fallback to local calculation (Serial or Single Rank)
                    composition_by_atom = self.composition_embedding(atomic_numbers_full)
                    composition = composition_by_atom.new_zeros(
                        csd_mixed_emb.shape[0],
                        self.sphere_channels,
                    ).index_reduce_(
                        0,
                        batch_full,
                        composition_by_atom,
                        reduce="mean",
                        include_self=np.isclose(self.model_version, 1.0).item(),
                    )
                embeddings.append(composition.unsqueeze(0))
            embeddings.append(csd_mixed_emb[None])

            expert_mixing_coefficients_before_norm = self.routing_mlp(
                torch.vstack(embeddings)
                .transpose(0, 1)
                .reshape(csd_mixed_emb.shape[0], -1)
            )
            coefficients = self.mole_expert_coefficient_norm(
                self.mole_dropout(expert_mixing_coefficients_before_norm)
            )
            
            # Store in cache
            self._cached_mole_coefficients = coefficients
            
            self.global_mole_tensors.expert_mixing_coefficients = coefficients

    def balance_channels(
        self,
        x_message_prime: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        natoms: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.charge_channel_end > self.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms_global=natoms,
                batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end,
                num_local=self.num_local,
                target_offset=0.0,
            )
        if self.spin_channel_end > self.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms_global=natoms,
                batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end,
                num_local=self.num_local,
                target_offset=1.0,
            )
        return x_message_prime

    def forward(self, data_dict):
        # Pre-computation (same as original)
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )

        num_local = None
        if "num_local" in data_dict:
             num_local = data_dict["num_local"]
             if isinstance(num_local, torch.Tensor) and num_local.numel() == 1:
                 num_local = num_local.item()
        self.num_local = num_local
        
        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )
        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)

        if self.profiler:
            self.profiler.start("generate_graph")
        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)
        if self.profiler:
            self.profiler.stop("generate_graph")
            
        # DISTNNP OPTIMIZATION: Filter redundant edges early
        # Keep reference to original edges for MOLE stats
        original_edge_index = graph_dict["edge_index"]
        
        # Filter if num_local is valid
        if num_local is not None and isinstance(num_local, int) and num_local < data_dict["pos"].shape[0]:
            self.num_local = num_local

            mask = graph_dict["edge_index"][1] < num_local

            # Apply filter to all edge tensors
            graph_dict["edge_index"] = graph_dict["edge_index"][:, mask]
            
            if "edge_distance" in graph_dict:
                 graph_dict["edge_distance"] = graph_dict["edge_distance"][mask]
                 
            if "edge_distance_vec" in graph_dict:
                graph_dict["edge_distance_vec"] = graph_dict["edge_distance_vec"][mask]

            if "cell_offsets" in graph_dict:
                graph_dict["cell_offsets"] = graph_dict["cell_offsets"][mask]

        if graph_dict["edge_index"].numel() == 0:
             raise ValueError(
                f"No edges found. Check cutoff {self.cutoff}."
            )

        if self.profiler:
            self.profiler.start("wigner")
        with record_function("obtain wigner"):
            wigner, wigner_inv = self._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"],
            )
            coefficient_index = (
                self.coefficient_index if self.mmax != self.lmax else None
            )
            wigner, wigner_inv = self.backend.prepare_wigner(
                wigner,
                wigner_inv,
                self.mappingReduced,
                coefficient_index,
            )
        if self.profiler:
            self.profiler.stop("wigner")

        if self.profiler:
            self.profiler.start("atom_embedding")
        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])
        if self.profiler:
            self.profiler.stop("atom_embedding")

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=original_edge_index,
        )
        self.log_MOLE_stats()

        if self.profiler:
            self.profiler.start("edge_embedding")
        with record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            
            # Exchange before edge_degree_embedding? 
            # edge_degree_embedding uses neighbors. If neighbors are ghosts, we need their features.
            # But x_message is just initialized from atomic numbers. 
            # Atomic numbers of ghosts are already known (exchanged in structural halo).
            # So x_message for ghosts is locally computable at this stage!
            # No exchange needed yet.
            
            wigner_inv_envelope = wigner_inv * edge_envelope

            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_inv_envelope,
                data_dict["gp_node_offset"],
            )
            if self._feature_exchangers:
                exchanger = self._feature_exchangers[-1]
                n_local = self.num_local
                orig_shape = x_message.shape # [n_atoms, sph_feature_size, sphere_channels]
                feat_dim = orig_shape[1] * orig_shape[2]
                x_local_flat = x_message[:n_local].reshape(n_local, feat_dim)

                x_full_flat = FeaturePushToExchange.apply(
                    x_local_flat, self._distributed_state, exchanger
                )

                n_full = x_full_flat.shape[0]
                x_message = x_full_flat.reshape(n_full, orig_shape[1], orig_shape[2])
        if self.profiler:
            self.profiler.stop("edge_embedding")

        # Update spherical node embeddings (Message Passing Layers)
        # Prepare local_edge_index for efficient message passing
        # We only need to aggregate messages INTO local atoms.
        # Edges where destination (edge_index[1]) is a ghost atom can be skipped.

        effective_num_local = self.num_local if self.num_local > 0 else data_dict["pos"].shape[0]

        local_dest_mask = graph_dict["edge_index"][1] < effective_num_local

        local_edge_index = graph_dict["edge_index"][:, local_dest_mask]

        # === OVERLAP OPTIMIZATION (Area 3) ===
        # Check config flag first to avoid any overhead when disabled
        use_overlap = getattr(self, 'enable_overlap', False)

        with record_function("layer_radial_emb"):
            x_edge_per_layer = self.backend.get_layer_radial_emb(x_edge, self)
        
        if use_overlap:
            # Only compute overlap-related tensors when enabled
            _could_use_overlap = (hasattr(self, 'interior_mask') and
                                  self.interior_mask is not None and
                                  self.interior_mask.numel() > 0 and
                                  dist.get_world_size() > 1)
            
            if not _could_use_overlap:
                use_overlap = False
            else:
                # Filter x_edge for overlap path
                local_x_edge_per_layer = [x_edge_per_layer[i][local_dest_mask] for i in range(self.num_layers)]
                
                # Build masks for interior edges (dst is interior atom) 
                # and boundary edges (dst is boundary atom)
                interior_atom_mask = self.interior_mask.to(local_edge_index.device)
                
                # Edge goes to interior atom if dst is marked as interior
                interior_edge_mask = interior_atom_mask[local_edge_index[1]]
                boundary_edge_mask = ~interior_edge_mask
                
                # Pre-filter edges for interior and boundary
                interior_edge_index = local_edge_index[:, interior_edge_mask]
                boundary_edge_index = local_edge_index[:, boundary_edge_mask]
                
                # === Area 3 Fix: Pre-compute Wigner filtered tensors ONCE (not per-layer) ===
                def _filter_wigner_once(wigner_dict, mask, full_mask):
                    """Filter wigner dict by edge mask, relative to local_dest_mask"""
                    if wigner_dict is None:
                        return None
                    if isinstance(wigner_dict, dict):
                        filtered = {}
                        for k, v in wigner_dict.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] == graph_dict["edge_index"].shape[1]:
                                filtered[k] = v[full_mask][mask]
                            else:
                                filtered[k] = v
                        return filtered
                    elif isinstance(wigner_dict, torch.Tensor):
                        return wigner_dict[full_mask][mask]
                    return wigner_dict
                
                # Log overlap stats once (skip under torch.compile to avoid dynamo errors)
                if not hasattr(self, '_logged_overlap_stats') and not torch.compiler.is_compiling():
                    interior_count = interior_atom_mask.sum().item()
                    total_local = self.num_local
                    interior_edges = interior_edge_index.shape[1]
                    boundary_edges = boundary_edge_index.shape[1]
                    if self.exchanger.rank == 0:
                        print(f"Overlap enabled: {interior_count}/{total_local} interior atoms ({100*interior_count/total_local:.1f}%)")
                        print(f"  Interior edges: {interior_edges}, Boundary edges: {boundary_edges}")
                    self._logged_overlap_stats = True
        
        # Compute correct local count for exchange calls
        _local_count_for_exchange = self.num_local

        # Update spherical node embeddings (Message Passing Layers)
        for i in range(self.num_layers):
            # === Wigner filtering: first iteration only (results reused) ===
            if i == 0:
                with record_function("wigner_filter"):
                    local_wigner = wigner
                    if isinstance(wigner, dict):
                        local_wigner = {}
                        for k, v in wigner.items():
                            if isinstance(v, torch.Tensor) and v.shape[0] == graph_dict["edge_index"].shape[1]:
                                 local_wigner[k] = v[local_dest_mask]
                            else:
                                 local_wigner[k] = v
                    elif isinstance(wigner, torch.Tensor):
                         local_wigner = wigner[local_dest_mask]

                    local_wigner_inv = wigner_inv
                    if wigner_inv is not None:
                         if isinstance(wigner_inv, dict):
                             assert False, "wigner_inv should be torch.Tensor"
                             local_wigner_inv = {}
                             for k, v in wigner_inv.items():
                                if isinstance(v, torch.Tensor) and v.shape[0] == graph_dict["edge_index"].shape[1]:
                                     local_wigner_inv[k] = v[local_dest_mask]
                                else:
                                     local_wigner_inv[k] = v
                         elif isinstance(wigner_inv, torch.Tensor):
                             local_wigner_inv = wigner_inv[local_dest_mask]

                    local_edge_envelope = edge_envelope[local_dest_mask] if edge_envelope is not None else None
                    local_wigner_inv_envelope = local_wigner_inv * local_edge_envelope
                    local_x_edge_per_layer = [x_edge_per_layer[i][local_dest_mask] for i in range(self.num_layers)]

            # Now run the block with updated ghost features
            if self.profiler:
                self.profiler.start(f"mp_block_{i}")
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    local_x_edge_per_layer[i],
                    local_edge_index,
                    local_wigner,
                    local_wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[0],
                    sys_node_embedding=sys_node_embedding,
                    node_offset=data_dict["gp_node_offset"],
                )
                x_message = self.balance_channels(
                    x_message,
                    charge=data_dict["charge"],
                    spin=data_dict["spin"],
                    natoms=data_dict["num_global"],
                    batch=data_dict["batch"],
                )
            if self.profiler:
                self.profiler.stop(f"mp_block_{i}")

            # === Feature exchange ===
            if self.profiler:
                self.profiler.start(f"feature_exchange_{i}")
            if self._feature_exchangers:
                exchanger = self._feature_exchangers[i]
                n_local = _local_count_for_exchange
                orig_shape = x_message.shape  # [n_atoms, sph_feature_size, sphere_channels]
                feat_dim = orig_shape[1] * orig_shape[2]
                x_local_flat = x_message[:n_local].reshape(n_local, feat_dim)

                x_full_flat = FeaturePushToExchange.apply(
                    x_local_flat, self._distributed_state, exchanger
                )

                n_full = x_full_flat.shape[0]
                x_message = x_full_flat.reshape(n_full, orig_shape[1], orig_shape[2])
            if self.profiler:
                self.profiler.stop(f"feature_exchange_{i}")
        # Final layer norm
        x_message = self.norm(x_message)
        
        out = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
        }
        return out

    def merge_MOLE_model(self, data):
        if self.num_experts == 0: # NOTE: if composition changes after the first time this is called, the model will give bad outputs
            return self
            
        # Extract num_local for distributed composition calculation
        if "num_local" in data:
            nl = data["num_local"]
            if isinstance(nl, torch.Tensor):
                self.num_local = int(nl.item())
            else:
                self.num_local = int(nl)
                
        csd_mixed_emb = self.csd_embedding(
            charge=data["charge"],
            spin=data["spin"],
            dataset=data.get("dataset", default=None),
        )
        self.set_MOLE_coefficients(
            atomic_numbers_full=data["atomic_numbers"],
            batch_full=data["batch"],
            csd_mixed_emb=csd_mixed_emb,
        )

        if self.mole_type != "so2":
            raise ValueError("Only mole_type=so2 supported for merging")

        model_search_and_replace(
            self, recursive_replace_so2_MOLE, replace_MOLE_with_linear
        )

        # drop moe parameters from merged model
        self.routing_mlp = None
        self.composition_embedding = None
        self.num_experts = 0

        # create a new non moe model and load weights into there
        # CRITICAL: Use DistributedESCNMDBackbone to preserve our custom class logic!
        # We must disable experts in the new instance so it matches the merged (dense) weights.
        new_kwargs = self.parent_kwargs.copy()
        new_kwargs["num_experts"] = 0
        new_kwargs["use_composition_embedding"] = False # Merged model doesn't use this routing embedding
        
        new_model = DistributedESCNMDBackbone(**new_kwargs)
        new_model.load_state_dict(self.state_dict())
        
        # Preserve distributed components
        new_model._feature_exchangers = self._feature_exchangers
        new_model._distributed_state = self._distributed_state
        new_model.enable_overlap = getattr(self, 'enable_overlap', False)
        new_model.differentiable_exchange = getattr(self, 'differentiable_exchange', False)
        new_model.interior_mask = getattr(self, 'interior_mask', None)
        if hasattr(self, "num_local"):
             new_model.num_local = self.num_local
        
        new_model.eval()
        return new_model

class MaskedHeadWrapper(nn.Module):
    def __init__(self, original_head):
        super().__init__()
        self.original_head = original_head

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "original_head":
                raise
            return getattr(self.original_head, name)
        
    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        prefix = getattr(self.original_head, "prefix", None)
        energy_key = f"{prefix}_energy" if prefix else "energy"
        forces_key = f"{prefix}_forces" if prefix else "forces"

        # 2. Fix ONLY the output scalar energy for distributed parity
        if "num_local" in data:
            outputs = {}
            num_local = int(data["num_local"])
            
            # Re-extract the L=0 scalar embedding directly (shape: [N, C])
            # emb["node_embedding"] is [N, (lmax+1)^2, C]. Index 0 is exactly L=0.
            scalar_embedding = emb["node_embedding"][:num_local, 0, :]
            
            # Recompute the per-atom energies manually using the head's block
            node_energies = self.original_head.energy_block(scalar_embedding).view(-1)
            # Sum ONLY the local atoms
            local_energy = node_energies.sum().view(1)

            # Apply mean reduction if the underlying model is configured for it
            if getattr(self.original_head, "reduce", "sum") == "mean":
                local_energy = local_energy / num_local

            # Safely overwrite the returned energy scalar in the outputs dictionary
            outputs[energy_key] = local_energy

            if data.pos.requires_grad:
                # Calculate gradient of masked energy w.r.t all positions (local + ghost)
                correct_forces = -1 * torch.autograd.grad(
                    local_energy,
                    data.pos, 
                    # create_graph=self.training, # Retain graph if training
                    # retain_graph=True
                )[0]
                outputs[forces_key] = correct_forces
        else:
            outputs = self.original_head(data, emb)
            outputs[energy_key] = outputs[energy_key]
            if forces_key in outputs:
                outputs[forces_key] = outputs[forces_key]
        return outputs

class UMAModel(DistNNPModel):
    def __init__(
        self,
        model_name: str,
        dataset: str,
        global_rank: int = None,
        inference_settings: InferenceSettings | str = "default", #"turbo",
        overrides: dict | None = None,
        device: str | torch.device | None = None,
        exchanger: Exchanger | None = None,
    ):
        if isinstance(device, torch.device):
            self.device = device.type # make sure this is a str because fairchem needs str (with no ordinal)
        else:
            self.device = device
        self.global_rank = global_rank
        self.dataset = dataset
        inference_settings = guess_inference_settings(inference_settings)
        if overrides is None:
            overrides = {"backbone": {"model": "distributed_escnmd_backbone"}}
        elif "backbone" not in overrides:
            overrides["backbone"] = {"model": "distributed_escnmd_backbone"}
        else:
            overrides["backbone"]["model"] = "distributed_escnmd_backbone"
        self.predictor = pretrained_mlip.get_predict_unit(model_name, inference_settings, overrides, self.device)
        self.main_model = self.predictor.model.module
        self.main_model.output_heads["energyandforcehead"].head = MaskedHeadWrapper(self.main_model.output_heads["energyandforcehead"].head)

        self.num_layers = self.main_model.backbone.num_layers
        self.position_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.atom_type_exchanger = (
            GPUDirectExchanger() if exchanger is None else exchanger
        )
        self.feature_exchangers = [(
            GPUDirectExchanger() if exchanger is None else exchanger
        ) for i in range(self.num_layers + 1)]

    def initialize_buffers(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: Literal["cuda", "cpu"] | None = None,
    ) -> None:
        device = device or self.device
        self.position_exchanger.setup_exchanger(
            feature_dim=3, partition_info=partition_info, device=device
        )
        self.atom_type_exchanger.setup_exchanger(
            feature_dim=1, partition_info=partition_info, device=device
        )

        total_feature_dim = self.main_model.backbone.sph_feature_size * self.main_model.backbone.sphere_channels
        for feature_exchanger in self.feature_exchangers:
            feature_exchanger.setup_exchanger(
                feature_dim=total_feature_dim, partition_info=partition_info, device=device
            )

        self.main_model.backbone._feature_exchangers = self.feature_exchangers

    def predict(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        if not self.predictor.lazy_model_intialized:
            self.predictor._lazy_init(data)

        # Handle single-atom systems (natoms==1 and pbc all False)
        # Skip this check if the model natively supports single atoms
        if not self.predictor.model.module.supports_single_atoms:
            single_atom_result = single_atom_prediction_from_lookup(
                data=data,
                atom_refs=self.predictor.atom_refs,
                tasks=self.predictor.tasks,
                device=self.device,
            )
            if single_atom_result is not None:
                return single_atom_result

        # Regular model prediction path
        # this needs to be .clone() to avoid issues with graph parallel modifying this data with MOLE
        data_device = data.to(self.device).clone()

        dtype = self.predictor.inference_settings.base_precision_dtype
        if not self.predictor._warned_upcast:
            self.predictor._warned_upcast = warn_if_upcasting(data_device.pos.dtype, dtype)
        for key, val in data_device:
            if torch.is_tensor(val) and val.is_floating_point():
                data_device[key] = val.to(dtype)

        # Model handles any per-prediction checks (e.g., MOLE consistency)
        self.predictor.model.module.on_predict_check(data_device)

        return self._run_inference(data_device, undo_element_references)

    def _run_inference(self, data: AtomicData, undo_refs: bool) -> dict:
        """
        Execute model inference.
        """
        inference_context = (
            torch.no_grad()
            if self.predictor.model.module.backbone.regress_config.direct_forces
            else nullcontext()
        )
        tf32_context = (
            tf32_context_manager() if self.predictor.inference_settings.tf32 else nullcontext()
        )

        with inference_context, tf32_context:
            output = self.predictor.model(data)
            return self._process_outputs(data, output, undo_refs)

    def _process_outputs(self, data: AtomicData, output: dict, undo_refs: bool) -> dict:
        """
        Denormalize and post-process model outputs.
        """
        pred_output = {}
        for task_name, task in self.predictor.model.module.tasks.items():
            if task_name not in output:
                continue

            if self.global_rank == 0:
                pred_output[task_name] = task.normalizer.denorm(
                    output[task_name][task.property]
                )
            else:
                pred_output[task_name] = task.normalizer.rmsd * output[task_name][task.property]
            if self.predictor.assert_on_nans:
                assert (
                    torch.isfinite(pred_output[task_name]).all()
                ), f"NaNs/Infs found in prediction for task {task_name}.{task.property}"
            if undo_refs and task.element_references is not None:
                pred_output[task_name] = task.element_references.undo_refs(
                    data, pred_output[task_name]
                )
                ghost_z = data["atomic_numbers_full"][data["num_local"]:]
                ghost_refs_energy = task.element_references.element_references[ghost_z].sum()
                pred_output[task_name] = pred_output[task_name] - ghost_refs_energy
        return pred_output

    def __call__(
        self,
        distributed_state: DistNNPState,
        charge: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        device: Literal["cuda", "cpu"] | None = None,
    ):
        self.main_model.eval()
        device = device or self.device
        self.main_model.backbone._distributed_state = distributed_state

        local_mol_info = distributed_state.local_molecular_info
        global_mol_info = distributed_state.global_molecular_info
        sim_config = distributed_state.sim_config
        if self.global_rank is None:
            self.global_rank = distributed_state.dist_env.global_rank

        positions = FeaturePushToExchange.apply(
            local_mol_info.atom_positions, distributed_state, self.position_exchanger
        )

        atom_types = FeaturePushToExchange.apply(
            local_mol_info.atom_types.float().unsqueeze(-1), distributed_state, self.atom_type_exchanger
        ).squeeze(-1).long()

        num_global = global_mol_info.num_atoms
        num_atoms = positions.shape[0]
        num_local = local_mol_info.num_atoms
        pbc = sim_config.pbc.unsqueeze(0)
        cell = sim_config.cell.unsqueeze(0)
        with torch.no_grad():
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                data=type(
                    "Data",
                    (object,),
                    {
                        "pos": positions,
                        "cell": cell,  # [1, 3, 3]
                        "natoms": [num_atoms],
                        "pbc": pbc,
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        batch = torch.zeros(num_local, dtype=torch.long, device=device)
        datasets = self.dataset
        num_systems = int(batch.max().item()) + 1 
        if charge is None:
            charge = torch.zeros(num_systems, dtype=torch.long, device=device)
        if spin is None:
            spin = torch.ones(num_systems, dtype=torch.long, device=device)

        data = AtomicData(
            positions,
            atom_types,
            cell,
            pbc,
            torch.tensor([num_atoms], dtype=torch.long, device=device),
            edge_index,
            -cell_offsets,
            torch.tensor([edge_index.shape[1]], dtype=torch.long, device=device),
            charge,
            spin,
            torch.zeros(num_atoms, dtype=torch.long, device=device), # fixed
            torch.zeros(num_atoms, dtype=torch.long, device=device), # tags
            # batch=batch_full,
            dataset=datasets,
        )
        data["num_local"] = num_local
        data["num_global"] = torch.tensor([num_global], device=device)
        data.to(device)

        self.predictor.move_to_device()
        pred = self.predict(data)

        energy = pred[f"{datasets}_energy"] * _EV_PER_A_TO_AMU_A_PER_FS2
        forces = pred[f"{datasets}_forces"] * _EV_PER_A_TO_AMU_A_PER_FS2

        dummy_local = torch.zeros((num_local, 3), device=device, requires_grad=True)
        dummy_full = FeaturePushToExchange.apply(
            dummy_local, distributed_state, self.position_exchanger
        )
        dummy_full.backward(forces)
        local_forces = dummy_local.grad
        return energy.detach(), local_forces.detach()
        return energy.detach(), forces.detach()
