import torch
import torch.distributed as dist
import numpy as np
import os
from torch.profiler import record_function
from typing import Dict, List, Tuple


from baseline.local_fairchem.models.uma.escn_moe import eSCNMDMoeBackbone
from baseline.local_fairchem.common.registry import registry
from baseline.local_fairchem.models.uma.nn.mole_utils import (
    model_search_and_replace,
    recursive_replace_so2_MOLE,
    replace_MOLE_with_linear,
)

class FeatureExchanger:
    """
    Handles the exchange of node features between ranks.
    """
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        # Maps: rank -> tensor of indices
        self.send_map: Dict[int, torch.Tensor] = {} 
        self.recv_map: Dict[int, torch.Tensor] = {}
        self.recv_ghost_indices: Dict[int, torch.Tensor] = {} # Indices in local x_message to update
        self.profiler = None  # Optional profiler for detailed timing

    def update_maps(self, send_map: Dict[int, torch.Tensor], recv_map: Dict[int, torch.Tensor], recv_ghost_indices: Dict[int, torch.Tensor]):
        """
        Update communication maps.
        send_map: rank -> indices of local atoms to send
        recv_map: rank -> indices of remote atoms to receive (not used directly for indexing, but for sizing)
        recv_ghost_indices: rank -> indices in local tensor where received data should go
        """
        self.send_map = send_map
        self.recv_map = recv_map
        self.recv_ghost_indices = recv_ghost_indices

    def exchange(self, x, num_local):
        """
        Exchange features with neighbors.
        x: (num_local + num_ghosts, ...) - input features (ghosts might be stale)
        num_local: int - number of local atoms
        Returns: (num_local + num_ghosts, ...) - updated features with synced ghosts
        """
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return x
            
        # Handle arbitrary shapes (e.g. 3D tensors for spherical harmonics)
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
            
        # 1. Prepare send buffers
        # We need to gather features for each neighbor
        # self.send_map: {rank: indices_tensor}
        
        send_data = []
        send_splits = []
        
        for r in range(self.world_size):
            if r in self.send_map:
                indices = self.send_map[r]
                # indices are on device, x is on device
                feat = x_flat[indices]
                send_data.append(feat)
                send_splits.append(feat.size(0))
            else:
                # Empty
                send_data.append(torch.empty((0, x_flat.size(1)), device=x.device, dtype=x.dtype))
                send_splits.append(0)
                
        send_tensor = torch.cat(send_data, dim=0)
        
        # 2. Prepare recv buffer
        recv_splits = [0] * self.world_size
        for r, indices in self.recv_ghost_indices.items():
            recv_splits[r] = indices.size(0)
            
        total_recv = sum(recv_splits)
        recv_tensor = torch.zeros((total_recv, x_flat.size(1)), device=x.device, dtype=x.dtype)
        
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()

        # 3. Exchange using Autograd Function
        recv_tensor = FeatureExchangeFunction.apply(send_tensor, recv_splits, send_splits)
        
        # 4. Scatter received features to ghost positions
        # OPTIMIZATION: Use batched scatter with pre-computed concatenated indices
        # This avoids x.clone() which is expensive for large feature tensors
        
        # Build concatenated indices for batch scatter (cache these if possible)
        if not hasattr(self, '_batched_recv_indices') or self._batched_recv_version != id(self.recv_ghost_indices):
            all_indices = []
            for r in range(self.world_size):
                if r in self.recv_ghost_indices:
                    all_indices.append(self.recv_ghost_indices[r])
            if all_indices:
                self._batched_recv_indices = torch.cat(all_indices)
            else:
                self._batched_recv_indices = torch.empty(0, dtype=torch.long, device=x.device)
            self._batched_recv_version = id(self.recv_ghost_indices)
        
        # Instead of clone(), modify x in-place for ghost positions only
        # This is safe because we're only updating ghost slots which are stale anyway
        x_with_ghosts = x_flat  # No clone! We update in-place
        
        if self._batched_recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
            # Use index_copy_ for batched in-place update
            x_with_ghosts.index_copy_(0, self._batched_recv_indices, recv_tensor)
        
        # Reshape back
        if x.dim() > 2:
            x_with_ghosts = x_with_ghosts.reshape(original_shape[0], *original_shape[1:])
        
        return x_with_ghosts

    def async_exchange_start(self, x, num_local):
        """
        Start non-blocking exchange. Returns a handle to wait on.
        
        Args:
            x: (num_local + num_ghosts, F) - input features
            num_local: int - number of local atoms
        
        Returns:
            handle: tuple (work_handle, recv_tensor, recv_splits, original_shape)
        """
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return None
        
        # Handle arbitrary shapes (e.g. 3D tensors for spherical harmonics)
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
        
        # 1. Prepare send buffers (gather local features to send)
        if self.profiler:
            self.profiler.start("exchange_prepare_send")
        send_data = []
        send_splits = []
        
        for r in range(self.world_size):
            if r in self.send_map:
                indices = self.send_map[r]
                feat = x_flat[indices]
                send_data.append(feat)
                send_splits.append(feat.size(0))
            else:
                send_data.append(torch.empty((0, x_flat.size(1)), device=x.device, dtype=x.dtype))
                send_splits.append(0)
                
        send_tensor = torch.cat(send_data, dim=0)
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
        if self.profiler:
            self.profiler.stop("exchange_prepare_send")
        
        # 2. Prepare recv buffer
        if self.profiler:
            self.profiler.start("exchange_prepare_recv")
        recv_splits = [0] * self.world_size
        for r, indices in self.recv_ghost_indices.items():
            recv_splits[r] = indices.size(0)
            
        total_recv = sum(recv_splits)
        recv_tensor = torch.empty((total_recv, x_flat.size(1)), device=x.device, dtype=x.dtype)
        if self.profiler:
            self.profiler.stop("exchange_prepare_recv")
        
        # 3. Start async all-to-all
        if self.profiler:
            self.profiler.start("exchange_async_a2a")
        work = dist.all_to_all_single(
            recv_tensor, 
            send_tensor, 
            output_split_sizes=recv_splits, 
            input_split_sizes=send_splits,
            async_op=True
        )
        if self.profiler:
            self.profiler.stop("exchange_async_a2a")
        
        return (work, recv_tensor, recv_splits, send_splits, original_shape, x_flat.size(1))

    def async_exchange_wait(self, handle, x):
        """
        Wait for async exchange to complete and update ghost features.
        
        Args:
            handle: tuple from async_exchange_start
            x: original tensor to update in-place
        
        Returns:
            x with ghost features updated
        """
        if handle is None:
            return x
        
        work, recv_tensor, recv_splits, send_splits, original_shape, feature_dim = handle
        
        # Wait for communication to complete
        work.wait()
        
        # Handle shape
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
        
        # Build concatenated indices for batch scatter (cache these if possible)
        if not hasattr(self, '_batched_recv_indices') or self._batched_recv_version != id(self.recv_ghost_indices):
            all_indices = []
            for r in range(self.world_size):
                if r in self.recv_ghost_indices:
                    all_indices.append(self.recv_ghost_indices[r])
            if all_indices:
                self._batched_recv_indices = torch.cat(all_indices)
            else:
                self._batched_recv_indices = torch.empty(0, dtype=torch.long, device=x.device)
            self._batched_recv_version = id(self.recv_ghost_indices)
        
        # Scatter received features to ghost positions
        if self._batched_recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
            x_flat.index_copy_(0, self._batched_recv_indices, recv_tensor)
        
        # Reshape back if needed
        if x.dim() > 2:
            return x_flat.reshape(original_shape[0], *original_shape[1:])
        
        return x_flat

class FeatureExchangeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, send_tensor, recv_splits, send_splits):
        ctx.recv_splits = recv_splits
        ctx.send_splits = send_splits
        
        # Allocate output buffer
        total_recv = sum(recv_splits)
        # Handle arbitrary dimensions by flattening in caller, 
        # but here we expect [total_atoms, feature_dim]
        # In caller (exchange), we flatten to 2D.
        
        recv_tensor = torch.empty(
            (total_recv, send_tensor.size(1)), 
            dtype=send_tensor.dtype, 
            device=send_tensor.device
        )
        
        # Exchange
        # Ensure send_tensor is contiguous
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
            
        dist.all_to_all_single(
            recv_tensor, 
            send_tensor, 
            output_split_sizes=recv_splits, 
            input_split_sizes=send_splits
        )
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        recv_splits = ctx.recv_splits
        send_splits = ctx.send_splits
        
        # Enable Gradient Exchange
        # We need to send the gradients on ghost nodes (grad_output) back to their owners.
        # This is the reverse of forward: 
        #   forward: send_tensor (Local) -> recv_tensor (Ghost)
        #   backward: grad_output (Ghost) -> grad_input (Local)
        
        # Ensure contiguous
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        total_send = sum(send_splits)
        grad_input = torch.empty(
            (total_send, grad_output.size(1)), 
            dtype=grad_output.dtype, 
            device=grad_output.device
        )
        
        if dist.is_initialized():
             dist.all_to_all_single(
                grad_input,
                grad_output,
                output_split_sizes=send_splits,
                input_split_sizes=recv_splits
            )
        else:
            # Fallback for single process debugging (though usually world_size check handles this)
             pass

        return grad_input, None, None


@registry.register_model("distributed_escnmd_backbone")
class DistributedESCNMDBackbone(eSCNMDMoeBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize exchanger with dummy values, will be updated later
        self.exchanger = FeatureExchanger(rank=0, world_size=1) 
        self.num_local = 0
        self.profiler = None  # CUDAEventProfiler for detailed timing
    
    def set_profiler(self, profiler):
        """Set profiler for detailed model timing."""
        self.profiler = profiler
        # Also pass to exchanger for communication timing
        self.exchanger.profiler = profiler

    def set_comm_maps(self, send_map, recv_ghost_indices, rank, world_size, interior_mask=None, enable_overlap=False):
        self.exchanger.rank = rank
        self.exchanger.world_size = world_size
        self.exchanger.update_maps(send_map, {}, recv_ghost_indices)
        # Store interior mask and overlap flag for optimization
        self.interior_mask = interior_mask
        self.enable_overlap = enable_overlap

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        if self.num_experts == 0:
            return
        
        # OPTIMIZATION: Cache MOLE coefficients to avoid per-step all_reduce
        # In MD simulations with merge_mole=True, composition is static.
        if not hasattr(self, '_cached_mole_coefficients'):
            self._cached_mole_coefficients = None
            self._cached_mole_num_atoms = -1
        
        # Check if we can use cache (same number of atoms = same composition)
        num_atoms = atomic_numbers_full.shape[0]
        if self._cached_mole_coefficients is not None and self._cached_mole_num_atoms == num_atoms:
            self.global_mole_tensors.expert_mixing_coefficients = self._cached_mole_coefficients
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
                    # Use num_local to identify unique atoms
                    num_local = self.num_local
                    if num_local is None:
                        num_local = atomic_numbers_full.shape[0] # Fallback if not set
                    
                    if num_local > atomic_numbers_full.shape[0]:
                         num_local = atomic_numbers_full.shape[0]

                    local_an = atomic_numbers_full[:num_local]
                    local_comp = self.composition_embedding(local_an)
                    local_sum = local_comp.sum(0)
                    local_count = torch.tensor([num_local], device=device, dtype=local_sum.dtype)
                    
                    # Handle CPU tensors with NCCL backend (which doesn't support CPU all_reduce)
                    is_cpu = local_sum.device.type == "cpu"
                    if is_cpu and dist.get_backend() == "nccl":
                        # Move to current cuda device
                        # Assuming torch.cuda.set_device has been called appropriately by launcher/script
                        # We use .cuda() which maps to current device
                        local_sum_cuda = local_sum.cuda()
                        local_count_cuda = local_count.cuda()
                        
                        dist.all_reduce(local_sum_cuda)
                        dist.all_reduce(local_count_cuda)
                        
                        local_sum = local_sum_cuda.cpu()
                        local_count = local_count_cuda.cpu()
                    else:
                        dist.all_reduce(local_sum)
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
            self._cached_mole_num_atoms = num_atoms
            
            self.global_mole_tensors.expert_mixing_coefficients = coefficients

    def forward(self, data_dict):
        # We need to inject self.exchanger.exchange(x_message) before each block.
        
        # Pre-computation (same as original)
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )

        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )

        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = self._get_displacement_and_cell(data_dict)

        # Profile: generate_graph
        if self.profiler:
            self.profiler.start("generate_graph")
        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)
        if self.profiler:
            self.profiler.stop("generate_graph")
            
        # DISTNNP OPTIMIZATION: Filter redundant edges early
        # This logic was originally in eSCNMD.forward but is REPLICATED here because
        # DistributedESCNMDBackbone overrides forward() without calling super().
        
        # 1. Extract num_local
        num_local = None
        if "num_local" in data_dict:
             num_local = data_dict["num_local"]
             # If it's a tensor, get item
             if isinstance(num_local, torch.Tensor):
                 if num_local.numel() == 1:
                     num_local = num_local.item()
                 else:
                     pass # Batched? default to None for now or handle
        
        # 2. Keep reference to original edges for MOLE stats
        original_edge_index = graph_dict["edge_index"]
        
        # 3. Filter if num_local is valid
        if num_local is not None and isinstance(num_local, int) and num_local < data_dict["pos"].shape[0]:
            # Update self.num_local for later usage in forward loop if needed
            self.num_local = num_local
            
            pre_edges = graph_dict["edge_index"].shape[1]
            # Filter edges: only keep edges where target < num_local
            mask = graph_dict["edge_index"][1] < num_local
            
            # Apply filter to all edge tensors
            graph_dict["edge_index"] = graph_dict["edge_index"][:, mask]
            
            if "edge_distance" in graph_dict:
                 graph_dict["edge_distance"] = graph_dict["edge_distance"][mask]
                 
            if "edge_distance_vec" in graph_dict:
                graph_dict["edge_distance_vec"] = graph_dict["edge_distance_vec"][mask]

            if "cell_offset" in graph_dict:
                graph_dict["cell_offset"] = graph_dict["cell_offset"][mask]

        if graph_dict["edge_index"].numel() == 0:
             raise ValueError(
                f"No edges found. Check cutoff {self.cutoff}."
            )

        # Profile: wigner matrix computation
        if self.profiler:
            self.profiler.start("wigner")
        with record_function("obtain wigner"):
            (wigner_and_M_mapping, wigner_and_M_mapping_inv) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec"],
                )
            )
        if self.profiler:
            self.profiler.stop("wigner")

        # Initialize node embeddings
        # Profile: atom_embedding
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

        # edge degree embedding
        # Profile: edge_embedding
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
            
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_and_M_mapping_inv,
                edge_envelope,
                graph_dict["node_offset"],
            )
        if self.profiler:
            self.profiler.stop("edge_embedding")

        # Update spherical node embeddings (Message Passing Layers)
        # Prepare local_edge_index for efficient message passing
        # We only need to aggregate messages INTO local atoms.
        # Edges where destination (edge_index[1]) is a ghost atom can be skipped.
        # This avoids redundant computation for ghost atoms since their features
        # are overwritten by exchange() in the next step/layer.
        
        # Handle single-rank case: if num_local is 0 or not set, use all atoms
        effective_num_local = self.num_local if self.num_local > 0 else data_dict["pos"].shape[0]
        
        local_dest_mask = graph_dict["edge_index"][1] < effective_num_local
        local_edge_index = graph_dict["edge_index"][:, local_dest_mask]
        
        # We also need to filter edge attributes accordingly
        local_edge_distance = graph_dict["edge_distance"][local_dest_mask]
        
        # === OVERLAP OPTIMIZATION ===
        # Check config flag first to avoid any overhead when disabled
        use_overlap = getattr(self, 'enable_overlap', False)
        
        if use_overlap:
            # Only compute overlap-related tensors when enabled
            _could_use_overlap = (hasattr(self, 'interior_mask') and 
                                  self.interior_mask is not None and 
                                  self.interior_mask.numel() > 0 and
                                  self.exchanger.world_size > 1)
            
            if not _could_use_overlap:
                use_overlap = False
            else:
                # Filter x_edge for overlap path
                local_x_edge = x_edge[local_dest_mask]
                
                # Build masks for interior edges (dst is interior atom) 
                # and boundary edges (dst is boundary atom)
                interior_atom_mask = self.interior_mask.to(local_edge_index.device)
                
                # Edge goes to interior atom if dst is marked as interior
                interior_edge_mask = interior_atom_mask[local_edge_index[1]]
                boundary_edge_mask = ~interior_edge_mask
                
                # Pre-filter edges for interior and boundary
                interior_edge_index = local_edge_index[:, interior_edge_mask]
                boundary_edge_index = local_edge_index[:, boundary_edge_mask]
                
                interior_edge_distance = local_edge_distance[interior_edge_mask]
                boundary_edge_distance = local_edge_distance[boundary_edge_mask]
                
                # Filter x_edge for interior and boundary
                interior_x_edge = local_x_edge[interior_edge_mask]
                boundary_x_edge = local_x_edge[boundary_edge_mask]
                
                # Log overlap stats once
                if not hasattr(self, '_logged_overlap_stats'):
                    interior_count = interior_atom_mask.sum().item()
                    total_local = self.num_local
                    interior_edges = interior_edge_index.shape[1]
                    boundary_edges = boundary_edge_index.shape[1]
                    if self.exchanger.rank == 0:
                        print(f"Overlap enabled: {interior_count}/{total_local} interior atoms ({100*interior_count/total_local:.1f}%)")
                        print(f"  Interior edges: {interior_edges}, Boundary edges: {boundary_edges}")
                    self._logged_overlap_stats = True
                
                # Prepare Wigner filtering function (only needed for overlap)
                def filter_wigner(wigner_dict, mask, full_mask):
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
        
        # Update spherical node embeddings (Message Passing Layers)
        for i in range(self.num_layers):
            if use_overlap:
                # === OVERLAPPED EXECUTION ===
                with record_function(f"async_exchange_start {i}"):
                    # Start async exchange (non-blocking)
                    exchange_handle = self.exchanger.async_exchange_start(x_message, self.num_local)
                
                with record_function(f"interior_compute {i}"):
                    # Compute interior atoms while exchange is in progress
                    # Interior atoms only have local neighbors, so they don't need
                    # the exchanged ghost features for this layer
                    if interior_edge_index.shape[1] > 0:
                        # Filter Wigner matrices for interior edges
                        interior_wigner = filter_wigner(wigner_and_M_mapping, interior_edge_mask, local_dest_mask)
                        interior_wigner_inv = filter_wigner(wigner_and_M_mapping_inv, interior_edge_mask, local_dest_mask)
                        interior_envelope = edge_envelope[local_dest_mask][interior_edge_mask] if edge_envelope is not None else None
                        
                        # Run block on interior edges only
                        # Save x_message state before block modifies it
                        x_interior_result = self.blocks[i](
                            x_message.clone(),  # Clone to avoid in-place modification issues
                            interior_x_edge,    # Use filtered x_edge!
                            interior_edge_distance,
                            interior_edge_index,
                            interior_wigner,
                            interior_wigner_inv,
                            interior_envelope,
                            total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[0],
                            sys_node_embedding=sys_node_embedding,
                            node_offset=graph_dict["node_offset"],
                        )
                
                with record_function(f"async_exchange_wait {i}"):
                    # Wait for exchange to complete
                    x_message = self.exchanger.async_exchange_wait(exchange_handle, x_message)
                
                with record_function(f"boundary_compute {i}"):
                    # Now compute boundary atoms (they need the exchanged ghost features)
                    if boundary_edge_index.shape[1] > 0:
                        # Filter Wigner matrices for boundary edges
                        boundary_wigner = filter_wigner(wigner_and_M_mapping, boundary_edge_mask, local_dest_mask)
                        boundary_wigner_inv = filter_wigner(wigner_and_M_mapping_inv, boundary_edge_mask, local_dest_mask)
                        boundary_envelope = edge_envelope[local_dest_mask][boundary_edge_mask] if edge_envelope is not None else None
                        
                        # Run block on boundary edges only
                        x_boundary_result = self.blocks[i](
                            x_message,
                            boundary_x_edge,    # Use filtered x_edge!
                            boundary_edge_distance,
                            boundary_edge_index,
                            boundary_wigner,
                            boundary_wigner_inv,
                            boundary_envelope,
                            total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[0],
                            sys_node_embedding=sys_node_embedding,
                            node_offset=graph_dict["node_offset"],
                        )
                        # Merge boundary results
                        # The boundary block only updated boundary atom indices
                        # We need to copy the interior results for interior atoms
                        x_message = x_boundary_result
                    else:
                        x_message = x_interior_result if interior_edge_index.shape[1] > 0 else x_message
                    
                    # Copy interior atom results from x_interior_result
                    if interior_edge_index.shape[1] > 0:
                        x_message[:self.num_local][interior_atom_mask] = x_interior_result[:self.num_local][interior_atom_mask]
                
            else:
                # === OPTIMIZED PATH: Overlap exchange with Wigner filtering ===
                # Start async exchange first
                # Profile: feature_exchange
                # dist.barrier()
                if self.profiler:
                    self.profiler.start(f"feature_exchange_{i}")
                with record_function(f"async_exchange_start {i}"):
                    exchange_handle = self.exchanger.async_exchange_start(x_message, self.num_local)
                
                # DEBUG:Wait for exchange to complete
                # with record_function(f"async_exchange_wait {i}"):
                #     x_message = self.exchanger.async_exchange_wait(exchange_handle, x_message)
                # if self.profiler:
                #     self.profiler.stop(f"feature_exchange_{i}")
                
                # Do Wigner filtering ONLY on first iteration (results don't change)
                # This overlaps with async exchange on first iteration
                if i == 0:
                    with record_function("wigner_filter"):
                        local_wigner_and_M_mapping = wigner_and_M_mapping
                        if isinstance(wigner_and_M_mapping, dict):
                            local_wigner_and_M_mapping = {}
                            for k, v in wigner_and_M_mapping.items():
                                if isinstance(v, torch.Tensor) and v.shape[0] == graph_dict["edge_index"].shape[1]:
                                     local_wigner_and_M_mapping[k] = v[local_dest_mask]
                                else:
                                     local_wigner_and_M_mapping[k] = v
                        elif isinstance(wigner_and_M_mapping, torch.Tensor):
                             local_wigner_and_M_mapping = wigner_and_M_mapping[local_dest_mask]

                                 
                        local_wigner_and_M_mapping_inv = wigner_and_M_mapping_inv
                        if wigner_and_M_mapping_inv is not None:
                             if isinstance(wigner_and_M_mapping_inv, dict):
                                 local_wigner_and_M_mapping_inv = {}
                                 for k, v in wigner_and_M_mapping_inv.items():
                                    if isinstance(v, torch.Tensor) and v.shape[0] == graph_dict["edge_index"].shape[1]:
                                         local_wigner_and_M_mapping_inv[k] = v[local_dest_mask]
                                    else:
                                         local_wigner_and_M_mapping_inv[k] = v
                             elif isinstance(wigner_and_M_mapping_inv, torch.Tensor):
                                 local_wigner_and_M_mapping_inv = wigner_and_M_mapping_inv[local_dest_mask]

                        local_edge_envelope = edge_envelope[local_dest_mask] if edge_envelope is not None else None
                # else: reuse local_wigner_*, local_edge_envelope from first iteration
                
                # Wait for exchange to complete
                with record_function(f"async_exchange_wait {i}"):
                    x_message = self.exchanger.async_exchange_wait(exchange_handle, x_message)
                if self.profiler:
                    self.profiler.stop(f"feature_exchange_{i}")

                # Now run the block with updated ghost features
                # Profile: mp_block
                if self.profiler:
                    self.profiler.start(f"mp_block_{i}")
                with record_function(f"message passing {i}"):
                    x_message = self.blocks[i](
                        x_message,
                        x_edge,
                        local_edge_distance,
                        local_edge_index,
                        local_wigner_and_M_mapping,
                        local_wigner_and_M_mapping_inv,
                        local_edge_envelope,
                        total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[0],
                        sys_node_embedding=sys_node_embedding,
                        node_offset=graph_dict["node_offset"],
                    )
                if self.profiler:
                    self.profiler.stop(f"mp_block_{i}")

        # Final layer norm
        x_message = self.norm(x_message)
        
        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
            "batch": data_dict["batch"],
        }
        return out

    def merge_MOLE_model(self, data):
        if self.num_experts == 0:
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
        new_model.exchanger = self.exchanger
        if hasattr(self, "num_local"):
             new_model.num_local = self.num_local
        
        new_model.eval()
        return new_model
