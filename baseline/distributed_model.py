import torch
import torch.distributed as dist
import numpy as np
import os
from torch.profiler import record_function
from typing import Dict, List, Tuple
from baseline.nccl_debug import log_nccl_call


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
        
        # === Area 1: Pre-compute cached indices for O(1) gather/scatter ===
        # Concatenate all send indices into a single tensor + compute split sizes
        send_indices_list = []
        send_splits = []
        for r in range(self.world_size):
            if r in send_map and send_map[r].numel() > 0:
                send_indices_list.append(send_map[r])
                send_splits.append(send_map[r].size(0))
            else:
                send_splits.append(0)
        if send_indices_list:
            self._cached_send_indices = torch.cat(send_indices_list)
        else:
            self._cached_send_indices = None
        self._cached_send_splits = send_splits
        
        # Pre-compute recv splits and batched recv indices
        recv_splits = [0] * self.world_size
        recv_indices_list = []
        for r in range(self.world_size):
            if r in recv_ghost_indices and recv_ghost_indices[r].numel() > 0:
                recv_splits[r] = recv_ghost_indices[r].size(0)
                recv_indices_list.append(recv_ghost_indices[r])
        self._cached_recv_splits = recv_splits
        if recv_indices_list:
            self._batched_recv_indices = torch.cat(recv_indices_list)
        else:
            self._batched_recv_indices = torch.empty(0, dtype=torch.long)
        self._batched_recv_version = id(recv_ghost_indices)

    def exchange(self, x, num_local):
        """
        Exchange features with neighbors.
        x: (num_local + num_ghosts, ...) - input features (ghosts might be stale)
        num_local: int - number of local atoms
        Returns: (num_local + num_ghosts, ...) - updated features with synced ghosts
        """
        # CRITICAL: Only check world_size, NOT recv_ghost_indices!
        # all_to_all_single is a collective - ALL ranks must participate.
        # Even if a rank has no ghost indices, it must join with zero-sized buffers.
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return x
            
        # Handle arbitrary shapes (e.g. 3D tensors for spherical harmonics)
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
            
        # 1. Prepare send buffer using pre-computed cached indices (Area 1)
        if self._cached_send_indices is not None:
            send_tensor = x_flat[self._cached_send_indices]
        else:
            send_tensor = torch.empty((0, x_flat.size(1)), device=x.device, dtype=x.dtype)
        
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()

        # 2. Exchange using Autograd Function with pre-computed splits
        recv_tensor = FeatureExchangeFunction.apply(send_tensor, self._cached_recv_splits, self._cached_send_splits)
        
        # 3. Scatter received features to ghost positions using pre-computed indices
        x_with_ghosts = x_flat  # No clone - update ghost slots in-place
        
        if self._batched_recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
            recv_idx = self._batched_recv_indices
            if recv_idx.device != x.device:
                recv_idx = recv_idx.to(x.device)
            x_with_ghosts.index_copy_(0, recv_idx, recv_tensor)
        
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
        # CRITICAL: Only check world_size, NOT recv_ghost_indices!
        # all_to_all_single is a collective - ALL ranks must participate.
        # Even if a rank has no ghost indices, it must join with zero-sized buffers.
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return None
        
        # Handle arbitrary shapes (e.g. 3D tensors for spherical harmonics)
        original_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
        
        # 1. Prepare send buffer using pre-computed cached indices (Area 1)
        if self.profiler:
            self.profiler.start("exchange_prepare_send")
        if self._cached_send_indices is not None:
            send_indices = self._cached_send_indices
            if send_indices.device != x.device:
                send_indices = send_indices.to(x.device)
                self._cached_send_indices = send_indices
            send_tensor = x_flat[send_indices]
        else:
            send_tensor = torch.empty((0, x_flat.size(1)), device=x.device, dtype=x.dtype)
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
        if self.profiler:
            self.profiler.stop("exchange_prepare_send")
        
        # 2. Prepare recv buffer using pre-computed splits
        if self.profiler:
            self.profiler.start("exchange_prepare_recv")
        total_recv = sum(self._cached_recv_splits)
        recv_tensor = torch.empty((total_recv, x_flat.size(1)), device=x.device, dtype=x.dtype)
        if self.profiler:
            self.profiler.stop("exchange_prepare_recv")
        
        # 3. Start async all-to-all
        if self.profiler:
            self.profiler.start("exchange_async_a2a")
        log_nccl_call(__file__, 188, "all_to_all_single_async")
        work = dist.all_to_all_single(
            recv_tensor, 
            send_tensor, 
            output_split_sizes=self._cached_recv_splits, 
            input_split_sizes=self._cached_send_splits,
            async_op=True
        )
        if self.profiler:
            self.profiler.stop("exchange_async_a2a")
        
        return (work, recv_tensor, self._cached_recv_splits, self._cached_send_splits, original_shape, x_flat.size(1))

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
        
        # Scatter received features to ghost positions using pre-computed indices
        if self._batched_recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
            recv_idx = self._batched_recv_indices
            if recv_idx.device != x.device:
                recv_idx = recv_idx.to(x.device)
                self._batched_recv_indices = recv_idx
            x_flat.index_copy_(0, recv_idx, recv_tensor)
        
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
        log_nccl_call(__file__, 270, "all_to_all_single_sync")
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
            log_nccl_call(__file__, 301, "all_to_all_single_backward")
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


class HierarchicalFeatureExchanger:
    """
    Hierarchical Feature Exchanger (Area 2 / PXN Aggregation).
    
    Reduces NIC contention on multi-GPU nodes sharing a single NIC by using
    a 3-phase communication pattern:
      Phase 1: Intra-node gather (NVLink) — all local ranks send to master via a2a
      Phase 2: Inter-node all_to_all (Network) — only master ranks participate
      Phase 3: Intra-node scatter (NVLink) — master distributes to local ranks via a2a
    
    Falls back to standard FeatureExchanger when world_size fits on one node.
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.send_map: Dict[int, torch.Tensor] = {}
        self.recv_map: Dict[int, torch.Tensor] = {}
        self.recv_ghost_indices: Dict[int, torch.Tensor] = {}
        self.profiler = None
        
        # Hierarchical group info (set via set_groups)
        self.intra_group = None
        self.inter_group = None   # Only meaningful for master ranks
        self.local_rank = 0
        self.local_world_size = 1
        self.is_master = False
        self.num_nodes = 1
        self.node_id = 0
        
        # Standard cached indices (same as FeatureExchanger)
        self._cached_send_indices = None
        self._cached_send_splits: List[int] = []
        self._cached_recv_splits: List[int] = []
        self._batched_recv_indices = torch.empty(0, dtype=torch.long)
    
    def set_groups(self, intra_group, inter_group, local_rank: int, local_world_size: int):
        """Set process groups for hierarchical communication."""
        self.intra_group = intra_group
        self.inter_group = inter_group
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.is_master = (local_rank == local_world_size - 1)
        self.num_nodes = self.world_size // local_world_size
        self.node_id = self.rank // local_world_size
    
    def update_maps(self, send_map: Dict[int, torch.Tensor], recv_map: Dict[int, torch.Tensor], 
                    recv_ghost_indices: Dict[int, torch.Tensor]):
        """Update communication maps — same interface as FeatureExchanger."""
        self.send_map = send_map
        self.recv_map = recv_map
        self.recv_ghost_indices = recv_ghost_indices
        
        # === Pre-compute cached indices (same as FeatureExchanger) ===
        send_indices_list = []
        send_splits = []
        for r in range(self.world_size):
            if r in send_map and send_map[r].numel() > 0:
                send_indices_list.append(send_map[r])
                send_splits.append(send_map[r].size(0))
            else:
                send_splits.append(0)
        if send_indices_list:
            self._cached_send_indices = torch.cat(send_indices_list)
        else:
            self._cached_send_indices = None
        self._cached_send_splits = send_splits
        
        recv_splits = [0] * self.world_size
        recv_indices_list = []
        for r in range(self.world_size):
            if r in recv_ghost_indices and recv_ghost_indices[r].numel() > 0:
                recv_splits[r] = recv_ghost_indices[r].size(0)
                recv_indices_list.append(recv_ghost_indices[r])
        self._cached_recv_splits = recv_splits
        if recv_indices_list:
            self._batched_recv_indices = torch.cat(recv_indices_list)
        else:
            self._batched_recv_indices = torch.empty(0, dtype=torch.long)
    
    def exchange(self, x, num_local):
        """Synchronous hierarchical exchange — delegates to async start + wait."""
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return x
        handle = self.async_exchange_start(x, num_local)
        return self.async_exchange_wait(handle, x)
    
    def async_exchange_start(self, x, num_local):
        """
        Start hierarchical exchange. Returns handle for async_exchange_wait.
        
        Algorithm:
          Phase 1: Each rank sends its send_tensor to master via intra-node a2a.
          Phase 2: Master re-packs by destination node and does inter-node a2a (async).
        """
        if self.world_size <= 1 or not self.recv_ghost_indices:
            return None
        
        original_shape = x.shape
        x_flat = x.reshape(original_shape[0], -1) if x.dim() > 2 else x
        feature_dim = x_flat.size(1)
        lws = self.local_world_size
        
        # 1. Prepare send buffer using pre-computed indices
        if self.profiler:
            self.profiler.start("exchange_prepare_send")
        if self._cached_send_indices is not None:
            send_indices = self._cached_send_indices
            if send_indices.device != x.device:
                send_indices = send_indices.to(x.device)
                self._cached_send_indices = send_indices
            send_tensor = x_flat[send_indices].contiguous()
        else:
            send_tensor = torch.empty((0, feature_dim), device=x.device, dtype=x.dtype)
        if self.profiler:
            self.profiler.stop("exchange_prepare_send")
        
        if self.profiler:
            self.profiler.start("exchange_async_a2a")
        
        # === PHASE 1: Intra-node gather to master ===
        # Use intra-node all_to_all_single: each rank sends all its data to master,
        # non-masters send nothing to each other.
        # send_splits for intra a2a: [0, 0, ..., my_count] (only send to master=lws-1)
        my_send_count = send_tensor.shape[0]
        intra_gather_send_splits = [0] * lws
        intra_gather_send_splits[lws - 1] = my_send_count  # send everything to master
        
        # Exchange counts so master knows how much to receive from each local rank
        send_count_t = torch.tensor([my_send_count], dtype=torch.long, device=x.device)
        all_counts = [torch.zeros(1, dtype=torch.long, device=x.device) for _ in range(lws)]
        log_nccl_call(__file__, 0, "hier_p1_allgather_counts")
        dist.all_gather(all_counts, send_count_t, group=self.intra_group)
        local_counts = [c.item() for c in all_counts]
        
        # Master receives from all local ranks; non-masters receive nothing
        if self.is_master:
            intra_gather_recv_splits = local_counts  # receive from each local rank
        else:
            intra_gather_recv_splits = [0] * lws
        
        total_intra_recv = sum(intra_gather_recv_splits)
        intra_recv_buf = torch.empty((total_intra_recv, feature_dim), device=x.device, dtype=x.dtype)
        
        log_nccl_call(__file__, 0, "hier_p1_intra_a2a")
        dist.all_to_all_single(
            intra_recv_buf, send_tensor,
            output_split_sizes=intra_gather_recv_splits,
            input_split_sizes=intra_gather_send_splits,
            group=self.intra_group
        )
        
        # Also gather each rank's per-dest send_splits so master can re-pack
        my_splits_t = torch.tensor(self._cached_send_splits, dtype=torch.long, device=x.device)
        all_splits = [torch.empty(self.world_size, dtype=torch.long, device=x.device) for _ in range(lws)]
        log_nccl_call(__file__, 0, "hier_p1_allgather_splits")
        dist.all_gather(all_splits, my_splits_t, group=self.intra_group)
        # all_splits[lr][dest_rank] = count from local_rank lr to dest_rank
        
        # === PHASE 2: Inter-node all_to_all (master only) ===
        if self.is_master:
            # intra_recv_buf layout: [data_from_lr0, data_from_lr1, ..., data_from_lr(lws-1)]
            # where data_from_lr_i has local_counts[i] atoms, ordered by dest_rank per all_splits[i]
            
            # Re-pack by destination NODE. 
            # For each dest_node n, collect atoms from all local ranks destined for ranks on node n.
            # Packing order within each node-bucket: for source_lr, for dest_rank_in_node
            node_buckets = [[] for _ in range(self.num_nodes)]
            node_bucket_sizes = [0] * self.num_nodes
            
            lr_offset = 0
            for lr in range(lws):
                lr_splits = all_splits[lr].tolist()
                lr_data = intra_recv_buf[lr_offset : lr_offset + local_counts[lr]]
                atom_offset = 0
                for dest_rank in range(self.world_size):
                    count = int(lr_splits[dest_rank])
                    if count > 0:
                        dest_node = dest_rank // lws
                        node_buckets[dest_node].append(lr_data[atom_offset:atom_offset+count])
                        node_bucket_sizes[dest_node] += count
                    atom_offset += count
                lr_offset += local_counts[lr]
            
            # Concatenate per-node buckets
            inter_send_parts = []
            for n in range(self.num_nodes):
                if node_buckets[n]:
                    inter_send_parts.append(torch.cat(node_buckets[n], dim=0))
            inter_send_tensor = torch.cat(inter_send_parts, dim=0).contiguous() if inter_send_parts else \
                torch.empty((0, feature_dim), device=x.device, dtype=x.dtype)
            inter_send_splits = node_bucket_sizes
            
            # Exchange sizes with other masters
            inter_send_splits_t = torch.tensor(inter_send_splits, dtype=torch.long, device=x.device)
            inter_recv_splits_t = torch.empty(self.num_nodes, dtype=torch.long, device=x.device)
            log_nccl_call(__file__, 0, "hier_p2_inter_a2a_splits")
            dist.all_to_all_single(inter_recv_splits_t, inter_send_splits_t, group=self.inter_group)
            inter_recv_splits = [int(s) for s in inter_recv_splits_t.tolist()]
            
            # Also exchange the detail matrix: for each dest_node, 
            # the (source_lr, dest_rank_within_node) breakdown
            # detail[source_lr * lws + dest_lr_within_node] = count
            detail_per_node = []
            for dest_node in range(self.num_nodes):
                detail = []
                for source_lr in range(lws):
                    for dest_lr in range(lws):
                        dest_rank = dest_node * lws + dest_lr
                        detail.append(int(all_splits[source_lr][dest_rank].item()))
                detail_per_node.append(torch.tensor(detail, dtype=torch.long, device=x.device))
            detail_send = torch.cat(detail_per_node)
            detail_recv = torch.empty(self.num_nodes * lws * lws, dtype=torch.long, device=x.device)
            log_nccl_call(__file__, 0, "hier_p2_inter_a2a_detail")
            dist.all_to_all_single(
                detail_recv, detail_send,
                output_split_sizes=[lws * lws] * self.num_nodes,
                input_split_sizes=[lws * lws] * self.num_nodes,
                group=self.inter_group
            )
            detail_matrix = detail_recv.reshape(self.num_nodes, lws, lws)
            # detail_matrix[src_node, src_lr, dst_lr] = atom count
            
            # Do the actual inter-node data exchange (async)
            total_inter_recv = sum(inter_recv_splits)
            inter_recv_tensor = torch.empty((total_inter_recv, feature_dim), device=x.device, dtype=x.dtype)
            
            log_nccl_call(__file__, 0, "hier_p2_inter_a2a_data")
            work = dist.all_to_all_single(
                inter_recv_tensor, inter_send_tensor,
                output_split_sizes=inter_recv_splits,
                input_split_sizes=inter_send_splits,
                group=self.inter_group,
                async_op=True
            )
            
            if self.profiler:
                self.profiler.stop("exchange_async_a2a")
            
            return ('master', work, inter_recv_tensor, inter_recv_splits,
                    detail_matrix, all_splits, local_counts,
                    original_shape, feature_dim, x_flat)
        else:
            # Non-master: Phase 2 is a no-op for us
            if self.profiler:
                self.profiler.stop("exchange_async_a2a")
            
            return ('worker', all_splits, local_counts,
                    original_shape, feature_dim, x_flat)
    
    def async_exchange_wait(self, handle, x):
        """
        Wait for hierarchical exchange to complete and update ghost features.
        
        Phase 3: Master unpacks inter-node data into per-local-rank portions
                  and scatters via intra-node all_to_all.
        """
        if handle is None:
            return x
        
        role = handle[0]
        lws = self.local_world_size
        
        if role == 'master':
            (_, work, inter_recv_tensor, inter_recv_splits,
             detail_matrix, all_splits, local_counts,
             original_shape, feature_dim, x_flat) = handle
            
            # Wait for inter-node all_to_all
            work.wait()
            
            # Unpack inter_recv_tensor into per-local-rank portions.
            # inter_recv_tensor is ordered by source_node, and within each source_node
            # chunk, the packing order is: for src_lr: for dst_rank: atoms
            # detail_matrix[src_node, src_lr, dst_lr] gives the count.
            per_lr_data = [[] for _ in range(lws)]
            
            offset = 0
            for src_node in range(self.num_nodes):
                for src_lr in range(lws):
                    for dst_lr in range(lws):
                        count = int(detail_matrix[src_node, src_lr, dst_lr].item())
                        if count > 0:
                            per_lr_data[dst_lr].append(inter_recv_tensor[offset:offset+count])
                        offset += count
            
            # Concatenate per-local-rank data
            scatter_bufs = []
            scatter_sizes = []
            for lr in range(lws):
                if per_lr_data[lr]:
                    buf = torch.cat(per_lr_data[lr], dim=0)
                else:
                    buf = torch.empty((0, feature_dim), device=x.device, dtype=x.dtype)
                scatter_bufs.append(buf)
                scatter_sizes.append(buf.shape[0])
            
            # Phase 3: scatter via intra-node all_to_all
            intra_scatter_send = torch.cat(scatter_bufs, dim=0).contiguous()
            intra_scatter_send_splits = scatter_sizes  # master sends different amounts to each lr
            
        else:  # worker
            (_, all_splits, local_counts,
             original_shape, feature_dim, x_flat) = handle
            
            # Workers have nothing to send in scatter phase
            intra_scatter_send = torch.empty((0, feature_dim), device=x.device, dtype=x.dtype)
            intra_scatter_send_splits = [0] * lws
        
        # All ranks need to know how much they'll RECEIVE from master in the scatter.
        # Workers receive from master only; master also receives from itself.
        # Exchange sizes via all_gather.
        # Master knows scatter_sizes, workers need their specific size.
        
        if role == 'master':
            my_scatter_size_t = torch.tensor(scatter_sizes, dtype=torch.long, device=x.device)
        else:
            my_scatter_size_t = torch.tensor([0] * lws, dtype=torch.long, device=x.device)
        
        # Gather sizes from all local ranks (only master's is meaningful)
        all_scatter_sizes = [torch.empty(lws, dtype=torch.long, device=x.device) for _ in range(lws)]
        log_nccl_call(__file__, 0, "hier_p3_allgather_scatter_sizes")
        dist.all_gather(all_scatter_sizes, my_scatter_size_t, group=self.intra_group)
        # all_scatter_sizes[master_lr][lr] = how much master sends to lr
        master_scatter_sizes = all_scatter_sizes[lws - 1].tolist()
        
        # Each rank receives only from master
        intra_scatter_recv_splits = [0] * lws
        my_recv_count = int(master_scatter_sizes[self.local_rank])
        intra_scatter_recv_splits[lws - 1] = my_recv_count
        
        intra_scatter_recv = torch.empty((my_recv_count, feature_dim), device=x.device, dtype=x.dtype)
        
        log_nccl_call(__file__, 0, "hier_p3_intra_a2a_scatter")
        dist.all_to_all_single(
            intra_scatter_recv, intra_scatter_send,
            output_split_sizes=intra_scatter_recv_splits,
            input_split_sizes=intra_scatter_send_splits,
            group=self.intra_group
        )
        
        recv_tensor = intra_scatter_recv
        
        # Update ghost positions with received data
        if x.dim() > 2:
            x_flat = x.reshape(original_shape[0], -1)
        else:
            x_flat = x
        
        if self._batched_recv_indices.numel() > 0 and recv_tensor.shape[0] > 0:
            recv_idx = self._batched_recv_indices
            if recv_idx.device != x.device:
                recv_idx = recv_idx.to(x.device)
                self._batched_recv_indices = recv_idx
            x_flat.index_copy_(0, recv_idx, recv_tensor)
        
        if x.dim() > 2:
            return x_flat.reshape(original_shape[0], *original_shape[1:])
        
        return x_flat


@registry.register_model("distributed_escnmd_backbone")
class DistributedESCNMDBackbone(eSCNMDMoeBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize exchanger with dummy values, will be updated later
        self.exchanger = FeatureExchanger(rank=0, world_size=1) 
        self.num_local = 0
        self.profiler = None  # CUDAEventProfiler for detailed timing
        self._cached_mole_coefficients = None  # MOLE cache for first-call pattern
    
    def set_profiler(self, profiler):
        """Set profiler for detailed model timing."""
        self.profiler = profiler
        # Also pass to exchanger for communication timing
        self.exchanger.profiler = profiler

    def set_comm_maps(self, send_map, recv_ghost_indices, rank, world_size, interior_mask=None, enable_overlap=False, hierarchical_params=None):
        # Switch to HierarchicalFeatureExchanger if requested and not already using one
        if hierarchical_params is not None and not isinstance(self.exchanger, HierarchicalFeatureExchanger):
            self.exchanger = HierarchicalFeatureExchanger(rank, world_size)
            self.exchanger.set_groups(
                hierarchical_params['intra_group'],
                hierarchical_params['inter_group'],
                hierarchical_params['local_rank'],
                hierarchical_params['local_world_size']
            )
        
        self.exchanger.rank = rank
        self.exchanger.world_size = world_size
        self.exchanger.update_maps(send_map, {}, recv_ghost_indices)
        # Store interior mask and overlap flag for optimization
        self.interior_mask = interior_mask
        self.enable_overlap = enable_overlap

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        _rank = getattr(self.exchanger, 'rank', -1)
        if self.num_experts == 0:
            # # print(f"[DEBUG] Rank {_rank}: set_MOLE_coefficients: num_experts=0, skipping", flush=True)
            return
        
        # OPTIMIZATION: Cache MOLE coefficients - compute once on first call, reuse forever.
        # In MD simulations with merge_mole=True, composition is static.
        # Check if we can use cache
        # Use getattr because __init__ might be bypassed during pickle loading (torch.load)
        cached = getattr(self, "_cached_mole_coefficients", None)
        if cached is not None:
            # # print(f"[DEBUG] Rank {_rank}: set_MOLE_coefficients: CACHE HIT, reusing coefficients", flush=True)
            self.global_mole_tensors.expert_mixing_coefficients = cached
            return
        
        # # print(f"[DEBUG] Rank {_rank}: set_MOLE_coefficients: CACHE MISS, computing coefficients", flush=True)
        
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
                        
                        # # print(f"[DEBUG] Rank {_rank}: set_MOLE: calling all_reduce (CPU->CUDA path)...", flush=True)
                        log_nccl_call(__file__, 393, "all_reduce_sum_cuda")
                        dist.all_reduce(local_sum_cuda)
                        log_nccl_call(__file__, 395, "all_reduce_count_cuda")
                        dist.all_reduce(local_count_cuda)
                        # # print(f"[DEBUG] Rank {_rank}: set_MOLE: all_reduce done (CPU->CUDA path)", flush=True)
                        
                        local_sum = local_sum_cuda.cpu()
                        local_count = local_count_cuda.cpu()
                    else:
                        # # print(f"[DEBUG] Rank {_rank}: set_MOLE: calling all_reduce (direct path)...", flush=True)
                        log_nccl_call(__file__, 402, "all_reduce_sum_direct")
                        dist.all_reduce(local_sum)
                        log_nccl_call(__file__, 404, "all_reduce_count_direct")
                        dist.all_reduce(local_count)
                        # # print(f"[DEBUG] Rank {_rank}: set_MOLE: all_reduce done (direct path)", flush=True)
                    
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

    def forward(self, data_dict):
        # DEBUG: Track forward pass entry
        _rank = self.exchanger.rank
        # # print(f"[DEBUG] Rank {_rank}: forward() START", flush=True)
        
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

        # # print(f"[DEBUG] Rank {_rank}: calling set_MOLE_coefficients()...", flush=True)
        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )
        # # print(f"[DEBUG] Rank {_rank}: set_MOLE_coefficients() done", flush=True)

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
        
        # === OVERLAP OPTIMIZATION (Area 3) ===
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
                
                # === Area 3 Fix: Pre-compute interior atom indices for efficient merge ===
                interior_indices = torch.where(interior_atom_mask[:self.num_local])[0]
                
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
                
                interior_wigner = _filter_wigner_once(wigner_and_M_mapping, interior_edge_mask, local_dest_mask)
                interior_wigner_inv = _filter_wigner_once(wigner_and_M_mapping_inv, interior_edge_mask, local_dest_mask)
                interior_envelope = edge_envelope[local_dest_mask][interior_edge_mask] if edge_envelope is not None else None
                
                boundary_wigner = _filter_wigner_once(wigner_and_M_mapping, boundary_edge_mask, local_dest_mask)
                boundary_wigner_inv = _filter_wigner_once(wigner_and_M_mapping_inv, boundary_edge_mask, local_dest_mask)
                boundary_envelope = edge_envelope[local_dest_mask][boundary_edge_mask] if edge_envelope is not None else None
                
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
        
        # Update spherical node embeddings (Message Passing Layers)
        for i in range(self.num_layers):
            if use_overlap:
                # === OVERLAPPED EXECUTION (Area 3 Fixed) ===
                if self.profiler:
                    self.profiler.start(f"feature_exchange_{i}")
                with record_function(f"async_exchange_start {i}"):
                    exchange_handle = self.exchanger.async_exchange_start(x_message, self.num_local)
                
                with record_function(f"interior_compute {i}"):
                    # Compute interior atoms while exchange is in progress
                    # Interior atoms only have local neighbors, don't need ghost features
                    if interior_edge_index.shape[1] > 0:
                        # Area 3 Fix: No clone! The block aggregates messages into destination atoms.
                        # Interior destinations only receive from local sources, so using x_message
                        # directly is safe — exchange only modifies ghost slots (>= num_local).
                        x_interior_result = self.blocks[i](
                            x_message,
                            interior_x_edge,
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
                    x_message = self.exchanger.async_exchange_wait(exchange_handle, x_message)
                if self.profiler:
                    self.profiler.stop(f"feature_exchange_{i}")
                
                with record_function(f"boundary_compute {i}"):
                    # Now compute boundary atoms (they need the exchanged ghost features)
                    if boundary_edge_index.shape[1] > 0:
                        x_boundary_result = self.blocks[i](
                            x_message,
                            boundary_x_edge,
                            boundary_edge_distance,
                            boundary_edge_index,
                            boundary_wigner,
                            boundary_wigner_inv,
                            boundary_envelope,
                            total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[0],
                            sys_node_embedding=sys_node_embedding,
                            node_offset=graph_dict["node_offset"],
                        )
                        # Area 3 Fix: Efficient merge using pre-computed interior_indices
                        # boundary_result has correct values for boundary atoms,
                        # but interior atoms need values from x_interior_result
                        x_message = x_boundary_result
                    else:
                        x_message = x_interior_result if interior_edge_index.shape[1] > 0 else x_message
                    
                    # Area 3 Fix: Use index_copy_ with pre-computed indices (not fancy indexing)
                    if interior_edge_index.shape[1] > 0 and interior_indices.numel() > 0:
                        # Extract interior atom embeddings from interior result and copy to merged output
                        x_message[interior_indices] = x_interior_result[interior_indices]
                
            else:
                # === OPTIMIZED PATH: Overlap exchange with Wigner filtering ===
                # Start async exchange first
                # Profile: feature_exchange
                # dist.barrier()  # CRITICAL: Sync before collective to prevent race condition
                if self.profiler:
                    self.profiler.start(f"feature_exchange_{i}")
                with record_function(f"async_exchange_start {i}"):
                    # # print(f"[DEBUG] Rank {_rank}: layer {i} async_exchange_start (non-overlap)...", flush=True)
                    exchange_handle = self.exchanger.async_exchange_start(x_message, self.num_local)
                    # # print(f"[DEBUG] Rank {_rank}: layer {i} async_exchange_start returned", flush=True)
                
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
                    # # print(f"[DEBUG] Rank {_rank}: layer {i} async_exchange_wait (non-overlap)...", flush=True)
                    x_message = self.exchanger.async_exchange_wait(exchange_handle, x_message)
                    # # print(f"[DEBUG] Rank {_rank}: layer {i} async_exchange_wait done", flush=True)
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

            # # print(f"[DEBUG] Rank {_rank}: MP layer {i} END", flush=True)

        # Final layer norm
        # # print(f"[DEBUG] Rank {_rank}: MP layers done, applying final norm", flush=True)
        x_message = self.norm(x_message)
        
        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
            "batch": data_dict["batch"],
        }
        # # print(f"[DEBUG] Rank {_rank}: forward() END", flush=True)
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
