"""
Halo Exchange for distributed molecular dynamics simulation.

Provides HaloExchanger class and halo map construction functions.
"""

import torch
import torch.distributed as dist
import ase
from typing import Dict, List, Optional, Tuple
from baseline.nccl_debug import log_nccl_call

from baseline.local_fairchem.graph.radius_graph_pbc import radius_graph_pbc_v2


def _build_edges_cutoff_fairchem_raw(pos: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor):
    """Helper to call fairchem graph builder and return edge_index."""    
    # Ensure inputs are on device and correct type
    if pos.device.type == 'cpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos = pos.to(device)
        cell = cell.to(device)
        pbc = pbc.to(device)
    else:
        device = pos.device
        
    class SimpleData:
        def __init__(self, pos, cell, pbc, natoms):
            self.pos = pos
            self.cell = cell.unsqueeze(0) # (1, 3, 3)
            self.pbc = pbc.unsqueeze(0)   # (1, 3)
            self.natoms = natoms
            self.batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    natoms = torch.tensor([pos.size(0)], dtype=torch.long, device=device)
    data = SimpleData(pos, cell, pbc, natoms)
    
    edge_index, cell_offsets, neighbors = radius_graph_pbc_v2(
        data, cutoff, 1000, pbc=pbc # max_num_neighbors_threshold
    )
    return edge_index, cell_offsets


def build_halo_maps(
    pos: torch.Tensor,
    partition_id: torch.Tensor,
    cutoff: float,
    gnn_layers: int,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    world_size: int
) -> Tuple[List[Dict[int, torch.Tensor]], List[torch.Tensor]]:
    """
    Build halo exchange maps using Graph BFS.
    Returns:
        send_maps: list of dicts, where list[rank_s][rank_r] = tensor of local indices 
                   that rank_s needs to send to rank_r.
        interior_masks: list of boolean tensors (one per rank), where interior_masks[r][i] = True
                        if local atom i of rank r has NO ghost neighbors (all neighbors are local).
    """
    device = pos.device
    num_atoms = pos.size(0)
    
    # 1. Build Base Graph (1-hop)
    edge_index, _ = _build_edges_cutoff_fairchem_raw(pos, cutoff, cell, pbc)
    edge_index = edge_index.to(device)
                
    # 2. Initialize Requirement Mask (N, WorldSize)
    # needed_by[i, r] means atom i is needed by rank r
    needed_by = torch.zeros((num_atoms, world_size), dtype=torch.bool, device=device)
    
    # Initial state: each atom is needed by its owner rank
    # partition_id is (N,)
    # We can use scatter to set true
    rows = torch.arange(num_atoms, device=device)
    needed_by[rows, partition_id] = True
    
    # 3. BFS Propagation
    src, dst = edge_index
    
    # For Layer-wise Feature Exchange, we only need 1-hop halo atoms.
    # The multi-hop BFS is legacy logic for Ghost Zone computation and is removed.
    
    # Mark 1-hop neighbors as needed
    # If edge is (src, dst), then src is a neighbor of dst.
    # If dst is owned by rank R, then src is needed by rank R.
    owner_dst = partition_id[dst]
    needed_by[src, owner_dst] = True

    # 4. Generate Send Maps
    send_maps = [{} for _ in range(world_size)]
    
    for r_sender in range(world_size):
        # Indices of atoms owned by r_sender
        sender_global_indices = torch.nonzero(partition_id == r_sender).squeeze(-1)
        
        if len(sender_global_indices) == 0:
            continue
            
        # Filter needed_by for these atoms
        # shape: (Num_Local, WorldSize)
        local_needed = needed_by[sender_global_indices]
        
        for r_receiver in range(world_size):
            if r_sender == r_receiver:
                continue
            
            # Which local atoms are needed by r_receiver?
            # boolean mask of shape (Num_Local,)
            mask = local_needed[:, r_receiver]
            
            if mask.any():
                # Get local indices (indices into the sender's array)
                send_indices = torch.nonzero(mask).squeeze(-1)
                send_maps[r_sender][r_receiver] = send_indices.to(torch.int32).cpu()
    
    # 5. Compute Interior Masks
    # Interior atom = local atom whose ALL neighbors are also local (same rank)
    # i.e., atom i is interior if for all edges (j, i), partition_id[j] == partition_id[i]
    interior_masks = []
    
    for r in range(world_size):
        # Get indices of atoms owned by rank r
        local_mask = (partition_id == r)
        local_count = local_mask.sum().item()
        
        if local_count == 0:
            interior_masks.append(torch.empty(0, dtype=torch.bool, device=device))
            continue
        
        # For each local atom, check if any neighbor is from a different rank (ghost)
        # Find edges where dst is owned by rank r
        dst_is_local_r = (partition_id[dst] == r)
        
        # Among those edges, check if src is from a different rank
        src_of_local_edges = src[dst_is_local_r]
        dst_of_local_edges = dst[dst_is_local_r]
        src_is_ghost = (partition_id[src_of_local_edges] != r)
        
        # Mark atoms that have at least one ghost neighbor as boundary
        # has_ghost_neighbor[global_idx] = True if this atom has ghost neighbor
        has_ghost_neighbor = torch.zeros(num_atoms, dtype=torch.bool, device=device)
        boundary_atom_indices = dst_of_local_edges[src_is_ghost]
        if boundary_atom_indices.numel() > 0:
            has_ghost_neighbor.scatter_(0, boundary_atom_indices, True)
        
        # Interior = local AND NOT has_ghost_neighbor
        # Extract only for this rank's atoms (in local indexing)
        local_indices = torch.nonzero(local_mask).squeeze(-1)
        interior_global = local_mask & ~has_ghost_neighbor
        interior_local = interior_global[local_indices]
        
        interior_masks.append(interior_local.cpu())
    
    return send_maps, interior_masks


def distributed_build_halo_maps(
    local_pos: torch.Tensor,    # [N_local, 3]
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
    """
    Distributed halo map construction - each rank computes independently.
    
    Steps:
    1. all_gather local_pos to get full_pos
    2. Each rank builds full edge graph (same result due to deterministic computation)
    3. Each rank extracts its own send/recv maps
    
    Returns:
        send_map: {recv_rank: local_indices_to_send}
        recv_ghost_indices: {send_rank: local_ghost_indices_to_receive_at}
        interior_mask: boolean mask for interior atoms [N_local]
    """
    N_local = local_pos.size(0)
    
    # Step 1: All-gather positions to get full graph
    local_count = torch.tensor([N_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    log_nccl_call(__file__, 185, "all_gather_counts")
    dist.all_gather(all_counts, local_count)
    counts = [c.item() for c in all_counts]
    total_atoms = sum(counts)
    
    # All-gather positions
    all_pos_list = [torch.zeros((c, 3), dtype=torch.float32, device=device) for c in counts]
    log_nccl_call(__file__, 191, "all_gather_pos")
    dist.all_gather(all_pos_list, local_pos.contiguous())
    full_pos = torch.cat(all_pos_list, dim=0)
    
    # Create global partition_id (each atom belongs to the rank it resides on)
    partition_id = torch.zeros(total_atoms, dtype=torch.long, device=device)
    offset = 0
    for r, c in enumerate(counts):
        partition_id[offset:offset+c] = r
        offset += c
    
    # Step 2: Build full edge graph (all ranks compute the same thing)
    edge_index, _ = _build_edges_cutoff_fairchem_raw(full_pos, cutoff, cell, pbc)
    edge_index = edge_index.to(device)
    src, dst = edge_index
    
    # Step 3: Compute this rank's send_map
    # For each edge (src, dst), if dst is owned by this rank and src is owned by another rank,
    # then src needs to be sent TO this rank (recv from perspective of this rank)
    # Conversely, if src is owned by this rank and dst is owned by another rank,
    # then src needs to be sent FROM this rank to that rank
    
    my_send_map = {}
    my_recv_indices = {}
    
    # Compute start indices for each rank in global indexing
    rank_start_indices = [0]
    for c in counts[:-1]:
        rank_start_indices.append(rank_start_indices[-1] + c)
    
    my_start = rank_start_indices[rank]
    my_end = my_start + counts[rank]
    
    # Find edges where source is mine and destination is another rank
    my_atoms_mask = (partition_id[src] == rank)
    for recv_rank in range(world_size):
        if recv_rank == rank:
            continue
        # Edges: my atoms -> recv_rank's atoms
        # These are atoms I need to send to recv_rank
        needs_send_mask = my_atoms_mask & (partition_id[dst] == recv_rank)
        if needs_send_mask.any():
            global_indices = src[needs_send_mask].unique()
            # Convert to local indices 
            local_indices = global_indices - my_start
            my_send_map[recv_rank] = local_indices.to(torch.int32).cpu()
    
    # Find edges where destination is mine and source is another rank
    dst_is_mine = (partition_id[dst] == rank)
    
    # Build recv_ghost_indices: which local ghost slots correspond to each sending rank
    # After redistribution, local tensor has N_local atoms, then ghosts are appended
    ghost_offset = N_local
    
    for send_rank in range(world_size):
        if send_rank == rank:
            continue
        # Edges: send_rank's atoms -> my atoms
        needs_recv_mask = (partition_id[src] == send_rank) & dst_is_mine
        if needs_recv_mask.any():
            # Global indices of atoms I need from send_rank
            global_ghost_indices = src[needs_recv_mask].unique()
            # Convert to indices relative to send_rank's local array
            send_rank_start = rank_start_indices[send_rank]
            local_ghost_indices_in_sender = global_ghost_indices - send_rank_start
            
            # Assign local ghost slots (ghosts are appended after local atoms)
            n_ghosts_from_rank = len(global_ghost_indices)
            ghost_slots = torch.arange(
                ghost_offset, 
                ghost_offset + n_ghosts_from_rank, 
                device=device, 
                dtype=torch.long
            )
            ghost_offset += n_ghosts_from_rank
            my_recv_indices[send_rank] = ghost_slots.cpu()
    
    # Step 4: Compute interior mask
    # Interior atom = local atom whose ALL neighbors are also local (same rank)
    # Find edges where dst is my atom but src is from a different rank
    src_is_ghost = (partition_id[src] != rank)
    has_ghost_neighbor = torch.zeros(counts[rank], dtype=torch.bool, device=device)
    
    ghost_edges_to_me = dst_is_mine & src_is_ghost
    if ghost_edges_to_me.any():
        # Convert dst to local indices
        local_dst = dst[ghost_edges_to_me] - my_start
        if local_dst.numel() > 0:
            has_ghost_neighbor.scatter_(0, local_dst, True)
    
    interior_mask = ~has_ghost_neighbor
    
    return my_send_map, my_recv_indices, interior_mask.cpu()


class HaloExchanger:
    """Exchanges ghost atom data between MPI ranks for distributed MD."""
    
    def __init__(self, rank: int, world_size: int, comm_map: Dict[int, torch.Tensor], 
                 device: torch.device, recv_ghost_indices: Dict[int, torch.Tensor] = None):
        self.rank = rank
        self.world_size = world_size
        self.comm_map = {r: indices.to(device) for r, indices in comm_map.items()} # Dict[target_rank, local_indices_to_send]
        self.device = device
        
        # Store recv_ghost_indices if provided (from distributed halo map)
        self._recv_ghost_indices_precomputed = None
        if recv_ghost_indices is not None:
            self._recv_ghost_indices_precomputed = {
                r: indices.to(device) for r, indices in recv_ghost_indices.items()
            }
        
        # Pre-compute receive counts
        self.recv_counts = {}
        self._exchange_counts()

    def _exchange_counts(self):
        """Exchange the number of atoms to be sent/received."""
        if self.world_size <= 1:
            return
            
        # Send counts to neighbors
        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=self.device)
        for r, indices in self.comm_map.items():
            send_counts[r] = len(indices)
            
        # All-to-all to get recv counts
        recv_counts_tensor = torch.zeros(self.world_size, dtype=torch.int32, device=self.device)
        log_nccl_call(__file__, 318, "all_to_all_exchange_counts")
        dist.all_to_all_single(recv_counts_tensor, send_counts)
        
        self.recv_counts = {}
        for r in range(self.world_size):
            c = recv_counts_tensor[r].item()
            if c > 0:
                self.recv_counts[r] = c
                
        # OPTIMIZATION: Pre-compute reduction indices for single index_add_
        # Concatenate indices in order of ranks (matching recv_tensor structure in reduce_forces)
        reduction_indices_list = []
        for r in range(self.world_size):
            if r in self.comm_map:
                reduction_indices_list.append(self.comm_map[r])
        
        if reduction_indices_list:
            self._reduction_indices = torch.cat(reduction_indices_list)
        else:
            self._reduction_indices = torch.empty(0, dtype=torch.long, device=self.device)
        
        # === Area 4: Pre-compute split sizes and concatenated send indices ===
        # For exchange:
        self._send_split_sizes = []
        send_indices_list = []
        for r in range(self.world_size):
            if r in self.comm_map and len(self.comm_map[r]) > 0:
                self._send_split_sizes.append(len(self.comm_map[r]))
                send_indices_list.append(self.comm_map[r])
            else:
                self._send_split_sizes.append(0)
        if send_indices_list:
            self._cached_send_indices = torch.cat(send_indices_list)
        else:
            self._cached_send_indices = None
        
        self._recv_split_sizes = []
        for r in range(self.world_size):
            self._recv_split_sizes.append(self.recv_counts.get(r, 0))
        self._total_recv = sum(self._recv_split_sizes)
        
        # For reduce_forces:
        self._force_send_splits = self._recv_split_sizes  # We send back what we received
        self._force_recv_splits = []
        total_force_recv = 0
        for r in range(self.world_size):
            count = len(self.comm_map[r]) if r in self.comm_map else 0
            self._force_recv_splits.append(count)
            total_force_recv += count
        self._total_force_recv = total_force_recv

    def exchange(self, atoms: ase.Atoms) -> Dict[str, torch.Tensor]:
        """Exchange ghost atoms based on pre-computed map."""
        if self.world_size <= 1:
            return {}

        # Prepare data
        # OPTIMIZATION: Reuse GPU buffers instead of creating new tensors every step
        num_atoms = len(atoms)
        if not hasattr(self, '_pos_buffer') or self._pos_buffer.shape[0] != num_atoms:
            self._pos_buffer = torch.empty((num_atoms, 3), dtype=torch.float32, device=self.device)
            self._vel_buffer = torch.empty((num_atoms, 3), dtype=torch.float32, device=self.device)
        
        # Use copy_ for efficient in-place transfer
        self._pos_buffer.copy_(torch.as_tensor(atoms.get_positions(), dtype=torch.float32))
        self._vel_buffer.copy_(torch.as_tensor(atoms.get_velocities(), dtype=torch.float32))
        pos = self._pos_buffer
        vel = self._vel_buffer
        
        # OPTIMIZATION: Cache Z tensor (atomic numbers are static during MD)
        if not hasattr(self, '_cached_Z') or self._cached_Z is None or self._cached_Z.shape[0] != num_atoms:
            self._cached_Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        Z = self._cached_Z
        
        # === Area 4: Fused halo packing using pre-computed indices ===
        if self._cached_send_indices is not None:
            s_pos = pos[self._cached_send_indices]     # [total_send, 3]
            s_vel = vel[self._cached_send_indices]     # [total_send, 3]
            s_Z = Z[self._cached_send_indices].float().unsqueeze(1)  # [total_send, 1]
            input_tensor = torch.cat([s_pos, s_vel, s_Z], dim=1)  # [total_send, 7]
        else:
            input_tensor = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        
        # 2. Prepare Output Tensor using pre-computed sizes
        output_tensor = torch.zeros((self._total_recv, 7), dtype=torch.float32, device=self.device)
        
        # 3. All-to-All
        log_nccl_call(__file__, 392, "all_to_all_exchange_ghosts")
        dist.all_to_all_single(output_tensor, input_tensor, 
                               output_split_sizes=self._recv_split_sizes, 
                               input_split_sizes=self._send_split_sizes)
        
        # 4. Unpack
        if self._total_recv == 0:
            return {}
        
        ghost_pos = output_tensor[:, 0:3]
        ghost_vel = output_tensor[:, 3:6]
        ghost_Z = output_tensor[:, 6].long()
        
        return {
            "pos": ghost_pos,
            "vel": ghost_vel,
            "Z": ghost_Z
        }

    def reduce_forces(self, local_forces, ghost_forces):
        """
        Redistribute ghost forces back to their owners and add to local forces.
        """
        if self.world_size <= 1:
            return local_forces
        
        # CRITICAL: All ranks must participate in all_to_all_single!
        if ghost_forces is None:
            ghost_forces = torch.zeros((0, 3), dtype=local_forces.dtype, device=self.device)

        send_tensor = ghost_forces
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
            
        recv_tensor = torch.zeros((self._total_force_recv, 3), dtype=local_forces.dtype, device=self.device)
        
        # Exchange using pre-computed splits
        log_nccl_call(__file__, 463, "all_to_all_reduce_forces")
        dist.all_to_all_single(recv_tensor, send_tensor, 
                               output_split_sizes=self._force_recv_splits, 
                               input_split_sizes=self._force_send_splits)
        
        # Scatter to local forces
        if self._reduction_indices.numel() > 0:
             local_forces.index_add_(0, self._reduction_indices, recv_tensor)
                
        return local_forces
    
    def reduce_forces_start(self, local_forces, ghost_forces):
        """
        Area 4: Start async force reduction. Returns handle for reduce_forces_wait.
        """
        if self.world_size <= 1:
            return None
        
        if ghost_forces is None:
            ghost_forces = torch.zeros((0, 3), dtype=local_forces.dtype, device=self.device)

        send_tensor = ghost_forces
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
            
        recv_tensor = torch.zeros((self._total_force_recv, 3), dtype=local_forces.dtype, device=self.device)
        
        log_nccl_call(__file__, 463, "all_to_all_reduce_forces_async")
        work = dist.all_to_all_single(recv_tensor, send_tensor, 
                                       output_split_sizes=self._force_recv_splits, 
                                       input_split_sizes=self._force_send_splits,
                                       async_op=True)
        
        return (work, recv_tensor, local_forces)
    
    def reduce_forces_wait(self, handle):
        """
        Area 4: Wait for async force reduction and scatter results.
        """
        if handle is None:
            return None
        
        work, recv_tensor, local_forces = handle
        work.wait()
        
        if self._reduction_indices.numel() > 0:
            local_forces.index_add_(0, self._reduction_indices, recv_tensor)
        
        return local_forces
