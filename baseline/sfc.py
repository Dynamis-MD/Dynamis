#!/usr/bin/env python3
"""
Space-Filling Curve (SFC) based graph partitioning algorithm.

This module implements partitioning using spatial locality through space-filling curves
such as Hilbert curves and Z-curves (Morton curves).
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
import torch.distributed as dist



def hilbert_curve_index_tensor(coords: torch.Tensor, order: int = 7, normalize: bool = True) -> torch.Tensor:
    """
    Convert 3D coordinates to 1D Hilbert curve index using PyTorch bitwise operations.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        order: Order of the Hilbert curve (determines resolution)
        normalize: Whether to normalize coordinates to [0, 1] range first
        
    Returns:
        Tensor of 1D indices along the Hilbert curve
    """
    device = coords.device
    N = coords.size(0)
    
    if normalize:
        # Normalize coordinates to [0, 1] range
        coords_min = coords.min(dim=0)[0]
        coords_max = coords.max(dim=0)[0]
        coords_range = coords_max - coords_min
        
        # Avoid division by zero
        coords_range = torch.where(coords_range > 0, coords_range, torch.ones_like(coords_range))
        coords_normalized = (coords - coords_min) / coords_range
    else:
        coords_normalized = coords
    
    # Scale to integer grid
    grid_size = 2**order
    coords_int = (coords_normalized * (grid_size - 1)).long()
    
    # Extract x, y, z coordinates
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    
    # Ensure coordinates are within bounds
    max_coord = (1 << order) - 1
    x = torch.clamp(x, 0, max_coord)
    y = torch.clamp(y, 0, max_coord)
    z = torch.clamp(z, 0, max_coord)
    
    # Initialize result tensor
    result = torch.zeros(N, dtype=torch.long, device=device)
    
    # Convert to Hilbert curve index
    for i in range(order):
        # Extract bit at position i for each coordinate
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        
        # Hilbert curve mapping
        if i == 0:
            result |= (x_bit << (3*i)) | (y_bit << (3*i+1)) | (z_bit << (3*i+2))
        else:
            # Apply Hilbert curve transformation using bitwise operations
            # Create masks for conditional operations
            x_mask = x_bit.bool()
            y_mask = y_bit.bool()
            z_mask = z_bit.bool()
            
            # Apply transformations
            new_x_bit = torch.where(x_mask, z_bit, x_bit)
            new_y_bit = torch.where(x_mask, y_bit, 
                          torch.where(y_mask, z_bit, y_bit))
            new_z_bit = torch.where(x_mask, y_bit,
                          torch.where(y_mask, x_bit,
                          torch.where(z_mask, y_bit, z_bit)))
            
            result |= (new_x_bit << (3*i)) | (new_y_bit << (3*i+1)) | (new_z_bit << (3*i+2))
    
    return result


def hilbert_curve_index(x: int, y: int, z: int, order: int) -> int:
    """
    Convert 3D coordinates to 1D Hilbert curve index.
    Fallback implementation for single values.
    
    Args:
        x, y, z: 3D coordinates (integers)
        order: Order of the Hilbert curve (determines resolution)
        
    Returns:
        1D index along the Hilbert curve
    """
    # Ensure coordinates are within bounds
    max_coord = (1 << order) - 1
    x = min(max(x, 0), max_coord)
    y = min(max(y, 0), max_coord)
    z = min(max(z, 0), max_coord)
    
    # Convert to Hilbert curve index
    index = 0
    for i in range(order):
        # Extract bit at position i
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        
        # Hilbert curve mapping
        if i == 0:
            index |= (x_bit << (3*i)) | (y_bit << (3*i+1)) | (z_bit << (3*i+2))
        else:
            # Apply Hilbert curve transformation
            if x_bit:
                y_bit, z_bit = z_bit, y_bit
            if y_bit:
                x_bit, z_bit = z_bit, x_bit
            if z_bit:
                x_bit, y_bit = y_bit, x_bit
                
            index |= (x_bit << (3*i)) | (y_bit << (3*i+1)) | (z_bit << (3*i+2))
    
    return index


def morton_curve_index_tensor(coords: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Convert 3D coordinates to 1D Morton curve (Z-curve) index using PyTorch bitwise operations.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        normalize: Whether to normalize coordinates to [0, 1] range first
        
    Returns:
        Tensor of 1D indices along the Morton curve
    """
    device = coords.device
    N = coords.size(0)
    
    if normalize:
        # Normalize coordinates to [0, 1] range
        coords_min = coords.min(dim=0)[0]
        coords_max = coords.max(dim=0)[0]
        coords_range = coords_max - coords_min
        
        # Avoid division by zero
        coords_range = torch.where(coords_range > 0, coords_range, torch.ones_like(coords_range))
        coords_normalized = (coords - coords_min) / coords_range
    else:
        coords_normalized = coords
    
    # Scale to integer grid (2^21 for Morton curve)
    grid_size = 2**21
    coords_int = (coords_normalized * (grid_size - 1)).long()
    
    # Extract x, y, z coordinates
    x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
    
    # Initialize result tensor
    result = torch.zeros(N, dtype=torch.long, device=device)
    
    # Support up to 21 bits per coordinate (63 bits total for 3D)
    for i in range(21):
        # Extract bit at position i for each coordinate
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        
        # Interleave bits: xyzxyzxyz...
        result |= x_bit << (3*i)
        result |= y_bit << (3*i + 1)
        result |= z_bit << (3*i + 2)
    
    return result


def morton_curve_index(x: int, y: int, z: int) -> int:
    """
    Convert 3D coordinates to 1D Morton curve (Z-curve) index.
    Fallback implementation for single values.
    
    Args:
        x, y, z: 3D coordinates (integers)
        
    Returns:
        1D index along the Morton curve
    """
    # Interleave bits: xyzxyzxyz...
    index = 0
    for i in range(21):  # Support up to 21 bits per coordinate
        index |= ((x >> i) & 1) << (3*i)
        index |= ((y >> i) & 1) << (3*i + 1)
        index |= ((z >> i) & 1) << (3*i + 2)
    
    return index


def coordinates_to_sfc_indices(coords: torch.Tensor, curve_type: str = "morton", 
                              normalize: bool = True) -> torch.Tensor:
    """
    Convert 3D coordinates to space-filling curve indices using PyTorch bitwise operations.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        curve_type: Type of curve ("morton" or "hilbert")
        normalize: Whether to normalize coordinates to [0, 1] range first
        
    Returns:
        Tensor of 1D indices along the space-filling curve
    """
    if curve_type == "morton":
        # Use tensor-based Morton curve calculation
        indices = morton_curve_index_tensor(coords, normalize=normalize)
        
    elif curve_type == "hilbert":
        # Determine appropriate order for Hilbert curve
        order = 7  # 2^7 = 128, should be sufficient for most cases
        indices = hilbert_curve_index_tensor(coords, order=order, normalize=normalize)
        
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")
    
    return indices


def sfc_partition(coords: torch.Tensor, num_partitions: int, curve_type: str = "morton",
                  device: Optional[torch.device] = None,
                  edge_counts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Partition atoms using space-filling curve.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        num_partitions: Number of partitions
        curve_type: Type of curve ("morton" or "hilbert")
        device: Device to run computation on
        edge_counts: Optional tensor of edge counts per atom. If provided,
                    partition by edge cost instead of equal atom count.
        
    Returns:
        Tuple of (partition_labels, metadata)
    """
    if device is None:
        device = coords.device
    
    start_time = time.perf_counter()
    
    # Move coordinates to device
    coords = coords.to(device)
    N = coords.size(0)
    
    # Convert coordinates to SFC indices
    sfc_indices = coordinates_to_sfc_indices(coords, curve_type=curve_type)
    
    # Sort atoms by their SFC indices
    sorted_indices = torch.argsort(sfc_indices)
    
    # Create partition assignments
    partition_id = torch.zeros(N, dtype=torch.long, device=device)
    
    use_weighted = edge_counts is not None
    if use_weighted:
        # Weighted partitioning: partition by cumulative edge cost
        edge_counts = edge_counts.to(device)
        sorted_costs = edge_counts[sorted_indices].float()
        cumulative_costs = sorted_costs.cumsum(dim=0)
        total_cost = cumulative_costs[-1].item()
        target_cost_per_partition = total_cost / num_partitions
        
        current_idx = 0
        for p in range(num_partitions):
            if p == num_partitions - 1:
                partition_id[sorted_indices[current_idx:]] = p
            else:
                target = (p + 1) * target_cost_per_partition
                mask = cumulative_costs >= target
                if mask.any():
                    end_idx = mask.nonzero()[0].item() + 1
                else:
                    end_idx = N
                end_idx = max(end_idx, current_idx + 1)
                partition_id[sorted_indices[current_idx:end_idx]] = p
                current_idx = end_idx
    else:
        # Original: equal atom count partitioning
        atoms_per_partition = N // num_partitions
        remainder = N % num_partitions
        
        current_idx = 0
        for p in range(num_partitions):
            partition_size = atoms_per_partition + (1 if p < remainder else 0)
            end_idx = current_idx + partition_size
            partition_id[sorted_indices[current_idx:end_idx]] = p
            current_idx = end_idx
    
    partition_time = time.perf_counter() - start_time
    
    metadata = {
        "partition_time": partition_time,
        "method": f"sfc_{curve_type}" + ("_weighted" if use_weighted else ""),
        "curve_type": curve_type,
        "weighted": use_weighted
    }
    
    return partition_id, metadata


def distributed_sfc_partition(
    local_coords: torch.Tensor,
    num_partitions: int,
    curve_type: str = "morton",
    device: Optional[torch.device] = None,
    local_edge_counts: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Distributed SFC partition: each rank computes local keys, all-gather keys only,
    then each rank independently determines partition assignments.
    
    This reduces memory usage compared to gathering all atom data to rank 0.
    
    Args:
        local_coords: Tensor of shape [N_local, 3] containing local coordinates
        num_partitions: Number of partitions (typically world_size)
        curve_type: Type of curve ("morton" or "hilbert")
        device: Device to run computation on
        local_edge_counts: Optional tensor of edge counts per local atom.
                          If provided, partition by edge cost instead of atom count.
        
    Returns:
        Tuple of (local_partition_id, metadata)
        - local_partition_id: partition assignment for each LOCAL atom
        - metadata: timing and method info
    """
    import torch.distributed as dist
    
    if device is None:
        device = local_coords.device
    
    start_time = time.perf_counter()
    local_coords = local_coords.to(device)
    N_local = local_coords.size(0)
    
    # Check if distributed
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        # Fall back to regular SFC partition
        return sfc_partition(local_coords, num_partitions, curve_type, device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Step 1: Compute local SFC keys (no communication)
    # Need to normalize using global min/max, so first exchange bounds
    local_min = local_coords.min(dim=0)[0]
    local_max = local_coords.max(dim=0)[0]
    
    # All-reduce to get global bounds
    global_min = local_min.clone()
    global_max = local_max.clone()

    dist.all_reduce(global_min, op=dist.ReduceOp.MIN)

    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    
    # Normalize using global bounds
    coords_range = global_max - global_min
    coords_range = torch.where(coords_range > 0, coords_range, torch.ones_like(coords_range))
    local_coords_normalized = (local_coords - global_min) / coords_range
    
    # Compute SFC keys for local atoms
    if curve_type == "morton":
        grid_size = 2**21
        coords_int = (local_coords_normalized * (grid_size - 1)).long()
        x, y, z = coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]
        local_keys = torch.zeros(N_local, dtype=torch.long, device=device)
        for i in range(21):
            x_bit = (x >> i) & 1
            y_bit = (y >> i) & 1
            z_bit = (z >> i) & 1
            local_keys |= x_bit << (3*i)
            local_keys |= y_bit << (3*i + 1)
            local_keys |= z_bit << (3*i + 2)
    else:  # hilbert
        order = 10  # Increased from 7 for better resolution
        grid_size = 2**order
        coords_int = (local_coords_normalized * (grid_size - 1)).long().clamp(0, grid_size - 1)
        local_keys = hilbert_curve_index_tensor(coords_int, order=order, normalize=False)
    
    # Step 2: All-gather keys from all ranks
    # First exchange counts
    local_count = torch.tensor([N_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]

    dist.all_gather(all_counts, local_count)
    counts = [c.item() for c in all_counts]
    total_atoms = sum(counts)
    
    # Prepare buffers for all-gather
    all_keys_list = [torch.zeros(c, dtype=torch.long, device=device) for c in counts]

    dist.all_gather(all_keys_list, local_keys)
    
    # Concatenate all keys
    all_keys = torch.cat(all_keys_list, dim=0)
    
    # Handle edge counts if provided (for weighted partitioning)
    use_weighted = local_edge_counts is not None
    if use_weighted:
        all_edge_counts_list = [torch.zeros(c, dtype=torch.long, device=device) for c in counts]

        dist.all_gather(all_edge_counts_list, local_edge_counts.to(device))
        all_edge_counts = torch.cat(all_edge_counts_list, dim=0)
    
    # Step 3: Each rank independently determines global partition assignments
    # Sort all keys to determine partition boundaries
    sorted_indices = torch.argsort(all_keys)
    
    # Create global partition assignments
    global_partition_id = torch.zeros(total_atoms, dtype=torch.long, device=device)
    
    if use_weighted:
        # Weighted partitioning: partition by cumulative edge cost
        sorted_costs = all_edge_counts[sorted_indices].float()
        cumulative_costs = sorted_costs.cumsum(dim=0)
        total_cost = cumulative_costs[-1].item()
        target_cost_per_partition = total_cost / num_partitions
        
        current_idx = 0
        for p in range(num_partitions):
            if p == num_partitions - 1:
                # Last partition gets remaining atoms
                global_partition_id[sorted_indices[current_idx:]] = p
            else:
                # Find index where cumulative cost exceeds target
                target = (p + 1) * target_cost_per_partition
                # Binary search for efficiency
                mask = cumulative_costs >= target
                if mask.any():
                    end_idx = mask.nonzero()[0].item() + 1
                else:
                    end_idx = total_atoms
                end_idx = max(end_idx, current_idx + 1)  # At least 1 atom
                global_partition_id[sorted_indices[current_idx:end_idx]] = p
                current_idx = end_idx
    else:
        # Original: equal atom count partitioning
        atoms_per_partition = total_atoms // num_partitions
        remainder = total_atoms % num_partitions
        
        current_idx = 0
        for p in range(num_partitions):
            partition_size = atoms_per_partition + (1 if p < remainder else 0)
            end_idx = current_idx + partition_size
            global_partition_id[sorted_indices[current_idx:end_idx]] = p
            current_idx = end_idx
    
    # Step 4: Extract partition IDs for local atoms
    # The global_partition_id is indexed by global atom order (concatenated local arrays)
    # We need to extract the portion for this rank
    start_idx = sum(counts[:rank])
    end_idx = start_idx + N_local
    local_partition_id = global_partition_id[start_idx:end_idx]
    
    # Step 5: Compute send_counts_matrix for all ranks (eliminates extra all_to_all later)
    # send_counts_matrix[sender, receiver] = number of atoms sender will send to receiver
    send_counts_matrix = torch.zeros((world_size, world_size), dtype=torch.long, device=device)
    offset = 0
    for sender in range(world_size):
        sender_count = counts[sender]
        sender_partition_ids = global_partition_id[offset:offset + sender_count]
        for receiver in range(world_size):
            send_counts_matrix[sender, receiver] = (sender_partition_ids == receiver).sum()
        offset += sender_count
    
    partition_time = time.perf_counter() - start_time
    
    metadata = {
        "partition_time": partition_time,
        "method": f"distributed_sfc_{curve_type}" + ("_weighted" if use_weighted else ""),
        "curve_type": curve_type,
        "total_atoms": total_atoms,
        "local_atoms": N_local,
        "weighted": use_weighted,
        "send_counts_matrix": send_counts_matrix  # Added for optimization
    }
    
    return local_partition_id, metadata


# ---------------------------------------------------------------------------
# KD-Tree partition (single-process and distributed)
# ---------------------------------------------------------------------------

def kdtree_partition(
    coords: torch.Tensor,
    num_partitions: int,
    device: Optional[torch.device] = None,
    edge_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    KD-Tree partition: recursive bisection on the longest axis using GPU torch ops.

    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        num_partitions: Number of partitions
        device: Device to run computation on
        edge_counts: Optional tensor of edge counts per atom. If provided,
                    split by cumulative edge cost instead of atom count.

    Returns:
        Tuple of (partition_labels, metadata)
    """
    if device is None:
        device = coords.device

    start_time = time.perf_counter()
    coords = coords.to(device)
    N = coords.size(0)
    labels = torch.zeros(N, dtype=torch.long, device=device)
    use_weighted = edge_counts is not None
    if use_weighted:
        edge_counts = edge_counts.to(device)

    def _bisect(idx: torch.Tensor, parts: int, base_label: int):
        if parts <= 1 or idx.numel() == 0:
            labels[idx] = base_label
            return
        sub_coords = coords[idx]
        extent = sub_coords.max(dim=0).values - sub_coords.min(dim=0).values
        axis = extent.argmax().item()
        _, order = sub_coords[:, axis].sort()
        sorted_idx = idx[order]

        left_parts = parts // 2
        right_parts = parts - left_parts

        if not use_weighted:
            mid = sorted_idx.size(0) // 2
        else:
            sorted_w = edge_counts[sorted_idx].float()
            cumsum = sorted_w.cumsum(dim=0)
            target = cumsum[-1] / 2.0
            mid = (cumsum >= target).nonzero(as_tuple=False)[0].item()
            mid = max(1, min(mid, sorted_idx.size(0) - 1))

        _bisect(sorted_idx[:mid], left_parts, base_label)
        _bisect(sorted_idx[mid:], right_parts, base_label + left_parts)

    _bisect(torch.arange(N, device=device), num_partitions, 0)

    partition_time = time.perf_counter() - start_time
    metadata = {
        "partition_time": partition_time,
        "method": "kdtree" + ("_weighted" if use_weighted else ""),
        "weighted": use_weighted,
    }
    return labels, metadata


def distributed_kdtree_partition(
    local_coords: torch.Tensor,
    num_partitions: int,
    device: Optional[torch.device] = None,
    local_edge_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Distributed KD-Tree partition: all-gather coordinates, then each rank
    independently runs the same recursive bisection.

    Communication: all-gather N×3 float32 coords (+ N×1 int64 weights if weighted).
    For 1M atoms this is ~12 MB coords + 8 MB weights = 20 MB — negligible.

    The interface and return values are identical to distributed_sfc_partition,
    so it can be used as a drop-in replacement.

    Args:
        local_coords: Tensor of shape [N_local, 3] containing local coordinates
        num_partitions: Number of partitions (typically world_size)
        device: Device to run computation on
        local_edge_counts: Optional tensor of edge counts per local atom.

    Returns:
        Tuple of (local_partition_id, metadata)
        - local_partition_id: partition assignment for each LOCAL atom
        - metadata: timing and method info (includes send_counts_matrix)
    """
    import torch.distributed as dist

    if device is None:
        device = local_coords.device

    start_time = time.perf_counter()
    local_coords = local_coords.to(device)
    N_local = local_coords.size(0)

    # Fallback for single-process
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return kdtree_partition(local_coords, num_partitions, device, local_edge_counts)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Step 1: Exchange counts
    local_count = torch.tensor([N_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    counts = [c.item() for c in all_counts]
    total_atoms = sum(counts)

    # Step 2: All-gather coordinates (N_total x 3, float32)
    all_coords_list = [torch.zeros(c, 3, dtype=torch.float32, device=device) for c in counts]
    dist.all_gather(all_coords_list, local_coords.float())
    all_coords = torch.cat(all_coords_list, dim=0)

    # Step 3: All-gather edge counts if weighted
    use_weighted = local_edge_counts is not None
    all_edge_counts = None
    if use_weighted:
        all_edge_counts_list = [torch.zeros(c, dtype=torch.long, device=device) for c in counts]
        dist.all_gather(all_edge_counts_list, local_edge_counts.to(device))
        all_edge_counts = torch.cat(all_edge_counts_list, dim=0)

    # Step 4: Each rank independently runs KD-tree on global data
    global_partition_id = torch.zeros(total_atoms, dtype=torch.long, device=device)

    def _bisect(idx: torch.Tensor, parts: int, base_label: int):
        if parts <= 1 or idx.numel() == 0:
            global_partition_id[idx] = base_label
            return
        sub_coords = all_coords[idx]
        extent = sub_coords.max(dim=0).values - sub_coords.min(dim=0).values
        axis = extent.argmax().item()
        _, order = sub_coords[:, axis].sort()
        sorted_idx = idx[order]

        left_parts = parts // 2
        right_parts = parts - left_parts

        if not use_weighted:
            mid = sorted_idx.size(0) // 2
        else:
            sorted_w = all_edge_counts[sorted_idx].float()
            cumsum = sorted_w.cumsum(dim=0)
            target = cumsum[-1] / 2.0
            mid = (cumsum >= target).nonzero(as_tuple=False)[0].item()
            mid = max(1, min(mid, sorted_idx.size(0) - 1))

        _bisect(sorted_idx[:mid], left_parts, base_label)
        _bisect(sorted_idx[mid:], right_parts, base_label + left_parts)

    _bisect(torch.arange(total_atoms, device=device), num_partitions, 0)

    # Step 5: Extract local partition IDs
    start_idx = sum(counts[:rank])
    end_idx = start_idx + N_local
    local_partition_id = global_partition_id[start_idx:end_idx]

    # Step 6: Compute send_counts_matrix (same as distributed_sfc_partition)
    send_counts_matrix = torch.zeros((world_size, world_size), dtype=torch.long, device=device)
    offset = 0
    for sender in range(world_size):
        sender_count = counts[sender]
        sender_partition_ids = global_partition_id[offset:offset + sender_count]
        for receiver in range(world_size):
            send_counts_matrix[sender, receiver] = (sender_partition_ids == receiver).sum()
        offset += sender_count

    partition_time = time.perf_counter() - start_time
    metadata = {
        "partition_time": partition_time,
        "method": "distributed_kdtree" + ("_weighted" if use_weighted else ""),
        "total_atoms": total_atoms,
        "local_atoms": N_local,
        "weighted": use_weighted,
        "send_counts_matrix": send_counts_matrix,
    }
    return local_partition_id, metadata


# ---------------------------------------------------------------------------
# PCA Slab partition (single-process and distributed)
# ---------------------------------------------------------------------------

def pca_slab_partition(
    coords: torch.Tensor,
    num_partitions: int,
    device: Optional[torch.device] = None,
    edge_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PCA Slab partition: project atoms onto the principal component direction
    (direction of maximum variance), then partition along this 1D projection.

    Guarantees ring communication topology: rank n's neighbors are only on
    rank n-1 and rank n+1 (provided slab thickness > cutoff radius).

    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        num_partitions: Number of partitions
        device: Device to run computation on
        edge_counts: Optional tensor of edge counts per atom for weighted partitioning

    Returns:
        Tuple of (partition_labels, metadata)
    """
    if device is None:
        device = coords.device

    start_time = time.perf_counter()
    coords = coords.to(device)
    N = coords.size(0)
    use_weighted = edge_counts is not None

    # Compute PCA direction (eigenvector of largest eigenvalue of covariance)
    centered = coords - coords.mean(dim=0)
    cov = centered.T @ centered  # [3, 3]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    principal_dir = eigenvectors[:, -1]  # largest eigenvalue is last

    # Project onto principal direction
    projections = coords @ principal_dir  # [N]

    # Sort by projection
    sorted_indices = torch.argsort(projections)

    # Partition into contiguous segments
    partition_id = torch.zeros(N, dtype=torch.long, device=device)

    if use_weighted:
        edge_counts = edge_counts.to(device)
        sorted_costs = edge_counts[sorted_indices].float()
        cumulative_costs = sorted_costs.cumsum(dim=0)
        total_cost = cumulative_costs[-1].item()
        target_cost_per_partition = total_cost / num_partitions

        current_idx = 0
        for p in range(num_partitions):
            if p == num_partitions - 1:
                partition_id[sorted_indices[current_idx:]] = p
            else:
                target = (p + 1) * target_cost_per_partition
                mask = cumulative_costs >= target
                if mask.any():
                    end_idx = mask.nonzero()[0].item() + 1
                else:
                    end_idx = N
                end_idx = max(end_idx, current_idx + 1)
                partition_id[sorted_indices[current_idx:end_idx]] = p
                current_idx = end_idx
    else:
        atoms_per_partition = N // num_partitions
        remainder = N % num_partitions
        current_idx = 0
        for p in range(num_partitions):
            partition_size = atoms_per_partition + (1 if p < remainder else 0)
            end_idx = current_idx + partition_size
            partition_id[sorted_indices[current_idx:end_idx]] = p
            current_idx = end_idx

    partition_time = time.perf_counter() - start_time
    metadata = {
        "partition_time": partition_time,
        "method": "pca_slab" + ("_weighted" if use_weighted else ""),
        "weighted": use_weighted,
        "principal_dir": principal_dir.cpu().numpy().tolist(),
    }
    return partition_id, metadata


def distributed_pca_slab_partition(
    local_coords: torch.Tensor,
    num_partitions: int,
    device: Optional[torch.device] = None,
    local_edge_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Distributed PCA Slab partition: all-gather coordinates, compute global PCA
    direction, project and partition. Each rank independently computes the same
    result (deterministic).

    Guarantees ring communication topology when slab thickness > cutoff.

    Args:
        local_coords: Tensor of shape [N_local, 3] containing local coordinates
        num_partitions: Number of partitions (typically world_size)
        device: Device to run computation on
        local_edge_counts: Optional tensor of edge counts per local atom

    Returns:
        Tuple of (local_partition_id, metadata)
    """
    import torch.distributed as dist

    if device is None:
        device = local_coords.device

    start_time = time.perf_counter()
    local_coords = local_coords.to(device)
    N_local = local_coords.size(0)

    # Fallback for single-process
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return pca_slab_partition(local_coords, num_partitions, device, local_edge_counts)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Step 1: Exchange counts
    local_count = torch.tensor([N_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    counts = [c.item() for c in all_counts]
    total_atoms = sum(counts)

    # Step 2: All-gather coordinates
    all_coords_list = [torch.zeros(c, 3, dtype=torch.float32, device=device) for c in counts]
    dist.all_gather(all_coords_list, local_coords.float())
    all_coords = torch.cat(all_coords_list, dim=0)

    # Step 3: All-gather edge counts if weighted
    use_weighted = local_edge_counts is not None
    all_edge_counts = None
    if use_weighted:
        all_edge_counts_list = [torch.zeros(c, dtype=torch.long, device=device) for c in counts]
        dist.all_gather(all_edge_counts_list, local_edge_counts.to(device))
        all_edge_counts = torch.cat(all_edge_counts_list, dim=0)

    # Step 4: Compute global PCA direction
    centered = all_coords - all_coords.mean(dim=0)
    cov = centered.T @ centered  # [3, 3]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    principal_dir = eigenvectors[:, -1]  # largest eigenvalue

    # Project all atoms onto principal direction
    projections = all_coords @ principal_dir  # [total_atoms]

    # Sort by projection
    sorted_indices = torch.argsort(projections)

    # Partition into contiguous segments
    global_partition_id = torch.zeros(total_atoms, dtype=torch.long, device=device)

    if use_weighted:
        sorted_costs = all_edge_counts[sorted_indices].float()
        cumulative_costs = sorted_costs.cumsum(dim=0)
        total_cost = cumulative_costs[-1].item()
        target_cost_per_partition = total_cost / num_partitions

        current_idx = 0
        for p in range(num_partitions):
            if p == num_partitions - 1:
                global_partition_id[sorted_indices[current_idx:]] = p
            else:
                target = (p + 1) * target_cost_per_partition
                mask = cumulative_costs >= target
                if mask.any():
                    end_idx = mask.nonzero()[0].item() + 1
                else:
                    end_idx = total_atoms
                end_idx = max(end_idx, current_idx + 1)
                global_partition_id[sorted_indices[current_idx:end_idx]] = p
                current_idx = end_idx
    else:
        atoms_per_partition = total_atoms // num_partitions
        remainder = total_atoms % num_partitions
        current_idx = 0
        for p in range(num_partitions):
            partition_size = atoms_per_partition + (1 if p < remainder else 0)
            end_idx = current_idx + partition_size
            global_partition_id[sorted_indices[current_idx:end_idx]] = p
            current_idx = end_idx

    # Step 5: Extract local partition IDs
    start_idx = sum(counts[:rank])
    end_idx = start_idx + N_local
    local_partition_id = global_partition_id[start_idx:end_idx]

    # Step 6: Compute send_counts_matrix
    send_counts_matrix = torch.zeros((world_size, world_size), dtype=torch.long, device=device)
    offset = 0
    for sender in range(world_size):
        sender_count = counts[sender]
        sender_partition_ids = global_partition_id[offset:offset + sender_count]
        for receiver in range(world_size):
            send_counts_matrix[sender, receiver] = (sender_partition_ids == receiver).sum()
        offset += sender_count

    # Compute slab thickness for diagnostics
    # Min distance between non-adjacent partition boundaries along principal direction
    proj_sorted = projections[sorted_indices]
    slab_thicknesses = []
    current_idx = 0
    for p in range(num_partitions):
        count_p = (global_partition_id == p).sum().item()
        slab_start = proj_sorted[current_idx].item()
        slab_end = proj_sorted[current_idx + count_p - 1].item()
        slab_thicknesses.append(slab_end - slab_start)
        current_idx += count_p

    partition_time = time.perf_counter() - start_time
    metadata = {
        "partition_time": partition_time,
        "method": "distributed_pca_slab" + ("_weighted" if use_weighted else ""),
        "total_atoms": total_atoms,
        "local_atoms": N_local,
        "weighted": use_weighted,
        "send_counts_matrix": send_counts_matrix,
        "principal_dir": principal_dir.cpu().numpy().tolist(),
        "slab_thicknesses": slab_thicknesses,
        "min_slab_thickness": min(slab_thicknesses) if slab_thicknesses else 0.0,
    }
    return local_partition_id, metadata


def compute_sfc_edge_cut(edge_index: torch.Tensor, partition_id: torch.Tensor) -> int:
    """
    Compute edge cut for SFC partition.
    
    Args:
        edge_index: Edge list tensor of shape [2, num_edges]
        partition_id: Partition assignment for each node
        
    Returns:
        Number of edges that cross partition boundaries
    """
    src, dst = edge_index[0], edge_index[1]
    # Count edges where source and destination are in different partitions
    cut_edges = (partition_id[src] != partition_id[dst]).sum().item()
    # print(f"Edge cut: {cut_edges}, edge_index: {edge_index.shape}")
    return cut_edges


def compute_sfc_balance_metrics(partition_id: torch.Tensor, num_partitions: int) -> Dict[str, float]:
    """
    Compute balance metrics for SFC partition.
    
    Args:
        partition_id: Partition assignment for each node
        num_partitions: Number of partitions
        
    Returns:
        Dictionary containing imbalance ratio and partition sizes
    """
    device = partition_id.device
    counts = torch.zeros(num_partitions, dtype=torch.long, device=device)
    
    for i in range(num_partitions):
        counts[i] = (partition_id == i).sum()
    
    mean_size = counts.float().mean()
    max_size = counts.max().float()
    
    if mean_size > 0:
        imbalance = max_size / mean_size
    else:
        imbalance = 1.0
    
    return {
        "imbalance": imbalance.item(),
        "max_size": max_size.item(),
        "mean_size": mean_size.item(),
        "partition_sizes": counts.cpu().numpy().tolist()
    }


def run_sfc_partition(coords: torch.Tensor, num_atoms: int, k: int, 
                     device: torch.device, curve_type: str = "morton") -> Tuple[List[int], Dict[str, float]]:
    """
    Run SFC partition algorithm.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        num_atoms: Number of atoms/nodes
        k: Number of partitions
        device: Device to run on
        curve_type: Type of curve ("morton" or "hilbert")
        
    Returns:
        Tuple of (partition_labels, metadata)
    """
    start_time = time.perf_counter()
    
    # Run SFC partition
    partition_id, metadata = sfc_partition(
        coords=coords,
        num_partitions=k,
        curve_type=curve_type,
        device=device
    )
    
    partition_time = time.perf_counter() - start_time
    
    # Convert to list format
    labels = partition_id.cpu().numpy().tolist()
    
    metadata.update({
        "partition_time": partition_time,
        "method": f"sfc_{curve_type}"
    })
    
    return labels, metadata


def test_sfc_partition():
    """Test SFC partition algorithm with synthetic data."""
    print("Testing SFC partition algorithm...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic 3D coordinates
    N = 1000
    coords = torch.rand(N, 3, device=device) * 10.0  # Random coordinates in [0, 10]
    
    print(f"Testing with {N} atoms")
    
    # Test Morton curve
    print("\n--- Morton Curve ---")
    partition_id, metadata = sfc_partition(coords, num_partitions=4, curve_type="morton", device=device)
    balance = compute_sfc_balance_metrics(partition_id, 4)
    
    print(f"Partition time: {metadata['partition_time']:.4f}s")
    print(f"Imbalance: {balance['imbalance']:.3f}")
    print(f"Partition sizes: {balance['partition_sizes']}")
    
    # Test Hilbert curve
    print("\n--- Hilbert Curve ---")
    partition_id, metadata = sfc_partition(coords, num_partitions=4, curve_type="hilbert", device=device)
    balance = compute_sfc_balance_metrics(partition_id, 4)
    
    print(f"Partition time: {metadata['partition_time']:.4f}s")
    print(f"Imbalance: {balance['imbalance']:.3f}")
    print(f"Partition sizes: {balance['partition_sizes']}")


if __name__ == "__main__":
    test_sfc_partition()
