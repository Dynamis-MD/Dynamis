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
                  device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Partition atoms using space-filling curve.
    
    Args:
        coords: Tensor of shape [N, 3] containing (x, y, z) coordinates
        num_partitions: Number of partitions
        curve_type: Type of curve ("morton" or "hilbert")
        device: Device to run computation on
        
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
    
    # Divide sorted atoms into roughly equal partitions
    atoms_per_partition = N // num_partitions
    remainder = N % num_partitions
    
    current_idx = 0
    for p in range(num_partitions):
        # Calculate partition size (distribute remainder among first partitions)
        partition_size = atoms_per_partition + (1 if p < remainder else 0)
        
        # Assign atoms to this partition
        end_idx = current_idx + partition_size
        partition_id[sorted_indices[current_idx:end_idx]] = p
        
        current_idx = end_idx
    
    partition_time = time.perf_counter() - start_time
    
    metadata = {
        "partition_time": partition_time,
        "method": f"sfc_{curve_type}",
        "curve_type": curve_type
    }
    
    return partition_id, metadata


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
