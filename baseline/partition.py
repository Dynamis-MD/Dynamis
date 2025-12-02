#!/usr/bin/env python3
"""
Performance test script for comparing different partition algorithms
using GPU fairchem graph building on datasets from 1k to 2M atoms.
"""

import os
import sys
import time
import csv
import argparse
from typing import List, Tuple, Dict, Optional
import glob
import re

import numpy as np
import torch

# Ensure project root is importable
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Import from current partition-analysis modules
from graph import read_xyz_as_atoms, build_graph_fairchem_timed, compute_edge_cut, balance_metrics
from methods import run_pymetis
from sfc import run_sfc_partition, compute_sfc_edge_cut, compute_sfc_balance_metrics

# Import our MS-BFS implementation
# MS-BFS is in the parent directory of the current project
msbfs_path = os.path.join(os.path.dirname(_ROOT), 'msbfs')
if msbfs_path not in sys.path:
    sys.path.append(msbfs_path)

try:
    from gpu_multi_seed_bfs import gpu_multi_seed_bfs_partition, compute_edge_cut as msbfs_compute_edge_cut, compute_balance_metrics as msbfs_compute_balance_metrics
    MSBFS_AVAILABLE = True
    print(f"MS-BFS module loaded from {msbfs_path}")
except ImportError as e:
    print(f"Warning: MS-BFS not available: {e}")
    print(f"Tried to load from: {msbfs_path}")
    print(f"Available files in msbfs directory:")
    if os.path.exists(msbfs_path):
        for f in os.listdir(msbfs_path):
            print(f"  {f}")
    MSBFS_AVAILABLE = False


def run_msbfs_partition(edge_index: torch.Tensor, num_atoms: int, k: int, device: torch.device) -> Tuple[List[int], Dict[str, float]]:
    """
    Run MS-BFS partition algorithm.
    
    Args:
        edge_index: Edge list tensor of shape [2, num_edges]
        num_atoms: Number of atoms/nodes
        k: Number of partitions
        device: Device to run on
        
    Returns:
        Tuple of (partition_labels, metadata)
    """
    start_time = time.perf_counter()
    
    # Run MS-BFS partition
    partition_id = gpu_multi_seed_bfs_partition(
        edge_index=edge_index,
        num_atoms=num_atoms,
        num_partitions=k,
        device=device
    )
    
    partition_time = time.perf_counter() - start_time
    
    # Convert to list format
    labels = partition_id.cpu().numpy().tolist()
    
    metadata = {
        "partition_time": partition_time,
        "method": "msbfs"
    }
    
    return labels, metadata


def run_pymetis_partition(adjacency: List[List[int]], k: int) -> Tuple[List[int], Dict[str, float]]:
    """
    Run PyMETIS partition algorithm.
    
    Args:
        adjacency: Adjacency list
        k: Number of partitions
        
    Returns:
        Tuple of (partition_labels, metadata)
    """
    start_time = time.perf_counter()
    
    # Run PyMETIS
    labels, _ = run_pymetis(adjacency, k)
    
    partition_time = time.perf_counter() - start_time
    
    metadata = {
        "partition_time": partition_time,
        "method": "pymetis"
    }
    
    return labels, metadata


def convert_edge_index_to_adjacency(edge_index: torch.Tensor, num_atoms: int) -> List[List[int]]:
    """
    Convert edge_index tensor to adjacency list format.
    
    Args:
        edge_index: Edge list tensor of shape [2, num_edges]
        num_atoms: Number of atoms
        
    Returns:
        Adjacency list
    """
    adjacency = [[] for _ in range(num_atoms)]
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    
    # Remove duplicates and sort
    for i in range(num_atoms):
        adjacency[i] = sorted(list(set(adjacency[i])))
    
    return adjacency


def benchmark_partition_methods(
    atoms,
    k_values: List[int] = [2, 4, 8, 16],
    device: torch.device = None,
    cutoff: float = 6.0,
    max_neighbors: int = 300,
    track_memory: bool = False,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Benchmark different partition methods on a given atoms object.
    
    Args:
        atoms: ASE Atoms object
        k_values: List of partition counts to test
        device: Device to run on
        
    Returns:
        Dictionary with results for each method
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Build graph using fairchem GPU
    print(f"  Building graph with fairchem GPU...")
    graph_result, prep_graph_time, graph_build_time, prep_partition_time, memory_stats = build_graph_fairchem_timed(
        atoms=atoms,
        radius=cutoff,
        max_num_neighbors_threshold=max_neighbors,
        enforce_max_neighbors_strictly=False,
        use_gpu=True,
        track_memory=track_memory,
        verbose=verbose
    )
    
    num_atoms, adjacency, edges = graph_result
    
    # Convert to edge_index format for MS-BFS
    edge_list = []
    for i, j in edges:
        edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
    
    print(f"  Graph: {num_atoms} nodes, {len(edges)} edges")
    print(f"  Graph build time: {graph_build_time:.4f}s")
    print(f"  Memory: CPU {memory_stats['peak_cpu_mb']:.1f}MB, GPU {memory_stats['peak_gpu_mb']:.1f}MB")
    
    # Test PyMETIS
    print(f"  Testing PyMETIS...")
    try:
        pymetis_results = {}
        for k in k_values:
            labels, metadata = run_pymetis_partition(adjacency, k)
            
            # Compute metrics
            edge_cut = compute_edge_cut(edges, labels)
            balance = balance_metrics(labels, k)
            
            pymetis_results[k] = {
                'partition_time': metadata['partition_time'],
                'edge_cut': edge_cut,
                'imbalance': balance['imbalance'],
                'partition_sizes': balance['max']
            }
        
        results['pymetis'] = pymetis_results
        print(f"  PyMETIS completed successfully")
        
    except Exception as e:
        print(f"  PyMETIS failed: {e}")
        results['pymetis'] = None
    
    # Test MS-BFS
    if MSBFS_AVAILABLE:
        print(f"  Testing MS-BFS...")
        try:
            msbfs_results = {}
            for k in k_values:
                labels, metadata = run_msbfs_partition(edge_index, num_atoms, k, device)
                
                # Compute metrics using our functions
                edge_cut = msbfs_compute_edge_cut(edge_index, torch.tensor(labels, device=device))
                balance = msbfs_compute_balance_metrics(torch.tensor(labels, device=device), k)
                
                msbfs_results[k] = {
                    'partition_time': metadata['partition_time'],
                    'edge_cut': edge_cut,
                    'imbalance': balance['imbalance'],
                    'partition_sizes': balance['partition_sizes']
                }
            
            results['msbfs'] = msbfs_results
            print(f"  MS-BFS completed successfully")
            
        except Exception as e:
            print(f"  MS-BFS failed: {e}")
            results['msbfs'] = None
    else:
        print(f"  MS-BFS not available, skipping")
        results['msbfs'] = None
    
    # Test SFC (Space-Filling Curve)
    print(f"  Testing SFC...")
    try:
        # Get atomic coordinates for SFC
        coords = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
        
        sfc_results = {}
        for k in k_values:
            # Test both Morton and Hilbert curves
            for curve_type in ["morton", "hilbert"]:
                labels, metadata = run_sfc_partition(coords, num_atoms, k, device, curve_type)
                
                # Compute metrics
                edge_cut = compute_sfc_edge_cut(edge_index, torch.tensor(labels, device=device))
                balance = compute_sfc_balance_metrics(torch.tensor(labels, device=device), k)
                
                method_name = f"sfc_{curve_type}"
                if method_name not in sfc_results:
                    sfc_results[method_name] = {}
                sfc_results[method_name][k] = {
                    'partition_time': metadata['partition_time'],
                    'edge_cut': edge_cut,
                    'imbalance': balance['imbalance'],
                    'partition_sizes': balance['partition_sizes']
                }
        
        results['sfc'] = sfc_results
        print(f"  SFC completed successfully")
        
    except Exception as e:
        print(f"  SFC failed: {e}")
        results['sfc'] = None
    
    # Store graph building info
    results['graph_info'] = {
        'num_atoms': num_atoms,
        'num_edges': len(edges),
        'prep_graph_time': prep_graph_time,
        'graph_build_time': graph_build_time,
        'prep_partition_time': prep_partition_time,
        'total_graph_time': prep_graph_time + graph_build_time + prep_partition_time,
        'peak_cpu_mb': memory_stats['peak_cpu_mb'],
        'peak_gpu_mb': memory_stats['peak_gpu_mb']
    }
    
    return results


def find_test_files(data_dir: str = "data", data_pattern: str = "ch4_*k_splited.xyz") -> List[str]:
    """Find test files in the data directory."""
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found")
        return []
    
    # Look for files matching the pattern
    pattern = os.path.join(data_dir, data_pattern)
    files = glob.glob(pattern)
    
    # Sort by file size (smallest first)
    files.sort(key=lambda x: os.path.getsize(x))
    
    return files


def run_benchmark(
    data_dir: str = "data",
    output_file: str = "results/partition_benchmark.csv",
    k_values: List[int] = [2, 4, 8, 16],
    max_files: Optional[int] = None,
    data_pattern: str = "ch4_*k_splited.xyz",
    cutoff: float = 6.0,
    max_neighbors: int = 300,
    track_memory: bool = False,
    verbose: bool = False,
    max_atoms: Optional[int] = None
):
    """Run the complete benchmark."""
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find test files
    test_files = find_test_files(data_dir, data_pattern)
    if not test_files:
        print(f"No test files found matching pattern: {data_pattern}")
        return
    
    if max_files:
        test_files = test_files[:max_files]
    
    print(f"Found {len(test_files)} test files")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare CSV output
    fieldnames = [
        'filename', 'atom_count', 'method', 'k', 'prep_graph_time_s', 'graph_build_time_s', 
        'prep_partition_time_s', 'total_graph_time_s', 'partition_time_s', 'num_nodes', 
        'num_edges', 'edge_cut', 'imbalance', 'peak_cpu_mb', 'peak_gpu_mb', 'notes'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, filename in enumerate(test_files):
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(test_files)}: {os.path.basename(filename)}")
            print(f"{'='*60}")
            
            try:
                # Load atoms
                atoms = read_xyz_as_atoms(filename)
                num_atoms = len(atoms)
                print(f"Loaded {num_atoms} atoms")
                
                # Skip if too many atoms
                if max_atoms is not None and num_atoms > max_atoms:
                    print(f"  Skipping {os.path.basename(filename)}: {num_atoms} atoms > {max_atoms} limit")
                    # continue
                    break
                
                # Run benchmark
                results = benchmark_partition_methods(
                    atoms, k_values, device, cutoff, max_neighbors, track_memory, verbose
                )
                
                # Write results to CSV
                for method in ['pymetis', 'msbfs', 'sfc']:
                    if results[method] is None:
                        continue
                    
                    # Handle SFC methods (sfc_morton, sfc_hilbert)
                    if method == 'sfc':
                        for sfc_method in results[method]:
                            for k in k_values:
                                if k not in results[method][sfc_method]:
                                    continue
                                
                                method_results = results[method][sfc_method][k]
                                graph_info = results['graph_info']
                                
                                row = {
                                    'filename': os.path.basename(filename),
                                    'atom_count': graph_info['num_atoms'],
                                    'method': sfc_method,
                                    'k': k,
                                    'prep_graph_time_s': graph_info['prep_graph_time'],
                                    'graph_build_time_s': graph_info['graph_build_time'],
                                    'prep_partition_time_s': graph_info['prep_partition_time'],
                                    'total_graph_time_s': graph_info['total_graph_time'],
                                    'partition_time_s': method_results['partition_time'],
                                    'num_nodes': graph_info['num_atoms'],
                                    'num_edges': graph_info['num_edges'],
                                    'edge_cut': method_results['edge_cut'],
                                    'imbalance': method_results['imbalance'],
                                    'peak_cpu_mb': graph_info['peak_cpu_mb'],
                                    'peak_gpu_mb': graph_info['peak_gpu_mb'],
                                    'notes': ''
                                }
                                
                                writer.writerow(row)
                                csvfile.flush()
                    else:
                        # Handle regular methods (pymetis, msbfs)
                        for k in k_values:
                            if k not in results[method]:
                                continue
                            
                            method_results = results[method][k]
                            graph_info = results['graph_info']
                            
                            row = {
                                'filename': os.path.basename(filename),
                                'atom_count': graph_info['num_atoms'],
                                'method': method,
                                'k': k,
                                'prep_graph_time_s': graph_info['prep_graph_time'],
                                'graph_build_time_s': graph_info['graph_build_time'],
                                'prep_partition_time_s': graph_info['prep_partition_time'],
                                'total_graph_time_s': graph_info['total_graph_time'],
                                'partition_time_s': method_results['partition_time'],
                                'num_nodes': graph_info['num_atoms'],
                                'num_edges': graph_info['num_edges'],
                                'edge_cut': method_results['edge_cut'],
                                'imbalance': method_results['imbalance'],
                                'peak_cpu_mb': graph_info['peak_cpu_mb'],
                                'peak_gpu_mb': graph_info['peak_gpu_mb'],
                                'notes': ''
                            }
                            
                            writer.writerow(row)
                            csvfile.flush()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print(f"\nBenchmark completed! Results saved to {output_file}")


def parse_arguments():
    """Parse command line arguments - consistent with test.py"""
    parser = argparse.ArgumentParser(
        description="Partition algorithm benchmark using GPU fairchem graph building",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing .xyz data files")
    parser.add_argument("--data-pattern", type=str, default="ch4_*k_splited.xyz",
                       help="Glob pattern for data files")
    parser.add_argument("--max-atoms", type=int, default=None,
                       help="Maximum number of atoms to process (skip larger files)")
    
    # Graph building parameters
    parser.add_argument("--cutoff", type=float, default=6.0,
                       help="Cutoff radius for graph building")
    parser.add_argument("--k-partitions", type=int, nargs='+', default=[2, 4, 8, 16],
                       help="Number of partitions for algorithms (can specify multiple values)")
    parser.add_argument("--max-neighbors", type=int, default=300,
                       help="Maximum neighbors threshold for fairchem methods")
    parser.add_argument("--track-memory", action="store_true",
                       help="Track peak CPU and GPU memory usage during graph building")
    
    # Output parameters
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output CSV filename (auto-generated if not specified)")
    
    # Verbose output
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # Generate output filename if not specified
    if args.output_file is None:
        k_str = "_".join(map(str, args.k_partitions))
        args.output_file = f"partition_benchmark_cutoff{args.cutoff}_k{k_str}_maxnb{args.max_neighbors}_fairchem_gpu.csv"
    
    output_path = os.path.join(args.results_dir, args.output_file)
    
    print("Partition Algorithm Benchmark")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Data pattern: {args.data_pattern}")
    print(f"Output file: {output_path}")
    print(f"K partitions: {args.k_partitions}")
    print(f"Cutoff: {args.cutoff}")
    print(f"Max neighbors: {args.max_neighbors}")
    print(f"Track memory: {args.track_memory}")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)
    
    # Run the benchmark
    run_benchmark(
        data_dir=args.data_dir,
        output_file=output_path,
        k_values=args.k_partitions,
        max_files=None,  # Process all files matching pattern
        data_pattern=args.data_pattern,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        track_memory=args.track_memory,
        verbose=args.verbose,
        max_atoms=args.max_atoms
    )
    
    print("\nBenchmark completed!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
