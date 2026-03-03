import os
import time
import hydra
import logging
import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger(__name__)

# Add project root to sys.path if running as script
import sys
if __name__ == "__main__" or "dist_md" in __file__:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)



import ase
import ase.io
import ase.md
import ase.md.velocitydistribution
import ase.md.langevin
from ase import units
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from baseline.local_fairchem import pretrained_mlip, FAIRChemCalculator
from baseline.local_fairchem.units.mlip_unit.api.inference import inference_settings_turbo
from baseline.local_fairchem.graph.radius_graph_pbc import radius_graph_pbc_v2
from baseline.distributed_model import DistributedESCNMDBackbone
import torch._dynamo
import torch._inductor
from baseline.sfc import sfc_partition, distributed_sfc_partition, distributed_kdtree_partition, distributed_pca_slab_partition
from ase.calculators.calculator import Calculator, all_changes

# Import refactored modules
from baseline.profiler import CUDAEventProfiler, StepProfiler, _ProfilerContext
from baseline.halo_exchange import (
    HaloExchanger,
    RingExchanger,
    build_halo_maps,
    distributed_build_halo_maps,
    distributed_build_ring_halo_maps,
    _build_edges_cutoff_fairchem_raw
)
from baseline.distributed_calculator import DistributedCalculator


# Configure torch inductor
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
# Force Inductor to use g++ instead of Cray Wrapper (CC) which causes linker errors
torch._inductor.config.cpp.cxx = ["g++"]

# -----------------------------------------------------------------------------
# Configuration Classes
# -----------------------------------------------------------------------------

@dataclass
class MDConfig:
    cell_size: float = 30.0
    temperature: float = 850.0
    timestep: float = 0.1
    friction: float = 0.01
    total_steps: int = 1000
    traj_interval: int = 100
    log_interval: int = 100
    steps_per_episode: int = 100

@dataclass
class UMAConfig:
    model: str = "uma-s-1p1" # ["uma-m-1p1", "uma-s-1p1"] or path to .pt Student model
    activation_checkpointing: bool = True
    compile: bool = False
    wigner_cuda: bool = True
    use_triton_wigner: bool = True  # Enable optimized Triton kernel by default
    merge_mole: bool = True
    cache_path: Optional[str] = None
    dynamo_cache_size: int = 64

@dataclass
class MACEConfig:
    model: str = "large"
    default_dtype: str = "float32"

@dataclass
class ProfilerConfig:
    enabled: bool = False
    log_dir: str = "./profiler_logs"
    torch_profiler: bool = False  # Enable PyTorch profiler for deep model analysis
    torch_profiler_steps: int = 5  # Number of steps to profile (skip first 2 as warmup)
    measure_bw: bool = False  # Measure NCCL all_to_all bandwidth with barrier+CUDA events

@dataclass
class DistConfig:
    starter: str = "slurm" # ["slurm", "torchrun", "python"]
    backend: str = "nccl"
    cutoff: float = 6.0
    ghost_cutoff_buffer: float = 0.5  # Extra distance for ghost node selection to avoid FP edge cases
    partition_algorithm: str = "sfc_morton" # ["sfc_morton", "sfc_hilbert", "simple", "kdtree", "pca_slab"]
    ring_exchange: bool = False  # Use ring topology (isend/irecv with rank n-1, n+1) instead of all_to_all
    k_partition: Optional[int] = None
    rebalance_interval: int = 100
    rebalance_threshold: float = 0.2
    use_gid: bool = True
    save_partition: bool = True
    save_timeseries: bool = True
    merge_results: bool = True
    gnn_layers: int = 1  # Number of GNN layers for halo depth
    use_distributed_model: bool = True # Use distributed model with direct feature fetching
    overlap: bool = False  # Enable computation-communication overlap (experimental)
    hierarchical: bool = False  # Enable hierarchical aggregation (Area 2 / PXN)
    edge_weighted_partition: bool = False  # Partition by edge count for balanced compute
    per_node_partition: bool = False  # Partition by node (not GPU) for NVSHMEM architecture
    use_nvshmem: bool = False  # Use NVSHMEM for inter-node feature exchange (requires per_node_partition)
    differentiable_exchange: bool = True  # Differentiable feature exchange (correct ghost gradients, adds L backward all_to_all)

@dataclass
class AppConfig:
    input_file: str = ""
    output_dir: str = "./results"
    prefix: str = "md"
    trial: int = 0
    save_config: bool = True
    md: MDConfig = field(default_factory=MDConfig)
    backend: str = "uma"  # ["mace", "uma"]
    uma: UMAConfig = field(default_factory=UMAConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"[{self.name}] took {end_time - self.start_time:.4f}s")


def dist_is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank0_print(*args, **kwargs):
    if not dist_is_enabled() or dist.get_rank() == 0:
        print(*args, **kwargs)


def dist_init_from_slurm(backend: Optional[str] = None) -> Tuple[int, int, int]:
    """Initialize torch.distributed using SLURM env if present."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank))))

    if world_size <= 1:
        return 0, 1, 0

    if not dist.is_initialized():
        root_dir = os.environ.get("DIST_INIT_DIR", os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        init_dir = os.path.join(root_dir, ".dist_init")
        os.makedirs(init_dir, exist_ok=True)
        init_file = os.path.join(init_dir, f"pg_{job_id}.init")
        init_method = f"file://{init_file}"
        # Force set device to LOCAL_RANK *before* init_process_group to prevent NCCL device mismatch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            print(f"Rank (local) {local_rank}: Set device to cuda:{local_rank}", flush=True)

        # Prepare kwargs
        init_kwargs = {
            "backend": backend,
            "init_method": init_method,
            "rank": rank,
            "world_size": world_size
        }
        # Add device_id for NCCL backend if supported (PyTorch 2.1+) to avoid "device unknown" warning
        if backend == "nccl" and torch.cuda.is_available():
             init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(**init_kwargs)

    return rank, world_size, local_rank

def dist_init_from_torchrun(backend: Optional[str] = None) -> Tuple[int, int, int]:
    """Initialize torch.distributed using Torchrun env."""
    # Force set device to LOCAL_RANK
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"Rank (local) {local_rank}: Set device to cuda:{local_rank}", flush=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return 0, 1, 0

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, world_size, local_rank

def _broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from src rank, handling device/backend compatibility."""
    if not dist_is_enabled():
        return t
    
    backend = dist.get_backend()
    if backend == "gloo" and t.device.type != 'cpu':
        # Gloo requires CPU tensors
        t_cpu = t.cpu()

        dist.broadcast(t_cpu, src=src)
        t.copy_(t_cpu)
    elif backend == "nccl" and t.device.type == 'cpu':
        # NCCL requires CUDA tensors
        device = torch.device("cuda", torch.cuda.current_device())
        t_cuda = t.to(device)

        dist.broadcast(t_cuda, src=src)
        t.copy_(t_cuda.cpu())
    else:
        # Backend and device are compatible

        dist.broadcast(t, src=src)
    return t

def _concat_per_rank_and_broadcast(fields_per_rank: Dict[str, List[torch.Tensor]], dtype_map: Dict[str, torch.dtype]) -> Dict[str, torch.Tensor]:
    """Rank0 concatenates each field across ranks and broadcasts both splits and data to all ranks."""
    assert dist_is_enabled(), "Distributed must be initialized"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Infer device from the first tensor in fields_per_rank if available
    # Or default to current device
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        
    # Step 1: broadcast sizes per field per rank
    splits: Dict[str, torch.Tensor] = {}
    for name, dtype in dtype_map.items():
        # print(f"Rank {rank}: Broadcasting size for {name}")
        if rank == 0:
            sizes = torch.tensor([int(x.size(0)) for x in fields_per_rank[name]], dtype=torch.int64, device=device)
        else:
            sizes = torch.empty(world_size, dtype=torch.int64, device=device)
        sizes = _broadcast_tensor(sizes, src=0)
        # print(f"Rank {rank}: Size for {name} received: {sizes}")
        splits[name] = sizes

    # Step 2: rank0 concatenates tensors and broadcasts full payloads
    payload: Dict[str, torch.Tensor] = {}
    for name, dtype in dtype_map.items():
        if rank == 0:
            # Concatenate all ranks' data for this field
            full_tensor = torch.cat(fields_per_rank[name], dim=0)
        else:
            # Prepare buffer
            total_size = splits[name].sum().item()
            shape_tail = tuple(int(x) for x in fields_per_rank[name][0].shape[1:]) if name in fields_per_rank else (3,)
            shape = (total_size,) + shape_tail
            full_tensor = torch.empty(shape, dtype=dtype, device=device)
            
        full_tensor = _broadcast_tensor(full_tensor, src=0)
        payload[name] = full_tensor
    return payload | {"__splits__": splits}

def _build_local_atoms_from_payload(payload: Dict[str, torch.Tensor], cell: np.ndarray, pbc: Tuple[bool, bool, bool]) -> ase.Atoms:
    rank = dist.get_rank() if dist_is_enabled() else 0
    sizes = payload["__splits__"]["pos"]
    pos_all = payload["pos"].detach().cpu().numpy()
    vel_all = payload["vel"].detach().cpu().numpy()
    Z_all = payload["Z"].detach().cpu().numpy().astype(np.int32)
    gid_all = payload.get("gid", None)
    gid_all = gid_all.detach().cpu().numpy().astype(np.int64) if gid_all is not None else None

    # Slice my segment
    start = int(sizes[:rank].sum().item()) if dist_is_enabled() else 0
    end = int(sizes[: rank + 1].sum().item()) if dist_is_enabled() else len(Z_all)
    pos = pos_all[start:end]
    vel = vel_all[start:end]
    Z = Z_all[start:end]
    gid = gid_all[start:end] if gid_all is not None else None

    a = ase.Atoms(numbers=Z.tolist(), positions=pos)
    a.set_cell(cell)
    a.set_pbc(list(pbc))
    a.set_velocities(vel)
    if gid is not None:
        a.new_array("global_id", gid)

    return a

def _gather_all_atoms_to_rank0(a: ase.Atoms) -> Optional[Dict[str, np.ndarray]]:
    """Gather positions/velocities/numbers from all ranks to rank0 using GPU tensors."""
    world_size = dist.get_world_size() if dist_is_enabled() else 1
    rank = dist.get_rank() if dist_is_enabled() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not dist_is_enabled() or world_size == 1:
        return {
            "pos": a.get_positions().astype(np.float32),
            "vel": a.get_velocities().astype(np.float32),
            "Z": a.get_atomic_numbers().astype(np.int32),
            "gid": a.get_array("global_id").astype(np.int64) if "global_id" in a.arrays else None
        }
    
    # Convert to GPU tensors
    local_pos = torch.tensor(a.get_positions(), dtype=torch.float32, device=device)
    local_vel = torch.tensor(a.get_velocities(), dtype=torch.float32, device=device)
    local_Z = torch.tensor(a.get_atomic_numbers(), dtype=torch.int32, device=device)
    has_gid = "global_id" in a.arrays
    local_gid = torch.tensor(a.get_array("global_id"), dtype=torch.int64, device=device) if has_gid else None
    
    # Exchange counts first
    local_count = torch.tensor([len(a)], dtype=torch.int64, device=device)
    all_counts = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]

    dist.all_gather(all_counts, local_count)
    counts = [c.item() for c in all_counts]
    total = sum(counts)
    
    # Gather positions
    pos_list = [torch.zeros((c, 3), dtype=torch.float32, device=device) for c in counts]

    dist.all_gather(pos_list, local_pos)
    
    # Gather velocities
    vel_list = [torch.zeros((c, 3), dtype=torch.float32, device=device) for c in counts]

    dist.all_gather(vel_list, local_vel)
    
    # Gather atomic numbers
    Z_list = [torch.zeros(c, dtype=torch.int32, device=device) for c in counts]

    dist.all_gather(Z_list, local_Z)
    
    # Gather global_id if present
    gid_list = None
    if has_gid:
        gid_list = [torch.zeros(c, dtype=torch.int64, device=device) for c in counts]

        dist.all_gather(gid_list, local_gid)
    
    if rank != 0:
        return None
    
    result = {
        "pos": torch.cat(pos_list, dim=0).cpu().numpy(),
        "vel": torch.cat(vel_list, dim=0).cpu().numpy(),
        "Z": torch.cat(Z_list, dim=0).cpu().numpy(),
    }
    if gid_list is not None:
        result["gid"] = torch.cat(gid_list, dim=0).cpu().numpy()
    
    return result


def do_partition_and_redistribute(
    local_pos: torch.Tensor,
    local_vel: torch.Tensor,
    local_Z: torch.Tensor,
    local_gid: Optional[torch.Tensor],
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    ghost_cutoff_buffer: float,
    partition_algorithm: str,
    rank: int,
    world_size: int,
    device: torch.device,
    profiler: 'CUDAEventProfiler' = None,
    edge_weighted: bool = True,
    ring_exchange: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
    """
    Unified partition and redistribution logic for both initial partition and repartition.
    
    Steps:
    1. distributed_sfc_partition to determine new partition
    2. all_to_all redistribute atoms to new owners
    3. distributed_build_halo_maps to build communication patterns (each rank independently)
    
    Args:
        edge_weighted: If True, partition by edge count for balanced compute workload.
    
    Returns:
        new_pos, new_vel, new_Z, new_gid, send_map, recv_ghost_indices, interior_mask
    """
    N_local = local_pos.size(0)
    has_gid = local_gid is not None
    
    # Step 0: Compute local edge counts if edge-weighted partitioning is enabled
    local_edge_counts = None
    if edge_weighted and partition_algorithm != "simple":
        # Build local graph to count edges per atom
        # We need to all_gather positions first to build consistent graph
        local_count = torch.tensor([N_local], dtype=torch.long, device=device)
        all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        counts = [c.item() for c in all_counts]
        
        # All-gather positions
        all_pos_list = [torch.zeros((c, 3), dtype=torch.float32, device=device) for c in counts]
        dist.all_gather(all_pos_list, local_pos.contiguous())
        full_pos = torch.cat(all_pos_list, dim=0)
        
        # Build edges on full graph
        edge_index, _ = _build_edges_cutoff_fairchem_raw(full_pos, cutoff, cell, pbc)
        edge_index = edge_index.to(device)
        
        # Count edges per atom (dst-based, as compute is on dst atoms)
        # Only count edges where dst is a local atom of this rank
        my_start = sum(counts[:rank])
        my_end = my_start + N_local
        
        # Mask for edges where dst is in my local range
        dst_is_local = (edge_index[1] >= my_start) & (edge_index[1] < my_end)
        my_dst_global = edge_index[1][dst_is_local]
        my_dst_local = my_dst_global - my_start
        
        # Count edges per local atom
        local_edge_counts = torch.zeros(N_local, dtype=torch.long, device=device)
        local_edge_counts.scatter_add_(0, my_dst_local, torch.ones(my_dst_local.shape[0], dtype=torch.long, device=device))
        
        if rank == 0:
            total_edges = edge_index.shape[1]
            edge_min, edge_max = local_edge_counts.min().item(), local_edge_counts.max().item()
            edge_mean = local_edge_counts.float().mean().item()
            print(f"Edge-weighted partition: {total_edges} total edges, per-atom edges min={edge_min}, max={edge_max}, mean={edge_mean:.1f}")
    
    # Step 1: Distributed SFC partition
    if profiler:
        profiler.start("sfc_partition")
    
    if partition_algorithm == "simple":
        # Simple partition: keep original assignment
        local_partition_id = torch.full((N_local,), rank, dtype=torch.long, device=device)
        # For simple partition, counts are straightforward
        send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        send_counts[rank] = N_local
        recv_counts = send_counts.clone()  # Each rank keeps its own
    elif "pca_slab" in partition_algorithm:
        local_partition_id, metadata = distributed_pca_slab_partition(
            local_coords=local_pos,
            num_partitions=world_size,
            device=device,
            local_edge_counts=local_edge_counts,
        )
        send_counts_matrix = metadata["send_counts_matrix"]
        send_counts = send_counts_matrix[rank]
        recv_counts = send_counts_matrix[:, rank]
        if rank == 0:
            print(f"PCA slab partition: dir={[f'{v:.3f}' for v in metadata['principal_dir']]}, "
                  f"min_slab_thickness={metadata['min_slab_thickness']:.2f}, "
                  f"cutoff={cutoff:.2f}")
            if metadata['min_slab_thickness'] < cutoff:
                print(f"  WARNING: min slab thickness ({metadata['min_slab_thickness']:.2f}) < cutoff ({cutoff:.2f}), "
                      f"ring exchange may miss some neighbors!")
    elif "kdtree" in partition_algorithm:
        local_partition_id, metadata = distributed_kdtree_partition(
            local_coords=local_pos,
            num_partitions=world_size,
            device=device,
            local_edge_counts=local_edge_counts,
        )
        send_counts_matrix = metadata["send_counts_matrix"]
        send_counts = send_counts_matrix[rank]
        recv_counts = send_counts_matrix[:, rank]
    else:
        curve_type = "hilbert" if "hilbert" in partition_algorithm else "morton"
        local_partition_id, metadata = distributed_sfc_partition(
            local_coords=local_pos,
            num_partitions=world_size,
            curve_type=curve_type,
            device=device,
            local_edge_counts=local_edge_counts  # Pass edge counts for weighted partition
        )

        # Use pre-computed send_counts_matrix (eliminates all_to_all for counts!)
        send_counts_matrix = metadata["send_counts_matrix"]
        send_counts = send_counts_matrix[rank]  # Counts I send to each rank
        recv_counts = send_counts_matrix[:, rank]  # Counts I receive from each rank
    
    if profiler:
        profiler.stop("sfc_partition")
    
    # Step 2: Atom redistribution via all-to-all
    if profiler:
        profiler.start("atom_redistribute")
    
    send_counts_list = [int(c) for c in send_counts.tolist()]
    recv_counts_list = [int(c) for c in recv_counts.tolist()]
    total_recv = sum(recv_counts_list)
    
    # Sort local atoms by target rank for packing
    sort_indices = torch.argsort(local_partition_id)
    sorted_pos = local_pos[sort_indices]
    sorted_vel = local_vel[sort_indices]
    sorted_Z = local_Z[sort_indices]
    sorted_gid = local_gid[sort_indices] if has_gid else None
    
    # Pack data: [pos(3) + vel(3) + Z(1) + gid(1)] = 8 floats per atom
    packed_send = torch.cat([
        sorted_pos,
        sorted_vel,
        sorted_Z.float().unsqueeze(1),
        (sorted_gid.float().unsqueeze(1) if has_gid else torch.zeros(N_local, 1, device=device))
    ], dim=1)  # [N_local, 8]
    
    # Prepare receive buffer
    packed_recv = torch.zeros((total_recv, 8), dtype=torch.float32, device=device)
    
    # All-to-all exchange

    dist.all_to_all_single(
        packed_recv, packed_send,
        output_split_sizes=recv_counts_list,
        input_split_sizes=send_counts_list
    )
    
    # Unpack received data
    new_pos = packed_recv[:, 0:3].contiguous()
    new_vel = packed_recv[:, 3:6].contiguous()
    new_Z = packed_recv[:, 6].long()
    new_gid = packed_recv[:, 7].long() if has_gid else None
    
    if profiler:
        profiler.stop("atom_redistribute")
    
    # Step 3: Build halo maps using distributed version
    if profiler:
        profiler.start("build_halo_maps")
    
    # Each rank computes its own maps independently (more scalable)
    ghost_cutoff = cutoff + ghost_cutoff_buffer

    if ring_exchange:
        send_map, recv_ghost_indices, interior_mask = distributed_build_ring_halo_maps(
            local_pos=new_pos,
            cutoff=ghost_cutoff,
            cell=cell,
            pbc=pbc,
            rank=rank,
            world_size=world_size,
            device=device
        )
    else:
        send_map, recv_ghost_indices, interior_mask = distributed_build_halo_maps(
            local_pos=new_pos,
            cutoff=ghost_cutoff,
            cell=cell,
            pbc=pbc,
            rank=rank,
            world_size=world_size,
            device=device
        )
    
    if profiler:
        profiler.stop("build_halo_maps")
    
    return new_pos, new_vel, new_Z, new_gid, send_map, recv_ghost_indices, interior_mask


def do_per_node_partition_and_redistribute(
    local_pos: torch.Tensor,
    local_vel: torch.Tensor,
    local_Z: torch.Tensor,
    local_gid: Optional[torch.Tensor],
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    ghost_cutoff_buffer: float,
    partition_algorithm: str,
    rank: int,
    world_size: int,
    device: torch.device,
    profiler: 'CUDAEventProfiler' = None,
    edge_weighted: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], 
           torch.Tensor, Dict]:
    """
    Per-node partition and redistribution (Strategy B).
    
    SFC partitions into num_nodes (not world_size) partitions.
    Atoms are redistributed to correct nodes, then evenly split within each node.
    Halo maps only track inter-node ghosts.
    
    Returns:
        new_pos, new_vel, new_Z, new_gid, 
        send_map (inter-node only), recv_ghost_indices (inter-node only),
        interior_mask (node-level), per_node_info dict
    """
    gpus_per_node = torch.cuda.device_count()  # 4 on Delta
    num_nodes = world_size // gpus_per_node
    node_id = rank // gpus_per_node
    local_gpu_id = rank % gpus_per_node
    N_local = local_pos.size(0)
    has_gid = local_gid is not None
    
    if num_nodes < 2:
        # Single node: do regular partition for atom redistribution, but return
        # EMPTY comm_maps because per-node mode uses all_gather for intra-node atoms
        # (no inter-node ghosts in single-node case)
        result = do_partition_and_redistribute(
            local_pos, local_vel, local_Z, local_gid,
            cell, pbc, cutoff, ghost_cutoff_buffer,
            partition_algorithm, rank, world_size, device, profiler, edge_weighted,
            ring_exchange=False,
        )

        # Unpack result to get redistributed atoms (ignore comm_maps from regular partition)
        new_pos, new_vel, new_Z, new_gid, _, _, interior_mask = result

        # In single-node per-node mode: no inter-node ghosts, so empty comm_maps
        # All atoms within the node are shared via all_gather, not ghost exchange
        empty_send_map = {}
        empty_recv_indices = {}

        # For single-node, all atoms are "interior" since there are no inter-node boundaries
        # Each GPU's atoms are interior to the node
        interior_mask = torch.ones(new_pos.size(0), dtype=torch.bool, device=device)

        # Calculate node_atom_counts for single node case using NEW positions (after redistribution)
        local_count = torch.tensor([new_pos.size(0)], dtype=torch.long, device=device)
        all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)

        per_node_info = {
            'gpus_per_node': gpus_per_node,
            'num_nodes': 1,
            'node_id': 0,
            'local_gpu_id': local_gpu_id,
            'node_atom_counts': [c.item() for c in all_counts],
        }

        return new_pos, new_vel, new_Z, new_gid, empty_send_map, empty_recv_indices, interior_mask, per_node_info
    
    # ======================================================================
    # Step 1: SFC partition into num_nodes partitions
    # ======================================================================
    if profiler:
        profiler.start("sfc_partition")
    
    # Each GPU calls distributed partition with num_nodes partitions
    # The partition_id now represents which NODE the atom should go to
    if "kdtree" in partition_algorithm:
        local_node_partition_id, metadata = distributed_kdtree_partition(
            local_coords=local_pos,
            num_partitions=num_nodes,
            device=device,
            local_edge_counts=None,
        )
    else:
        curve_type = "hilbert" if "hilbert" in partition_algorithm else "morton"
        local_node_partition_id, metadata = distributed_sfc_partition(
            local_coords=local_pos,
            num_partitions=num_nodes,
            curve_type=curve_type,
            device=device,
            local_edge_counts=None,  # Skip edge weighting for now
        )
    
    # Convert node partition_id to GPU rank: each atom goes to the first GPU of target node
    # Then we'll scatter within node
    # For all_to_all, target_rank = node_id * gpus_per_node + local_gpu_id
    # But all_to_all needs per-rank split, so we route all atoms for a node to that node's GPU 
    # with the same local_gpu_id as sender (round-robin load balance across node GPUs)
    local_target_rank = local_node_partition_id * gpus_per_node + local_gpu_id
    
    # Compute send counts for all_to_all (per rank)
    send_counts = torch.bincount(local_target_rank.long(), minlength=world_size).to(dtype=torch.long)
    
    # Transpose send_counts via all_to_all to get recv_counts
    recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)

    dist.all_to_all_single(recv_counts, send_counts)
    
    if profiler:
        profiler.stop("sfc_partition")
    
    # ======================================================================
    # Step 2: Atom redistribution via all-to-all
    # ======================================================================
    if profiler:
        profiler.start("atom_redistribute")
    
    send_counts_list = [int(c) for c in send_counts.tolist()]
    recv_counts_list = [int(c) for c in recv_counts.tolist()]
    total_recv = sum(recv_counts_list)
    
    # Sort by target rank
    sort_indices = torch.argsort(local_target_rank)
    sorted_pos = local_pos[sort_indices]
    sorted_vel = local_vel[sort_indices]
    sorted_Z = local_Z[sort_indices]
    sorted_gid = local_gid[sort_indices] if has_gid else None
    
    # Pack: [pos(3) + vel(3) + Z(1) + gid(1)] = 8 floats per atom
    packed_send = torch.cat([
        sorted_pos,
        sorted_vel,
        sorted_Z.float().unsqueeze(1),
        (sorted_gid.float().unsqueeze(1) if has_gid else torch.zeros(N_local, 1, device=device))
    ], dim=1)
    
    packed_recv = torch.zeros((total_recv, 8), dtype=torch.float32, device=device)
    

    dist.all_to_all_single(
        packed_recv, packed_send,
        output_split_sizes=recv_counts_list,
        input_split_sizes=send_counts_list
    )
    
    new_pos = packed_recv[:, 0:3].contiguous()
    new_vel = packed_recv[:, 3:6].contiguous()
    new_Z = packed_recv[:, 6].long()
    new_gid = packed_recv[:, 7].long() if has_gid else None
    
    if profiler:
        profiler.stop("atom_redistribute")
    
    # ======================================================================
    # Step 3: Build node-aware halo maps (only inter-node ghosts)
    # ======================================================================
    if profiler:
        profiler.start("build_halo_maps")
    
    ghost_cutoff = cutoff + ghost_cutoff_buffer
    
    send_map, recv_ghost_indices, interior_mask = distributed_build_halo_maps(
        local_pos=new_pos,
        cutoff=ghost_cutoff,
        cell=cell,
        pbc=pbc,
        rank=rank,
        world_size=world_size,
        device=device,
        per_node_partition=True,
        gpus_per_node=gpus_per_node
    )
    
    if profiler:
        profiler.stop("build_halo_maps")
    
    # ======================================================================
    # Step 4: Compute node atom counts (needed for intra-node all_gather)
    # ======================================================================
    # Gather atom counts from all GPUs in this node
    my_count = torch.tensor([new_pos.size(0)], dtype=torch.long, device=device)
    # Use all_gather on the global group and filter to get per-node counts
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]

    dist.all_gather(all_counts, my_count)
    
    node_start_rank = node_id * gpus_per_node
    node_atom_counts = [all_counts[node_start_rank + i].item() for i in range(gpus_per_node)]
    
    per_node_info = {
        'gpus_per_node': gpus_per_node,
        'num_nodes': num_nodes,
        'node_id': node_id,
        'local_gpu_id': local_gpu_id,
        'node_atom_counts': node_atom_counts,  # [N_gpu0, N_gpu1, N_gpu2, N_gpu3]
        'num_local_on_gpu': new_pos.size(0),
    }
    
    if rank == 0:
        total_node_atoms = sum(node_atom_counts)
        n_inter_ghosts = sum(len(v) for v in recv_ghost_indices.values())
        print(f"Per-node partition: {num_nodes} nodes, {gpus_per_node} GPUs/node")
        print(f"  Node 0: {total_node_atoms} atoms, {n_inter_ghosts} inter-node ghosts")
        print(f"  GPU atom counts: {node_atom_counts}")
        int_count = interior_mask.sum().item()
        print(f"  Interior atoms (node-level): {int_count}/{new_pos.size(0)} ({100*int_count/new_pos.size(0):.1f}%)")
    
    return new_pos, new_vel, new_Z, new_gid, send_map, recv_ghost_indices, interior_mask, per_node_info


# -----------------------------------------------------------------------------
# Main Distributed Simulation Logic
# -----------------------------------------------------------------------------

def run_md_simulation_distributed(calc, cfg: AppConfig, rank: int, world_size: int, local_rank: int) -> None:
    # Prepare output naming
    output_dir = os.path.join(cfg.output_dir, cfg.prefix)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, f"{rank}.xyz")
    md_log_path = os.path.join(output_dir, f"{rank}.log")
    
    cell_size = cfg.md.cell_size
    temperature_K = cfg.md.temperature
    timestep = cfg.md.timestep * units.fs
    friction = cfg.md.friction / units.fs
    total_steps = cfg.md.total_steps
    traj_interval = cfg.md.traj_interval
    log_interval = cfg.md.log_interval
    cutoff = cfg.dist.cutoff
    rebalance_interval = cfg.dist.rebalance_interval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === DISTRIBUTED INITIAL LOADING ===
    # All ranks read the xyz file and each loads a portion of atoms
    # Then use distributed_sfc_partition for consistent behavior with rebalance
    
    # All ranks read the full file to get metadata
    full_atoms = ase.io.read(cfg.input_file)
    n_atoms_total = len(full_atoms)
    
    # Set cell if needed
    current_cell_lengths = full_atoms.get_cell().lengths()
    if not (current_cell_lengths > 0).all() and cell_size > 0:
        full_atoms.set_cell([cell_size] * 3)
    full_atoms.set_pbc([True] * 3)
    
    # Get cell and pbc (all ranks have this)
    cell_np = np.asarray(full_atoms.get_cell().array, dtype=np.float32)
    pbc_tuple = tuple(bool(x) for x in full_atoms.get_pbc())
    cell = torch.from_numpy(cell_np).to(device)
    pbc = torch.tensor([int(x) for x in pbc_tuple], dtype=torch.int64).to(device)
    
    # Each rank loads a portion of atoms (initial simple split before SFC)
    atoms_per_rank = n_atoms_total // world_size
    start_idx = rank * atoms_per_rank
    end_idx = (rank + 1) * atoms_per_rank if rank < world_size - 1 else n_atoms_total
    
    # Extract local portion
    positions = full_atoms.get_positions().astype(np.float32)[start_idx:end_idx]
    numbers = full_atoms.get_atomic_numbers().astype(np.int32)[start_idx:end_idx]
    
    # Initialize velocities on rank 0, then broadcast need for consistency
    if rank == 0:
        # Generate velocities for all atoms
        ase.md.velocitydistribution.MaxwellBoltzmannDistribution(full_atoms, temperature_K=temperature_K)
        all_velocities = full_atoms.get_velocities().astype(np.float32)
    else:
        all_velocities = np.zeros((n_atoms_total, 3), dtype=np.float32)
    
    # Broadcast velocities from rank 0 for consistency
    if world_size > 1:
        all_velocities_tensor = torch.from_numpy(all_velocities).to(device)
        _broadcast_tensor(all_velocities_tensor, src=0)
        velocities = all_velocities_tensor[start_idx:end_idx].cpu().numpy()
    else:
        velocities = all_velocities[start_idx:end_idx]
    
    # Prepare local data on device
    local_pos = torch.from_numpy(positions).to(device)
    local_vel = torch.from_numpy(velocities).to(device)
    local_Z = torch.from_numpy(numbers).to(device)
    N_local = local_pos.size(0)
    
    if cfg.dist.use_gid:
        # Global IDs for this rank's portion
        local_gid = torch.arange(start_idx, end_idx, dtype=torch.int64, device=device)
    else:
        local_gid = None
    
    # === DATA PARALLEL PROCESS GROUPS ===
    # Setup process groups early if per_node_partition is enabled
    intra_group = None
    per_node_params = None
    
    if cfg.dist.per_node_partition and world_size > 1 and dist_is_enabled():
        gpus_per_node = torch.cuda.device_count()
        num_nodes = world_size // gpus_per_node
        node_id = rank // gpus_per_node
        
        # Create intra-node groups
        for n in range(num_nodes):
            ranks = list(range(n * gpus_per_node, (n + 1) * gpus_per_node))
            g = dist.new_group(ranks)
            if rank in ranks:
                intra_group = g
        
        rank0_print(f"Per-node partitioning enabled: {num_nodes} nodes, {gpus_per_node} GPUs/node")

    # === DISTRIBUTED PARTITION AND REDISTRIBUTE ===
    rank0_print(f"Partitioning into {world_size} ranks using {cfg.dist.partition_algorithm} (distributed)...")
    
    if world_size > 1:
        if cfg.dist.per_node_partition:
             (local_pos, local_vel, local_Z, local_gid, 
              my_send_map, my_recv_indices, my_interior_mask, per_node_info) = do_per_node_partition_and_redistribute(
                local_pos=local_pos,
                local_vel=local_vel,
                local_Z=local_Z,
                local_gid=local_gid,
                cell=cell,
                pbc=pbc,
                cutoff=cfg.dist.cutoff,
                ghost_cutoff_buffer=cfg.dist.ghost_cutoff_buffer,
                partition_algorithm=cfg.dist.partition_algorithm,
                rank=rank,
                world_size=world_size,
                device=device,
                edge_weighted=cfg.dist.edge_weighted_partition
            )
             per_node_params = {
                 'intra_group': intra_group,
                 'use_nvshmem': cfg.dist.use_nvshmem,
                 **per_node_info
             }
        else:
            (local_pos, local_vel, local_Z, local_gid,
             my_send_map, my_recv_indices, my_interior_mask) = do_partition_and_redistribute(
                local_pos=local_pos,
                local_vel=local_vel,
                local_Z=local_Z,
                local_gid=local_gid,
                cell=cell,
                pbc=pbc,
                cutoff=cfg.dist.cutoff,
                ghost_cutoff_buffer=cfg.dist.ghost_cutoff_buffer,
                partition_algorithm=cfg.dist.partition_algorithm,
                rank=rank,
                world_size=world_size,
                device=device,
                edge_weighted=cfg.dist.edge_weighted_partition,
                ring_exchange=cfg.dist.ring_exchange,
            )

        N_local = local_pos.size(0)
        # Use simple sum for logging (interior mask logic differs per mode)
        print(f"  Rank {rank}: {my_interior_mask.sum().item()}/{N_local} interior atoms ({100*my_interior_mask.sum().item()/N_local:.1f}%)")
    else:
        my_send_map = {}
        my_recv_indices = {}
        my_interior_mask = None

    cell_np = cell.cpu().numpy()
    pbc_tuple = tuple(bool(int(x)) for x in pbc.cpu().numpy().tolist())

    # Build local_atoms from distributed data
    local_atoms = ase.Atoms(
        numbers=local_Z.cpu().numpy().tolist(),
        positions=local_pos.cpu().numpy()
    )
    local_atoms.set_cell(cell_np)
    local_atoms.set_pbc(list(pbc_tuple))
    local_atoms.set_velocities(local_vel.cpu().numpy())
    if cfg.dist.use_gid and local_gid is not None:
        local_atoms.new_array("global_id", local_gid.cpu().numpy())
    
    # Initialize charge and spin for OMOL task
    local_atoms.info["charge"] = 0
    local_atoms.info["spin"] = 1

    # Wrap calculator with DistributedCalculator (even for single rank for consistent profiling)
    
    # === Create hierarchical process groups if requested ===
    hierarchical_params = None
    if cfg.dist.hierarchical and world_size > 1 and dist_is_enabled():
        local_world_size = torch.cuda.device_count()  # 4 on Delta
        num_nodes = world_size // local_world_size
        if num_nodes > 1:
            rank0_print(f"Creating hierarchical process groups: {num_nodes} nodes, {local_world_size} GPUs/node")
            # Create intra-node groups
            intra_group = None
            for n in range(num_nodes):
                ranks = list(range(n * local_world_size, (n + 1) * local_world_size))
                g = dist.new_group(ranks)
                if rank in ranks:
                    intra_group = g
            # Create inter-node group (master = last rank on each node)
            master_ranks = [(n + 1) * local_world_size - 1 for n in range(num_nodes)]
            inter_group = dist.new_group(master_ranks)
            
            hierarchical_params = {
                'intra_group': intra_group,
                'inter_group': inter_group,
                'local_rank': local_rank,
                'local_world_size': local_world_size,
            }
            rank0_print(f"  Master ranks: {master_ranks}, this rank is {'MASTER' if rank in master_ranks else 'worker'}")
        else:
            rank0_print("Hierarchical mode requested but only 1 node detected, falling back to flat exchange")
    
    if world_size > 1 and dist_is_enabled():
        # Build my_comm_map in expected format: {recv_rank: indices_to_send}
        # my_recv_indices tells us where to place received ghost data
        calc = DistributedCalculator(
            calc,
            cutoff,
            comm_map=my_send_map,
            interior_mask=my_interior_mask,
            enable_overlap=cfg.dist.overlap,
            recv_ghost_indices=my_recv_indices,
            hierarchical_params=hierarchical_params,
            per_node_params=per_node_params
        )
        # Replace exchanger with RingExchanger if ring_exchange is enabled
        if cfg.dist.ring_exchange:
            calc.exchanger = RingExchanger(
                calc.rank, calc.world_size, my_send_map, device,
                recv_ghost_indices=my_recv_indices
            )
            rank0_print("Ring exchange ENABLED (isend/irecv with rank n-1, n+1 only)")
    else:
        # Single rank: still wrap with DistributedCalculator for consistent profiling path
        calc = DistributedCalculator(
            calc, 
            cutoff, 
            comm_map={},  # No comm map for single rank
            interior_mask=None, 
            enable_overlap=False,
            recv_ghost_indices={}
        )
    
    # Differentiable feature exchange: backward auto-sends ghost grads
    calc.differentiable_exchange = cfg.dist.differentiable_exchange
    if cfg.dist.differentiable_exchange and rank == 0:
        print("Differentiable feature exchange ENABLED (adds L backward all_to_all for ghost gradients)")

    # Optimization #2: Skip energy all-reduce on non-logging steps.
    # MD integration only needs forces; energy is for monitoring.
    calc._energy_allreduce_interval = cfg.md.log_interval

    local_atoms.calc = calc

    # === NVSHMEM INITIALIZATION ===
    # Must happen after DistributedCalculator is created but before first calculate()
    # The model's set_comm_maps (called during first calculate) will create the
    # NVSHMEMNodeFeatureExchanger. We pre-initialize NVSHMEM here so it's ready.
    if cfg.dist.use_nvshmem and cfg.dist.per_node_partition and world_size > 1:
        import baseline.nvshmem_ext as nvshmem_ext
        # JIT compile on rank 0 only, then barrier, then all ranks load
        if rank == 0:
            rank0_print("Compiling NVSHMEM extension...")
            nvshmem_ext.compile()
            rank0_print("NVSHMEM extension compiled.")
        dist.barrier()
        # UniqueID bootstrap: rank 0 generates ID, broadcast to all, then init
        # NCCL backend only supports CUDA tensors, so move to GPU for broadcast
        if rank == 0:
            uid = nvshmem_ext.get_unique_id().cuda()
        else:
            uid = torch.zeros(128, dtype=torch.uint8, device='cuda')
        dist.broadcast(uid, src=0)
        nvshmem_ext.init(uid.cpu(), rank, world_size)
        rank0_print("NVSHMEM initialized for inter-node feature exchange")

    # === PROFILER SETUP ===
    step_profiler = StepProfiler(enabled=cfg.profiler.enabled, rank=rank)
    if isinstance(calc, DistributedCalculator):
        calc.set_profiler(step_profiler)
        calc.measure_bw = cfg.profiler.measure_bw
    else:
        # Single-rank case: set profiler directly on FAIRChemCalculator
        calc._profiler = step_profiler
        # Also wire profiler to model backbone for detailed timing
        model = calc.predictor.model
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "backbone") and hasattr(model.backbone, "set_profiler"):
            model.backbone.set_profiler(step_profiler)
    
    # 5. Run MD
    # We need to run MD for total_steps
    # But we need to rebalance every rebalance_interval
    
    total_done = 0
    
    # Print initial energy before MD starts
    init_energy = local_atoms.get_potential_energy()
    rank0_print(f"Step 0 (initial): Total Energy = {init_energy:.4f} eV")

    # Save initial forces (before any MD steps) for force comparison
    if cfg.md.traj_interval > 0:
        init_forces = local_atoms.get_forces()
        init_positions = local_atoms.get_positions()
        init_local_energy = calc.results.get('local_energy', init_energy) if hasattr(calc, 'results') else init_energy
        init_save_dict = {
            'step': -1,
            'positions': init_positions,
            'energy': init_local_energy,
            'forces': init_forces,
            'atomic_numbers': local_atoms.get_atomic_numbers(),
        }
        if local_atoms.has("global_id"):
            init_save_dict['global_id'] = local_atoms.get_array("global_id")
        init_path = os.path.join(output_dir, f"traj_step_init_rank{rank}.npz")
        np.savez(init_path, **init_save_dict)

    # Reset calc step counter so MD step N maps to _calc_step N.
    # The init call above incremented _calc_step to 1; resetting to 0
    # ensures log_interval-aligned steps (100, 200, ...) hit the
    # energy allreduce condition (_calc_step % interval == 0).
    calc._calc_step = 0
    

    while total_done < total_steps:
        start_time = time.time()
        steps_to_run = min(rebalance_interval, total_steps - total_done)
        # Rebalance / Partitioning
        if world_size > 1 and (total_done % cfg.dist.rebalance_interval == 0):
            # ============================================
            # DISTRIBUTED REBALANCE using unified function
            # ============================================
            
            # Get local atom data
            local_pos = torch.tensor(local_atoms.get_positions(), dtype=torch.float32, device=device)
            local_vel = torch.tensor(local_atoms.get_velocities(), dtype=torch.float32, device=device)
            local_Z = torch.tensor(local_atoms.get_atomic_numbers(), dtype=torch.int32, device=device)
            has_gid = "global_id" in local_atoms.arrays
            local_gid = torch.tensor(local_atoms.get_array("global_id"), dtype=torch.int64, device=device) if has_gid else None
            
            # Call unified partition and redistribute function
            if cfg.dist.per_node_partition:
                (new_pos, new_vel, new_Z, new_gid, 
                 my_send_map, my_recv_indices, my_interior_mask, per_node_info) = do_per_node_partition_and_redistribute(
                    local_pos=local_pos,
                    local_vel=local_vel,
                    local_Z=local_Z,
                    local_gid=local_gid,
                    cell=cell,
                    pbc=pbc,
                    cutoff=cfg.dist.cutoff,
                    ghost_cutoff_buffer=cfg.dist.ghost_cutoff_buffer,
                    partition_algorithm=cfg.dist.partition_algorithm,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    profiler=step_profiler,
                    edge_weighted=cfg.dist.edge_weighted_partition
                )
                # Update per_node_params with new info (intra_group stays same)
                per_node_params.update(per_node_info)
            else:
                (new_pos, new_vel, new_Z, new_gid,
                 my_send_map, my_recv_indices, my_interior_mask) = do_partition_and_redistribute(
                    local_pos=local_pos,
                    local_vel=local_vel,
                    local_Z=local_Z,
                    local_gid=local_gid,
                    cell=cell,
                    pbc=pbc,
                    cutoff=cfg.dist.cutoff,
                    ghost_cutoff_buffer=cfg.dist.ghost_cutoff_buffer,
                    partition_algorithm=cfg.dist.partition_algorithm,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    profiler=step_profiler,
                    edge_weighted=cfg.dist.edge_weighted_partition,
                    ring_exchange=cfg.dist.ring_exchange,
                )

            # Update calculator with new maps
            if isinstance(calc, DistributedCalculator):
                ExchangerClass = RingExchanger if cfg.dist.ring_exchange else HaloExchanger
                calc.exchanger = ExchangerClass(
                    rank, world_size, my_send_map, device,
                    recv_ghost_indices=my_recv_indices
                )
                calc.exchanger._measure_bw = cfg.profiler.measure_bw
                calc.interior_mask = my_interior_mask.to(device) if my_interior_mask is not None else None
                # CRITICAL: Invalidate caches after repartition
                calc._cached_local_Z_tensor = None
                calc._cached_combined_atoms = None
                calc._cached_num_local = None
                calc._cached_num_ghosts = None
                calc._comm_map_version += 1  # Invalidate comm map cache
                
                # Update per-node params if applicable
                if cfg.dist.per_node_partition:
                    calc.per_node_params = per_node_params
                    # Must trigger set_comm_maps update which happens in _prepare_combined_atoms
                    # The cache invalidation above is sufficient for maps, but we also need to update
                    # atom counts in the exchanger. This happens automatically via set_comm_maps
                    # because we update calc.per_node_params which is passed to set_comm_maps.
            
            # Update local atoms
            local_atoms = ase.Atoms(
                numbers=new_Z.cpu().numpy(),
                positions=new_pos.cpu().numpy(),
                velocities=new_vel.cpu().numpy()
            )
            local_atoms.set_cell(cell_np)
            local_atoms.set_pbc(list(pbc_tuple))
            
            if has_gid and new_gid is not None:
                local_atoms.set_array("global_id", new_gid.cpu().numpy())
            
            # Initialize charge and spin for OMOL task
            local_atoms.info["charge"] = 0
            local_atoms.info["spin"] = 1
            
            # Re-attach calculator
            local_atoms.calc = calc
            

        # Simulation step - create dyn each episode (MDLogger holds atoms reference)
        # MDLogger holds atoms reference at creation time. Moving dyn outside loop causes slowdown. Overhead of recreating dyn is negligible.
        dyn = ase.md.langevin.Langevin(local_atoms, timestep, temperature_K=temperature_K, friction=friction)
        
        def output_cb():
            local_atoms.info["Step"] = total_done
            local_atoms.info["Time"] = total_done * (timestep / units.fs)
            local_atoms.info["T_K"] = local_atoms.get_temperature()
            local_atoms.write(output_file_path, append=True)

        dyn.attach(output_cb, interval=traj_interval)
        dyn.attach(ase.md.MDLogger(dyn, local_atoms, md_log_path), interval=log_interval)
        
        # Start profiling for this episode
        step_profiler.start_episode()
        
        # PyTorch Profiler for deep model analysis (first episode only)
        use_torch_profiler = (cfg.profiler.torch_profiler and total_done == 0 and rank == 0)
        if use_torch_profiler:
            from torch.profiler import profile, ProfilerActivity, schedule
            
            profiler_dir = os.path.join(output_dir, "torch_profiler")
            os.makedirs(profiler_dir, exist_ok=True)
            
            rank0_print(f"PyTorch Profiler enabled. Output: {profiler_dir}")
            
            # Profile with warmup
            prof_schedule = schedule(
                wait=0,  # No wait
                warmup=2,  # First 2 steps are warmup
                active=cfg.profiler.torch_profiler_steps,  # Active profiling
                repeat=1
            )
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=lambda p: p.export_chrome_trace(
                    os.path.join(profiler_dir, f"trace_step_{p.step_num}.json")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                for step_i in range(min(cfg.md.steps_per_episode, 2 + cfg.profiler.torch_profiler_steps)):
                    dyn.run(steps=1)
                    step_profiler.step_done()
                    prof.step()
                
                # Print aggregated stats
                rank0_print("\n===== PyTorch Profiler: Top 20 CUDA Operations =====")
                rank0_print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                
                # Save table to file
                with open(os.path.join(profiler_dir, "profiler_summary.txt"), "w") as f:
                    f.write("===== Top 20 CUDA Operations =====\n")
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
                    f.write("\n\n===== Top 20 CPU Operations =====\n")
                    f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            
            # Run remaining steps without profiler
            remaining_steps = cfg.md.steps_per_episode - (2 + cfg.profiler.torch_profiler_steps)
            if remaining_steps > 0:
                for step_i in range(remaining_steps):
                    dyn.run(steps=1)
                    step_profiler.step_done()
        else:
            # Normal execution without PyTorch profiler
            # Run step-by-step to properly record profiler timing for each step
            if step_profiler.enabled:
                for step_i in range(cfg.md.steps_per_episode):
                    dyn.run(steps=1)
                    step_profiler.step_done()  # Accumulate per-step times
            else:
                dyn.run(steps=cfg.md.steps_per_episode)
        
        if dist_is_enabled():

            dist.barrier()
        run_time = time.time() - start_time
        
        # Report profiling for this episode (only rank 0 prints to stdout)
        if rank == 0:
            step_profiler.report()

        # Bandwidth report (all ranks print since BW may differ)
        if cfg.profiler.measure_bw and isinstance(calc, DistributedCalculator):
            calc.exchanger.bw_report()
            model = calc.calculator.predictor.model
            if hasattr(model, 'module'):
                model = model.module
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'exchanger'):
                model.backbone.exchanger.bw_report()
        
        # Save profiling results to CSV for ALL ranks
        current_episode = total_done // cfg.md.steps_per_episode
        step_profiler.save_to_csv(output_dir, episode=current_episode)
        
        # Get energy and forces for this step
        step_energy = local_atoms.get_potential_energy()  # This is global energy
        step_forces = local_atoms.get_forces()
        step_positions = local_atoms.get_positions()
        # Get per-atom energies from Student (must be done right after forces, before any other calculation)
        # For StudentModelCalculator, energies are stored directly in calc.results
        # For DistributedCalculator wrapping StudentModelCalculator, need to access inner calculator
        inner_calc = local_atoms.calc
        if hasattr(inner_calc, 'calculator'):  # DistributedCalculator wraps inner calc
            inner_calc = inner_calc.calculator
        step_energies = inner_calc.results.get('energies', None) if hasattr(inner_calc, 'results') else None
        
        # Get local energy for this rank (for trajectory NPZ)
        calc = local_atoms.calc
        local_energy = calc.results.get('local_energy', step_energy) if hasattr(calc, 'results') else step_energy
        
        # Save trajectory data to NPZ file
        traj_data_path = os.path.join(output_dir, f"traj_step_{total_done}_rank{rank}.npz")
        
        global_id = local_atoms.get_array("global_id") if local_atoms.has("global_id") else None
        
        save_dict = {
            'step': total_done,
            'positions': step_positions, 
            'energy': local_energy,
            'forces': step_forces,
            'atomic_numbers': local_atoms.get_atomic_numbers()
        }
        if global_id is not None:
            save_dict['global_id'] = global_id
        np.savez(traj_data_path, **save_dict)
        total_done += cfg.md.steps_per_episode
        rank0_print(f"Step {total_done}: Total Energy = {step_energy:.4f} eV")
        rank0_print(f"Ran {cfg.md.steps_per_episode} steps in {run_time:.2f}s. (perf: {cfg.md.steps_per_episode/run_time:.2f} steps/s, {total_done}/{total_steps})")
        
    
    if dist_is_enabled():

        dist.barrier()
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def merge_mole_with_global_data(predictor, global_atoms, charge=0, spin=1, dataset="omol"):
    """
    Merge MOLE layers using global atomic data.
    
    In distributed mode, each rank only sees local atoms, but MOLE merge needs
    data from ALL atoms to compute consistent expert mixing coefficients.
    This function gathers global atomic data and performs the merge.
    
    Args:
        predictor: The MLIPPredictUnit containing the model
        global_atoms: ASE Atoms object containing ALL atoms in the system
        charge: System charge (default 0)
        spin: System spin (default 1)
        dataset: Dataset name (default "omol")
    """
    from baseline.local_fairchem.units.mlip_unit.predict import merge_uma_model
    
    # Build minimal data dict for MOLE merge
    # MOLE only needs: atomic_numbers, batch, charge, spin, dataset
    atomic_numbers = torch.tensor(global_atoms.get_atomic_numbers(), dtype=torch.long)
    natoms = len(global_atoms)
    
    # Create a simple dict-like object (merge_MOLE_model just needs dict access)
    class SimpleData(dict):
        """Simple dict-like object that also supports attribute access."""
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)
    
    data = SimpleData({
        "atomic_numbers": atomic_numbers,
        "natoms": torch.tensor([natoms], dtype=torch.long),
        "batch": torch.zeros(natoms, dtype=torch.long),  # Single system
        "charge": torch.tensor([int(charge)], dtype=torch.long),  # Must be int for embedding lookup
        "spin": torch.tensor([int(spin)], dtype=torch.long),  # Must be int for embedding lookup
        "dataset": [dataset],  # Must be list for iteration in DatasetEmbedding.forward
    })
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Rank {rank}: Merging MOLE with global data (natoms={natoms}, charge={charge}, spin={spin})", flush=True)
    
    # Call merge with global data
    model = predictor.model.module if hasattr(predictor.model, 'module') else predictor.model
    merge_uma_model(model, data)
    
    # IMPORTANT: Do NOT set lazy_model_intialized = True here!
    # We need predict() to call move_to_device() on first predict call.
    # But we MUST ensure merge_mole = False in inference_settings to prevent re-merge.
    # The merge already happened above with global data.
    predictor.inference_settings.merge_mole = False
    
    print(f"Rank {rank}: MOLE merge complete!", flush=True)

def setup_calculator(cfg, rank, world_size, device="cuda", use_barrier=True):
    # print(f"[DEBUG] Rank {rank}: setup_calculator START, file={__file__}, line={sys._getframe().f_lineno}", flush=True)
    # print(f"[DEBUG] Rank {rank}: backend={cfg.backend}, device={device}", flush=True)
    
    if cfg.backend == "mace":
        # print(f"[DEBUG] Rank {rank}: Loading MACE model...", flush=True)
        from mace.calculators import mace_mp
        calc = mace_mp(model=cfg.mace.model, default_dtype=cfg.mace.default_dtype, device=device)
    else:
        # Check if using Student model (file path ending with .pt)
        if str(cfg.uma.model).endswith(".pt"):
            with Timer(f"Setting up Student model ({cfg.uma.model})"):
                model_path = cfg.uma.model
                if not os.path.isabs(model_path):
                    model_path = os.path.join(os.path.dirname(__file__), "..", model_path)
                
                rank0_print(f"Loading Student model from: {model_path}")
                student = StudentModel.load(model_path, device=device)
                
                # Create StudentModelCalculator (defined below)
                calc = StudentModelCalculator(student, cutoff=student.cutoff)
                rank0_print(f"Student model loaded: {student.config.name}")
                rank0_print(f"  Params: {student.num_params:,}")
                rank0_print(f"  Cutoff: {student.cutoff}")
        else:
            # Standard UMA model loading
            # print(f"[DEBUG] Rank {rank}: Building inference_settings_turbo...", flush=True)
            settings = inference_settings_turbo()
            settings.activation_checkpointing = cfg.uma.activation_checkpointing
            settings.merge_mole = bool(cfg.uma.merge_mole)
            settings.wigner_cuda = cfg.uma.wigner_cuda
            settings.compile = bool(cfg.uma.compile)
            settings.compile_cache_path = cfg.uma.cache_path  # Distributed compile cache sharing
            
            rank0_print(f"Merge Mole: {settings.merge_mole}")
            rank0_print(f"Wigner CUDA: {settings.wigner_cuda}")
            rank0_print(f"Torch Compile: {settings.compile}")
            rank0_print(f"Compile Cache Path: {settings.compile_cache_path}")
            
            if settings.compile:
                torch._dynamo.config.cache_size_limit = cfg.uma.dynamo_cache_size
                # Suppress dynamo errors so it falls back to eager instead of crashing
                # (distributed forward has NCCL ops and dynamic control flow)
                torch._dynamo.config.suppress_errors = True

            # print(f"[DEBUG] Rank {rank}: Preparing overrides...", flush=True)
            with Timer(f"Setting up calculator ({cfg.uma.model})"):
                
                overrides = {}
                if cfg.dist.use_distributed_model:
                    # Override backbone to use distributed version
                    overrides = {"backbone": {"model": "distributed_escnmd_backbone"}}
                    rank0_print(f"Overriding model backbone to distributed_escnmd_backbone")

                # Inject Triton Wigner config into backbone
                if "backbone" not in overrides:
                    overrides["backbone"] = {}
                overrides["backbone"]["use_triton_wigner"] = cfg.uma.use_triton_wigner
                if cfg.uma.use_triton_wigner:
                     rank0_print(f"Enabled Triton Wigner Kernel Optimization")

                # print(f"[DEBUG] Rank {rank}: Calling pretrained_mlip.get_predict_unit...", flush=True)
                predictor = pretrained_mlip.get_predict_unit(cfg.uma.model, device=device, inference_settings=settings, overrides=overrides)
                # print(f"[DEBUG] Rank {rank}: get_predict_unit returned successfully", flush=True)
                
                # print(f"[DEBUG] Rank {rank}: Creating FAIRChemCalculator...", flush=True)
                calc = FAIRChemCalculator(predictor, task_name="omol")
                # print(f"[DEBUG] Rank {rank}: FAIRChemCalculator created successfully", flush=True)
                
    if dist_is_enabled() and use_barrier:
        # print(f"[DEBUG] Rank {rank}: Entering dist.barrier() after calculator setup...", flush=True)

        dist.barrier()
        # print(f"[DEBUG] Rank {rank}: dist.barrier() completed", flush=True)
    
    # print(f"[DEBUG] Rank {rank}: setup_calculator END", flush=True)
    return calc

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: AppConfig) -> None:
    if cfg.dist.starter == "slurm":
        rank, world_size, local_rank = dist_init_from_slurm(backend=cfg.dist.backend)
    elif cfg.dist.starter == "torchrun":
        rank, world_size, local_rank = dist_init_from_torchrun(backend=cfg.dist.backend)
    elif cfg.dist.starter == "python":
        rank, world_size, local_rank = 0, 1, 0
    else:
        raise ValueError(f"Invalid starter: {cfg.dist.starter}")

    if not cfg.input_file:
        raise ValueError("input_file must be provided")
    if not os.path.exists(cfg.input_file):
        raise ValueError(f"input_file {cfg.input_file} does not exist")

    if rank == 0:
        print(f"Config: {cfg}")
        output_dir = to_absolute_path(os.path.join(cfg.output_dir, cfg.prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if cfg.save_config:
            with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
    
    print(f"Rank {rank}: Config processed. Output dir: {cfg.output_dir}")
    if rank == 0:
        pass  # per_node_partition config validated

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        # For single-GPU debugging with multiple ranks, map all ranks to device 0
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = "cuda"
    print(f"Rank {rank} Using device: {device}")

    # Build calculator
    calc = setup_calculator(cfg, rank, world_size, device=device)
    
    print(f"Rank {rank}: Calculator built. Starting simulation...")
    run_md_simulation_distributed(calc, cfg, rank, world_size, local_rank)

if __name__ == "__main__":
    main()