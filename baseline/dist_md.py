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
from baseline.sfc import sfc_partition
from ase.calculators.calculator import Calculator, all_changes


# Configure torch inductor
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

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
    model: str = "uma-s-1p1" # ["uma-m-1p1", "uma-s-1p1"]
    compile: bool = False
    wigner_cuda: bool = True
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

@dataclass
class DistConfig:
    starter: str = "slurm" # ["slurm", "torchrun", "python"]
    backend: str = "nccl"
    cutoff: float = 6.0
    ghost_cutoff_buffer: float = 0.5  # Extra distance for ghost node selection to avoid FP edge cases
    partition_algorithm: str = "sfc_morton" # ["sfc_morton", "sfc_hilbert", "simple"]
    k_partition: Optional[int] = None
    rebalance_interval: int = 100
    rebalance_threshold: float = 0.2
    use_gid: bool = True
    save_partition: bool = True
    save_timeseries: bool = True
    merge_results: bool = True
    gnn_layers: int = 4  # Number of GNN layers for halo depth
    use_distributed_model: bool = True # Use distributed model with direct feature fetching
    overlap: bool = False  # Enable computation-communication overlap (experimental)

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
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)

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



class HaloExchanger:
    def __init__(self, rank: int, world_size: int, comm_map: Dict[int, torch.Tensor], device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.comm_map = {r: indices.to(device) for r, indices in comm_map.items()} # Dict[target_rank, local_indices_to_send]
        self.device = device
        
        # Pre-compute receive counts?
        # We can exchange counts at the beginning of exchange, or pre-compute if static.
        # Since comm_map is static between rebalances, we can pre-compute recv_counts.
        # But here we only have send_map.
        # We need to exchange counts once during init (or update_topology).
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
        
        # 1. Prepare Input Tensor (Packed)
        # We need to order data by rank 0..world_size-1
        input_chunks = []
        input_split_sizes = []
        
        for r in range(self.world_size):
            if r in self.comm_map:
                indices = self.comm_map[r]
                s_pos = pos[indices]
                s_vel = vel[indices]
                s_Z = Z[indices].float().unsqueeze(1)
                packed = torch.cat([s_pos, s_vel, s_Z], dim=1)
                input_chunks.append(packed)
                input_split_sizes.append(packed.size(0))
            else:
                # Empty chunk with correct feature dim (7)
                input_chunks.append(torch.empty((0, 7), dtype=torch.float32, device=self.device))
                input_split_sizes.append(0)
                
        input_tensor = torch.cat(input_chunks, dim=0)
        
        # 2. Prepare Output Tensor
        # We need output split sizes (recv counts)
        output_split_sizes = []
        for r in range(self.world_size):
            output_split_sizes.append(self.recv_counts.get(r, 0))
            
        total_recv = sum(output_split_sizes)
        output_tensor = torch.zeros((total_recv, 7), dtype=torch.float32, device=self.device)
        
        # 3. All-to-All
        dist.all_to_all_single(output_tensor, input_tensor, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
        
        # 4. Unpack
        # Split output_tensor back into chunks
        # torch.split returns a tuple of views
        output_chunks = torch.split(output_tensor, output_split_sizes, dim=0)
        
        ghost_pos_list = []
        ghost_vel_list = []
        ghost_Z_list = []
        
        # We only care about chunks where we expected data (recv_counts > 0)
        # But we need to iterate all chunks to match rank index
        for r, chunk in enumerate(output_chunks):
            if output_split_sizes[r] > 0:
                ghost_pos_list.append(chunk[:, 0:3])
                ghost_vel_list.append(chunk[:, 3:6])
                ghost_Z_list.append(chunk[:, 6].long())
        
        if not ghost_pos_list:
             return {}
             
        ghost_pos = torch.cat(ghost_pos_list, dim=0)
        ghost_vel = torch.cat(ghost_vel_list, dim=0)
        ghost_Z = torch.cat(ghost_Z_list, dim=0)
        
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
            
        if ghost_forces is None:
            return local_forces

        # 1. Prepare Send Buffer (send ghost_forces back to owners)
        # ghost_forces are already ordered by rank (same as recv order in exchange)
        # Input splits (what we have) -> recv_counts
        input_split_sizes = []
        for r in range(self.world_size):
            input_split_sizes.append(self.recv_counts.get(r, 0))
            
        send_tensor = ghost_forces
        if not send_tensor.is_contiguous():
            send_tensor = send_tensor.contiguous()
            
        # 2. Prepare Recv Buffer (receive forces for atoms we sent)
        # Output splits (what we expect) -> send_counts (len(comm_map[r]))
        output_split_sizes = []
        send_counts = {}
        total_recv = 0
        for r in range(self.world_size):
            count = len(self.comm_map.get(r, [])) if r in self.comm_map else 0
            output_split_sizes.append(count)
            send_counts[r] = count
            total_recv += count
            
        recv_tensor = torch.zeros((total_recv, 3), dtype=local_forces.dtype, device=self.device)
        
        # 3. Exchange
        dist.all_to_all_single(recv_tensor, send_tensor, 
                               output_split_sizes=output_split_sizes, 
                               input_split_sizes=input_split_sizes)
        
        # 4. Scatter to local forces
        # OPTIMIZATION: Batched index_add_
        if self._reduction_indices.numel() > 0:
             local_forces.index_add_(0, self._reduction_indices, recv_tensor)
                
        return local_forces



class DistributedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, calculator, cutoff: float, comm_map: Dict[int, torch.Tensor] = None, interior_mask: torch.Tensor = None, enable_overlap: bool = False):
        super().__init__()
        self.calculator = calculator
        self.cutoff = cutoff
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map if comm_map else {}, device)
        self.num_local = 0
        
        # Store interior mask and overlap flag for optimization
        self.interior_mask = interior_mask.to(device) if interior_mask is not None else None
        self.enable_overlap = enable_overlap
        
        # OPTIMIZATION: Cache comm maps to avoid rebuilding every step
        self._cached_recv_ghost_indices = None
        self._cached_device_send_map = None
        self._cached_comm_map_version = -1  # Version counter to invalidate cache on update
        self._comm_map_version = 0
        
        # OPTIMIZATION: Cache atomic numbers (they don't change during MD)
        self._cached_local_Z_tensor = None
        
    def update_comm_map(self, comm_map: Dict[int, torch.Tensor], interior_mask: torch.Tensor = None):
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map, self.exchanger.device)
        # Update interior mask
        if interior_mask is not None:
            self.interior_mask = interior_mask.to(self.exchanger.device)
        # OPTIMIZATION: Invalidate cached comm maps
        self._comm_map_version += 1

    def _prepare_combined_atoms(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, Dict[str, torch.Tensor]]:
        # 1. Exchange ghosts
        ghosts = self.exchanger.exchange(atoms)
        
        self.num_local = len(atoms)
        num_local = self.num_local
        
        # Update model comm maps (only rebuild if version changed)
        model = self.calculator.predictor.model
        if hasattr(model, "module"):
            model = model.module
        
        if hasattr(model, "backbone") and hasattr(model.backbone, "set_comm_maps"):
            if self._cached_comm_map_version != self._comm_map_version or self._cached_recv_ghost_indices is None:
                recv_ghost_indices = {}
                current_idx = num_local
                
                for r in range(self.world_size):
                    count = self.exchanger.recv_counts.get(r, 0)
                    if count > 0:
                        indices = torch.arange(current_idx, current_idx + count, device=self.exchanger.device)
                        recv_ghost_indices[r] = indices
                        current_idx += count
                
                device_send_map = {r: idx.to(self.exchanger.device) for r, idx in self.exchanger.comm_map.items()}
                
                self._cached_recv_ghost_indices = recv_ghost_indices
                self._cached_device_send_map = device_send_map
                self._cached_comm_map_version = self._comm_map_version
            
            model.backbone.set_comm_maps(self._cached_device_send_map, self._cached_recv_ghost_indices, self.rank, self.world_size, self.interior_mask, self.enable_overlap)
            model.backbone.num_local = num_local
        
        # 2. Build combined atoms
        # OPTIMIZATION: Cache combined_atoms to avoid expensive object creation
        if not hasattr(self, "_cached_combined_atoms"):
             self._cached_combined_atoms = None
             self._cached_num_ghosts = -1
             self._cached_num_local = -1

        num_ghosts = 0
        if ghosts:
            ghost_pos = ghosts["pos"].cpu().numpy()
            ghost_Z = ghosts["Z"].cpu().numpy()
            num_ghosts = len(ghost_pos)

        # Check if cache is valid
        # We also need to check if atoms length changed (rebalance or just different atoms object)
        cache_valid = False
        if (self._cached_combined_atoms is not None and 
            self._cached_num_local == len(atoms) and 
            self._cached_num_ghosts == num_ghosts):
            cache_valid = True
        
        if cache_valid:
            combined_atoms = self._cached_combined_atoms
            # Update local part (in-place)
            combined_atoms.positions[:len(atoms)] = atoms.get_positions()
            # Cell/PBC usually constant in NVE/NVT but might change in NPT.
            # Safe to update them (cheap).
            combined_atoms.cell[:] = atoms.get_cell()
            
            # Update ghost part
            if num_ghosts > 0:
                combined_atoms.positions[len(atoms):] = ghost_pos
                # Z updates if needed (usually static, but safe to update)
                combined_atoms.numbers[len(atoms):] = ghost_Z
                
        else:
            # Rebuild cache
            if ghosts:
                combined_atoms = atoms.copy()
                combined_atoms.extend(ase.Atoms(numbers=ghost_Z, positions=ghost_pos))
                combined_atoms.set_cell(atoms.get_cell())
                combined_atoms.set_pbc(atoms.get_pbc())
            else:
                combined_atoms = atoms.copy()

            self._cached_combined_atoms = combined_atoms
            self._cached_num_local = len(atoms)
            self._cached_num_ghosts = num_ghosts
        
        return combined_atoms, ghosts
        

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
            
        combined_atoms, ghosts = self._prepare_combined_atoms(atoms)
        
        # Run calculation on combined atoms
        combined_atoms.calc = self.calculator
        self.calculator.num_local = self.num_local
        
        # Calculate forces
        all_forces_tensor = None
        if hasattr(self.calculator, "results_raw") and "forces" in self.calculator.results_raw:
            if self.calculator.calculation_required(combined_atoms, ["forces"]):
                self.calculator.calculate(combined_atoms, ["forces"], [])
            all_forces_tensor = self.calculator.results_raw.get("forces")

        if all_forces_tensor is None:
            all_forces = self.calculator.get_forces(combined_atoms)
            if isinstance(all_forces, np.ndarray):
                all_forces_tensor = torch.from_numpy(all_forces).to(self.exchanger.device)
            else:
                all_forces_tensor = all_forces
            
        local_forces = all_forces_tensor[:self.num_local]
        
        if ghosts:
            ghost_forces = all_forces_tensor[self.num_local:]
            local_forces = self.exchanger.reduce_forces(local_forces, ghost_forces)
            
        self.results["forces"] = local_forces.cpu().numpy()
        
        # Calculate energy
        energies = self.calculator.get_potential_energies(combined_atoms)
        
        if isinstance(energies, torch.Tensor):
            local_energy_tensor = energies[:self.num_local].sum().to(dtype=torch.float64)
            global_energy_tensor = local_energy_tensor.clone()
            dist.all_reduce(global_energy_tensor, op=dist.ReduceOp.SUM)
            
            self.results['energy'] = global_energy_tensor.item()
            self.results['local_energy'] = local_energy_tensor.item()
        else:
            local_energy_sum = np.sum(energies[:self.num_local])
            self.results['local_energy'] = local_energy_sum
            energy_tensor = torch.tensor([local_energy_sum], dtype=torch.float64, device=self.exchanger.device)
            dist.all_reduce(energy_tensor, op=dist.ReduceOp.SUM)
            self.results['energy'] = energy_tensor.item()

    def get_potential_energies(self, atoms=None):
        # Delegate to the inner calculator
        # We assume atoms is the same as self.atoms (which is local_atoms)
        # The inner calculator has results for combined_atoms (local + ghosts)
        # We should return energies for all atoms in the inner calculator's results
        # The caller (verify_forces.py) will handle slicing/gathering.
        
        if self.calculation_required(atoms, ["energy"]):
             # We can't easily trigger a full distributed calc from here without atoms being passed correctly
             # But usually calculate() has already been called.
             pass
             
        # Check if inner calculator has energies
        if hasattr(self.calculator, "results") and "energies" in self.calculator.results:
            return self.calculator.results["energies"]
        else:
            raise PropertyNotImplementedError("energies property not implemented by the inner calculator")

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
    
    # Rank0 reads full system
    if rank == 0:
        full_atoms = ase.io.read(cfg.input_file)
        # Set cell if needed
        current_cell_lengths = full_atoms.get_cell().lengths()
        if not (current_cell_lengths > 0).all() and cell_size > 0:
            full_atoms.set_cell([cell_size] * 3)
        full_atoms.set_pbc([True] * 3)
        
        ase.md.velocitydistribution.MaxwellBoltzmannDistribution(full_atoms, temperature_K=temperature_K)
        
        positions = full_atoms.get_positions().astype(np.float32)
        velocities = full_atoms.get_velocities().astype(np.float32)
        numbers = full_atoms.get_atomic_numbers().astype(np.int32)
        
        rank0_print(f"Partitioning into {world_size} ranks using {cfg.dist.partition_algorithm}...")
        
        # Prepare data on device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_tensor = torch.from_numpy(positions).to(device)
        vel_tensor = torch.from_numpy(velocities).to(device)
        Z_tensor = torch.from_numpy(numbers).to(device)
        if cfg.dist.use_gid:
            global_ids = np.arange(len(numbers), dtype=np.int64)
            gid_tensor = torch.from_numpy(global_ids).to(device)
        
        if world_size > 1:
            if cfg.dist.partition_algorithm == "simple":
                # Simple partition: just split atoms sequentially, preserves original order
                # Useful for comparison testing with serial version
                n_atoms = len(numbers)
                atoms_per_rank = n_atoms // world_size
                partition_id = torch.zeros(n_atoms, dtype=torch.long, device=device)
                for r in range(world_size):
                    start_idx = r * atoms_per_rank
                    end_idx = (r + 1) * atoms_per_rank if r < world_size - 1 else n_atoms
                    partition_id[start_idx:end_idx] = r
            else:
                # Use SFC partition
                curve_type = "hilbert" if "hilbert" in cfg.dist.partition_algorithm else "morton"
                partition_id, _ = sfc_partition(pos_tensor, world_size, curve_type=curve_type, device=device)
        else:
            partition_id = torch.zeros(len(numbers), dtype=torch.long, device=device)
            
        # Build tensors per rank using GPU masking
        fields_per_rank: Dict[str, List[torch.Tensor]] = {
            "pos": [], "vel": [], "Z": []
        }
        
        for r in range(world_size):
            mask = (partition_id == r)
            fields_per_rank["pos"].append(pos_tensor[mask])
            fields_per_rank["vel"].append(vel_tensor[mask])
            fields_per_rank["Z"].append(Z_tensor[mask])
            if cfg.dist.use_gid:
                fields_per_rank.setdefault("gid", []).append(gid_tensor[mask])
        
        cell = torch.from_numpy(np.asarray(full_atoms.get_cell().array, dtype=np.float32)).to(device)
        pbc = torch.tensor([bool(x) for x in full_atoms.get_pbc()], dtype=torch.int64).to(device)
        
        # Build initial halo maps
        if world_size > 1:
            rank0_print(f"Building initial halo maps for {cfg.dist.gnn_layers} GNN layers...")
            # Rebuild halo maps
            # If using distributed model, we only need 1-hop halo for graph construction
            halo_gnn_layers = 1 if cfg.dist.use_distributed_model else cfg.dist.gnn_layers
            # Ghost cutoff includes buffer to avoid FP edge cases
            ghost_cutoff = cfg.dist.cutoff + cfg.dist.ghost_cutoff_buffer
            
            all_comm_maps, all_interior_masks = build_halo_maps(
                pos=pos_tensor,
                partition_id=partition_id,
                cutoff=ghost_cutoff,
                gnn_layers=halo_gnn_layers,
                cell=cell,
                pbc=pbc,
                world_size=world_size
            )
            # Log interior/boundary stats
            for r, mask in enumerate(all_interior_masks):
                if mask.numel() > 0:
                    interior_count = mask.sum().item()
                    total_count = mask.numel()
                    rank0_print(f"  Rank {r}: {interior_count}/{total_count} interior atoms ({100*interior_count/total_count:.1f}%)")
        else:
            all_comm_maps = None
            all_interior_masks = None
            
    else:
        fields_per_rank = {
            "pos": [torch.zeros((1, 3), dtype=torch.float32)] * world_size,
            "vel": [torch.zeros((1, 3), dtype=torch.float32)] * world_size,
            "Z": [torch.zeros((1,), dtype=torch.int32)] * world_size,
        }
        if cfg.dist.use_gid:
            fields_per_rank["gid"] = [torch.zeros((1,), dtype=torch.int64)] * world_size
        cell = torch.empty((3, 3), dtype=torch.float32)
        pbc = torch.empty((3,), dtype=torch.int64)

    # Broadcast cell and pbc
    if world_size > 1:
        cell = _broadcast_tensor(cell, src=0)
        pbc = _broadcast_tensor(pbc, src=0)
    
    cell_np = cell.detach().cpu().numpy()
    pbc_tuple = tuple(bool(int(x)) for x in pbc.detach().cpu().numpy().tolist())

    # Broadcast initial payload
    if world_size > 1 and dist_is_enabled():
        dtype_map = {"pos": torch.float32, "vel": torch.float32, "Z": torch.int32}
        if cfg.dist.use_gid:
            dtype_map["gid"] = torch.int64
        payload = _concat_per_rank_and_broadcast(fields_per_rank, dtype_map)
        local_atoms = _build_local_atoms_from_payload(payload, cell_np, pbc_tuple)
        # Initialize charge and spin for OMOL task
        local_atoms.info["charge"] = 0
        local_atoms.info["spin"] = 1
        
        # Distribute initial comm_maps and interior_masks
        my_comm_map_container = [None]
        dist.scatter_object_list(my_comm_map_container, all_comm_maps if rank == 0 else [], src=0)
        my_comm_map = my_comm_map_container[0]
        
        my_interior_mask_container = [None]
        dist.scatter_object_list(my_interior_mask_container, all_interior_masks if rank == 0 else [], src=0)
        my_interior_mask = my_interior_mask_container[0]
        
    else:
        if rank == 0:
            local_atoms = ase.Atoms(numbers=numbers.tolist(), positions=positions)
            local_atoms.set_cell(cell_np)
            local_atoms.set_pbc(list(pbc_tuple))
            local_atoms.set_velocities(velocities)
            if cfg.dist.use_gid:
                local_atoms.new_array("global_id", global_ids)
            
            # Initialize charge and spin for OMOL task
            local_atoms.info["charge"] = 0
            local_atoms.info["spin"] = 1
        else:
            local_atoms = ase.Atoms(cell=cell_np, pbc=list(pbc_tuple))
            # Initialize charge and spin for OMOL task
            local_atoms.info["charge"] = 0
            local_atoms.info["spin"] = 1
        
        my_comm_map = {}
        my_interior_mask = None

    # Wrap calculator with DistributedCalculator if distributed
    if world_size > 1 and dist_is_enabled():
        calc = DistributedCalculator(calc, cutoff, comm_map=my_comm_map, interior_mask=my_interior_mask, enable_overlap=cfg.dist.overlap)
    else:
        # For single rank, we might still want the wrapper for consistency or just use raw
        pass
    
    local_atoms.calc = calc
    
    # 5. Run MD
    # We need to run MD for total_steps
    # But we need to rebalance every rebalance_interval
    
    total_done = 0
    while total_done < total_steps:
        steps_to_run = min(rebalance_interval, total_steps - total_done)
        # Rebalance / Partitioning
        if world_size > 1 and (total_done % cfg.dist.rebalance_interval == 0):
            # 1. Gather all atoms to Rank 0
            # We need current positions/velocities/Z/gid
            # Note: local_atoms might have ghost atoms attached if we just stepped?
            # No, ASE MD step updates positions of atoms object. 
            # If we used DistributedCalculator, it filters results, so atoms object only has local atoms updated?
            # Wait, DistributedCalculator.get_forces returns forces for local atoms.
            # ASE updates local atoms.
            # So local_atoms contains only local atoms.
            
            # 1. Gather all atoms to Rank 0
            gathered = _gather_all_atoms_to_rank0(local_atoms)
            
            # 2. Partition on Rank 0
            partition_id = None
            all_comm_maps = None
            
            if rank == 0:
                # Reconstruct global state
                pos_all = torch.from_numpy(gathered["pos"]).to(device)
                vel_all = torch.from_numpy(gathered["vel"]).to(device)
                Z_all = torch.from_numpy(gathered["Z"]).to(device)
                gid_all = torch.from_numpy(gathered["gid"]).to(device) if gathered.get("gid") is not None else None
                
                # Partition
                if cfg.dist.partition_algorithm == "simple":
                    # Simple partition: just split atoms sequentially (skip SFC)
                    n_atoms = pos_all.shape[0]
                    atoms_per_rank = n_atoms // world_size
                    partition_id = torch.zeros(n_atoms, dtype=torch.long, device=device)
                    for r in range(world_size):
                        start_idx = r * atoms_per_rank
                        end_idx = (r + 1) * atoms_per_rank if r < world_size - 1 else n_atoms
                        partition_id[start_idx:end_idx] = r
                else:
                    curve_type = "hilbert" if "hilbert" in cfg.dist.partition_algorithm else "morton"
                    partition_id, _ = sfc_partition(
                        coords=pos_all,
                        num_partitions=world_size,
                        curve_type=curve_type,
                        device=device
                    )
                partition_id = partition_id.to(dtype=torch.int32)
                
                # Build Halo Maps (Graph BFS)
                # We need cell and pbc
                cell_tensor = torch.tensor(local_atoms.get_cell().array, dtype=torch.float32, device=device)
                pbc_tensor = torch.tensor(local_atoms.get_pbc(), dtype=torch.bool, device=device)
                
                rank0_print(f"Building halo maps for {cfg.dist.gnn_layers} GNN layers...")
                ghost_cutoff = cutoff + cfg.dist.ghost_cutoff_buffer
                all_comm_maps, all_interior_masks = build_halo_maps(
                    pos_all, 
                    partition_id, 
                    ghost_cutoff, 
                    cfg.dist.gnn_layers, 
                    cell_tensor, 
                    pbc_tensor, 
                    world_size
                )
                
            # 3. Broadcast Partitioning and Maps
            # Distribute comm_maps and interior_masks
            my_comm_map = [None]
            dist.scatter_object_list(my_comm_map, all_comm_maps if rank == 0 else [], src=0)
            my_comm_map = my_comm_map[0]
            
            my_interior_mask = [None]
            dist.scatter_object_list(my_interior_mask, all_interior_masks if rank == 0 else [], src=0)
            my_interior_mask = my_interior_mask[0]
            
            # Update calculator with new map and interior_mask
            if isinstance(calc, DistributedCalculator):
                calc.update_comm_map(my_comm_map, my_interior_mask)
            
            # Distribute atoms (pos, vel, Z, gid)
            
            if rank == 0:
                pos_global = torch.from_numpy(gathered["pos"]).to(device)
                vel_global = torch.from_numpy(gathered["vel"]).to(device)
                Z_global = torch.from_numpy(gathered["Z"]).to(device)
                gid_global = torch.from_numpy(gathered["gid"]).to(device) if gathered.get("gid") is not None else torch.empty(0, device=device)
            else:
                # We need to know N to allocate buffers
                # Broadcast N
                n_tensor = torch.tensor([0], dtype=torch.int64, device=device)
                _broadcast_tensor(n_tensor, src=0)
                N = n_tensor.item()
                
                pos_global = torch.zeros((N, 3), dtype=torch.float32, device=device)
                vel_global = torch.zeros((N, 3), dtype=torch.float32, device=device)
                Z_global = torch.zeros((N,), dtype=torch.int32, device=device)
                gid_global = torch.zeros((N,), dtype=torch.int64, device=device) # Or empty
                partition_id = torch.zeros((N,), dtype=torch.int32, device=device)

            # Broadcast data
            if rank == 0:
                n_tensor = torch.tensor([len(pos_global)], dtype=torch.int64, device=device)
                _broadcast_tensor(n_tensor, src=0)
                
            _broadcast_tensor(pos_global, src=0)
            _broadcast_tensor(vel_global, src=0)
            _broadcast_tensor(Z_global, src=0)
            _broadcast_tensor(partition_id, src=0)
            # Handle gid
            # Check if gid exists
            has_gid = torch.tensor([1 if gid_global.numel() > 0 else 0], device=device)
            _broadcast_tensor(has_gid, src=0)
            if has_gid.item():
                if rank != 0:
                    gid_global = torch.zeros((pos_global.size(0),), dtype=torch.int64, device=device)
                _broadcast_tensor(gid_global, src=0)
            
            
            # 4. Update Local Atoms
            # Slice using partition_id
            mask = (partition_id == rank)
            
            local_pos = pos_global[mask].cpu().numpy()
            local_vel = vel_global[mask].cpu().numpy()
            local_Z = Z_global[mask].cpu().numpy()
            
            local_atoms = ase.Atoms(numbers=local_Z, positions=local_pos, velocities=local_vel)
            local_atoms.set_cell(cell_np)
            local_atoms.set_pbc(list(pbc_tuple))
            
            if has_gid.item():
                local_gid = gid_global[mask].cpu().numpy()
                local_atoms.set_array("global_id", local_gid)
            
            # Initialize charge and spin for OMOL task
            local_atoms.info["charge"] = 0
            local_atoms.info["spin"] = 1

                
            # Re-attach calculator
            local_atoms.calc = calc
            
            # Update dyn atoms
            # The dyn object is created inside the loop, so it will pick up the new local_atoms
            # No explicit update needed here, just ensure the next dyn.run() uses the new local_atoms.
            # The loop structure needs to be adjusted to create dyn *after* repartitioning.
            # Let's move the dyn creation inside the loop.
            
        # Simulation step
        dyn = ase.md.langevin.Langevin(local_atoms, timestep, temperature_K=temperature_K, friction=friction)
        
        def output_cb():
            local_atoms.info["Step"] = total_done
            local_atoms.info["Time"] = total_done * (timestep / units.fs)
            local_atoms.info["T_K"] = local_atoms.get_temperature()
            local_atoms.write(output_file_path, append=True)

        dyn.attach(output_cb, interval=traj_interval)
        dyn.attach(ase.md.MDLogger(dyn, local_atoms, md_log_path), interval=log_interval)
        
        start_time = time.time()
        dyn.run(steps=cfg.md.steps_per_episode) # Run for one step
        if dist_is_enabled():
            dist.barrier()
        run_time = time.time() - start_time
        
        # Get energy and forces for this step
        step_energy = local_atoms.get_potential_energy()  # This is global energy
        step_forces = local_atoms.get_forces()
        step_positions = local_atoms.get_positions()
        
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
        
        rank0_print(f"Step {total_done}: Total Energy = {step_energy:.4f} eV")
        
        total_done += cfg.md.steps_per_episode
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
    print(f"Rank {rank}: Building calculator...")
    if cfg.backend == "mace":
        from mace.calculators import mace_mp
        calc = mace_mp(model=cfg.mace.model, default_dtype=cfg.mace.default_dtype, device=device)
    else:
        settings = inference_settings_turbo()
        settings.merge_mole = bool(cfg.uma.merge_mole)
        settings.wigner_cuda = cfg.uma.wigner_cuda
        settings.compile = bool(cfg.uma.compile)
        
        rank0_print(f"Merge Mole: {settings.merge_mole}")
        rank0_print(f"Wigner CUDA: {settings.wigner_cuda}")
        rank0_print(f"Torch Compile: {settings.compile}")
        
        if settings.compile:
            torch._dynamo.config.cache_size_limit = cfg.uma.dynamo_cache_size

        with Timer(f"Setting up calculator ({cfg.uma.model})"):
            
            overrides = {}
            if cfg.dist.use_distributed_model:
                # Override backbone to use distributed version
                overrides = {"backbone": {"model": "distributed_escnmd_backbone"}}
                rank0_print(f"Overriding model backbone to distributed_escnmd_backbone")

            predictor = pretrained_mlip.get_predict_unit(cfg.uma.model, device=device, inference_settings=settings, overrides=overrides)
            calc = FAIRChemCalculator(predictor, task_name="omol")
            if dist_is_enabled() and use_barrier:
                dist.barrier()
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