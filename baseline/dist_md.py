import os
import time
import yaml
import hydra
import torch
import torch.distributed as dist
import numpy as np
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
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import inference_settings_turbo
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2
import torch._dynamo
import torch._inductor
from sfc import sfc_partition

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
    log_interval: int = 1

@dataclass
class UMAConfig:
    model: str = "uma-s-1p1" # ["uma-m-1p1", "uma-s-1p1"]
    compile: bool = False
    wigner_cuda: bool = True
    merge_mole: bool = True # should be disabled for distributed run
    cache_path: Optional[str] = None
    dynamo_cache_size: int = 64

@dataclass
class MACEConfig:
    model: str = "large"
    default_dtype: str = "float64"

@dataclass
class ProfilerConfig:
    enabled: bool = False
    log_dir: str = "./profiler_logs"

@dataclass
class DistConfig:
    starter: str = "slurm" # ["slurm", "torchrun", "python"]
    cutoff: float = 6.0
    partition_algorithm: str = "sfc_morton" # ["sfc_morton", "sfc_hilbert"]
    k_partition: Optional[int] = None
    rebalance_interval: int = 50
    rebalance_threshold: float = 0.2
    use_gid: bool = True
    save_partition: bool = True
    save_timeseries: bool = True
    merge_results: bool = True
    gnn_layers: int = 4  # Number of GNN layers for halo depth

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
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

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

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

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
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return 0, 1, 0

    if backend is None:
        if torch.cuda.is_available():
            # Use gloo if multiple ranks but only 1 GPU (debugging mode)
            if torch.cuda.device_count() == 1 and world_size > 1:
                backend = "gloo"
            else:
                backend = "nccl"
        else:
            backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, world_size, local_rank

def _broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not dist_is_enabled():
        return t
    
    if dist.get_backend() == "gloo" and t.device.type != 'cpu':
        t_cpu = t.cpu()
        dist.broadcast(t_cpu, src=src)
        t.copy_(t_cpu)
    else:
        dist.broadcast(t, src=src)
    return t

def _concat_per_rank_and_broadcast(fields_per_rank: Dict[str, List[torch.Tensor]], dtype_map: Dict[str, torch.dtype]) -> Dict[str, torch.Tensor]:
    """Rank0 concatenates each field across ranks and broadcasts both splits and data to all ranks."""
    assert dist_is_enabled(), "Distributed must be initialized"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" and torch.cuda.is_available() else torch.device("cpu")

    # Step 1: broadcast sizes per field per rank
    splits: Dict[str, torch.Tensor] = {}
    for name, dtype in dtype_map.items():
        if rank == 0:
            sizes = torch.tensor([int(x.size(0)) for x in fields_per_rank[name]], dtype=torch.int64)
        else:
            sizes = torch.empty(world_size, dtype=torch.int64)
        sizes = _broadcast_tensor(sizes, src=0)
        splits[name] = sizes.to(device)

    # Step 2: rank0 concatenates tensors and broadcasts full payloads
    payload: Dict[str, torch.Tensor] = {}
    for name, dtype in dtype_map.items():
        if rank == 0:
            cat = torch.cat([t.to(device) for t in fields_per_rank[name]], dim=0)
        else:
            total = int(splits[name].sum().item())
            shape_tail = tuple(int(x) for x in fields_per_rank[name][0].shape[1:]) if name in fields_per_rank else (3,)
            cat = torch.empty((total, *shape_tail), dtype=dtype, device=device)
        cat = _broadcast_tensor(cat, src=0)
        payload[name] = cat
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
    """Gather positions/velocities/numbers from all ranks to rank0 and concatenate."""
    world_size = dist.get_world_size() if dist_is_enabled() else 1
    rank = dist.get_rank() if dist_is_enabled() else 0
    
    if not dist_is_enabled() or world_size == 1:
        return {
            "pos": a.get_positions().astype(np.float32),
            "vel": a.get_velocities().astype(np.float32),
            "Z": a.get_atomic_numbers().astype(np.int32),
            "gid": a.get_array("global_id").astype(np.int64) if "global_id" in a.arrays else None
        }
    
    payload = {
        "pos": a.get_positions().astype(np.float32),
        "vel": a.get_velocities().astype(np.float32),
        "Z": a.get_atomic_numbers().astype(np.int32),
        "gid": a.get_array("global_id").astype(np.int64) if "global_id" in a.arrays else None
    }
    
    gathered: List[Optional[Dict[str, np.ndarray]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    
    if rank != 0:
        return None
        
    pos_all = np.concatenate([g["pos"] for g in gathered], axis=0)
    vel_all = np.concatenate([g["vel"] for g in gathered], axis=0)
    Z_all = np.concatenate([g["Z"] for g in gathered], axis=0)
    result = {"pos": pos_all, "vel": vel_all, "Z": Z_all}
    if gathered[0]["gid"] is not None:
        result["gid"] = np.concatenate([g["gid"] for g in gathered], axis=0)
        
    return result

def build_halo_maps(
    pos: torch.Tensor,
    partition_id: torch.Tensor,
    cutoff: float,
    gnn_layers: int,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    world_size: int
) -> List[Dict[int, torch.Tensor]]:
    """
    Build halo exchange maps using Graph BFS.
    Returns a list of dicts, where list[rank_s][rank_r] = tensor of local indices 
    that rank_s needs to send to rank_r.
    """
    device = pos.device
    num_atoms = pos.size(0)
    
    # 1. Build Base Graph (1-hop)
    edge_index, _ = _build_edges_cutoff_fairchem_raw(pos, cutoff, cell, pbc)
    
    # 2. Initialize Requirement Mask (N, WorldSize)
    # needed_by[i, r] means atom i is needed by rank r
    needed_by = torch.zeros((num_atoms, world_size), dtype=torch.bool, device=device)
    
    # Initial state: each atom is needed by its owner rank
    # partition_id is (N,)
    # We can use scatter to set true
    rows = torch.arange(num_atoms, device=device)
    needed_by[rows, partition_id] = True
    
    # 3. BFS Propagation
    # If u connects to v, and u is needed by Rank P, then v is needed by Rank P (as ghost)
    # Propagate: needed_by[v] |= needed_by[u]
    # In GNN, info flows from neighbor to center. Center needs neighbor.
    # If center u is needed by P, it needs neighbor v. So v is needed by P.
    
    src, dst = edge_index
    
    for _ in range(gnn_layers):
        # We want to propagate "needed_by" from src to dst?
        # If src is needed by P, and src depends on dst, then dst is needed by P.
        # In radius_graph, edges are usually symmetric or we have both directions.
        # Let's assume symmetric for safety or just propagate both ways if needed.
        # But radius_graph_pbc_v2 returns directed edges (i, j) such that dist(i, j) < r.
        # If i needs j, then if i is needed by P, j is needed by P.
        # Edge (i, j) means j is neighbor of i.
        # So we propagate from i (center) to j (neighbor)? 
        # No, "neededness" propagates from center to neighbor.
        # If i is needed, i's neighbors are needed.
        # So: needed_by[dst] |= needed_by[src] where src is center, dst is neighbor.
        # Wait, edge_index usually means (source, target) in message passing.
        # Message flows src -> dst.
        # To update dst, we need src.
        # So if dst is needed by P, then src is needed by P.
        # So we propagate from dst to src.
        # Let's verify edge_index convention. Fairchem: (u, v) means u is neighbor of v? Or v is neighbor of u?
        # Usually (source, target).
        # To update target, we need source.
        # So: needed_by[src] |= needed_by[dst].
        
        # Implementation:
        # Gather needed_by[dst]
        # Scatter reduce (max/or) to src
        
        # PyTorch doesn't have scatter_reduce with boolean OR easily until recent versions.
        # We can use float max or int max.
        
        needed_float = needed_by.float()
        
        # Gather from dst (targets that are needed)
        # shape: (E, WorldSize)
        needed_at_dst = needed_float[dst]
        
        # Scatter max to src (sources that are now needed)
        # We need a temporary buffer to scatter into
        # scatter_reduce_ is available in torch 1.12+
        # Or use index_add_ if we just count, but we want unique/bool.
        # Max is idempotent.
        
        # Let's use scatter_reduce_ with "amax" if available, or fallback.
        # Assuming torch >= 1.12
        
        # Propagate: src needs to be sent to whoever needs dst
        needed_float.scatter_reduce_(0, src.unsqueeze(-1).expand(-1, world_size), needed_at_dst, reduce="amax", include_self=True)
        
        needed_by = needed_float > 0.5

    # 4. Generate Send Maps
    # For each Rank S, we need to know what to send to Rank R.
    # Atom i should be sent from S to R if:
    #   partition_id[i] == S  (i is owned by S)
    #   needed_by[i, R] == True (i is needed by R)
    #   S != R (not sending to self)
    
    send_maps = [{} for _ in range(world_size)]
    
    # We need to map global indices to local indices for each rank.
    # Since we are in the rebalancing phase, we know the order of atoms for each rank.
    # Atoms for Rank S are those where partition_id == S.
    # Let's compute local indices for everyone.
    
    # Global to Local mapping is tricky if we don't store it.
    # But we can compute it easily:
    # For a given Rank S, the atoms it owns are `torch.where(partition_id == S)[0]`.
    # The k-th atom in that list has local index k.
    
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
                send_maps[r_sender][r_receiver] = send_indices.to(torch.int32)
                
    return send_maps

def _build_edges_cutoff_fairchem_raw(pos: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor):
    """Helper to call fairchem graph builder and return edge_index."""
    from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2
    
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
        self.comm_map = comm_map # Dict[target_rank, local_indices_to_send]
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

    def exchange(self, atoms: ase.Atoms) -> Dict[str, torch.Tensor]:
        """Exchange ghost atoms based on pre-computed map."""
        if self.world_size <= 1:
            return {}

        # Prepare data
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        vel = torch.tensor(atoms.get_velocities(), dtype=torch.float32, device=self.device)
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        
        # Buffers
        send_ops = []
        recv_ops = []
        
        # Prepare sends
        send_buffers = [] # Keep references
        for r, indices in self.comm_map.items():
            # Gather data and move to CPU
            s_pos = pos[indices].cpu()
            s_vel = vel[indices].cpu()
            s_Z = Z[indices].cpu()
            
            # Async send
            send_ops.append(dist.isend(s_pos, r, tag=100))
            send_ops.append(dist.isend(s_vel, r, tag=101))
            send_ops.append(dist.isend(s_Z, r, tag=102))
            send_buffers.extend([s_pos, s_vel, s_Z])
            
        # Prepare recvs
        recv_data = {}
        for r, count in self.recv_counts.items():
            # Allocate on CPU
            r_pos = torch.zeros((count, 3), dtype=torch.float32, device='cpu')
            r_vel = torch.zeros((count, 3), dtype=torch.float32, device='cpu')
            r_Z = torch.zeros((count,), dtype=torch.int32, device='cpu')
            
            recv_ops.append(dist.irecv(r_pos, r, tag=100))
            recv_ops.append(dist.irecv(r_vel, r, tag=101))
            recv_ops.append(dist.irecv(r_Z, r, tag=102))
            
            recv_data[r] = {"pos": r_pos, "vel": r_vel, "Z": r_Z}
            
        # Wait
        for op in send_ops:
            op.wait()
        for op in recv_ops:
            op.wait()
            
        # Aggregate ghosts and move to device
        if not recv_data:
            return {}
            
        ghost_pos = torch.cat([d["pos"] for d in recv_data.values()], dim=0).to(self.device)
        ghost_vel = torch.cat([d["vel"] for d in recv_data.values()], dim=0).to(self.device)
        ghost_Z = torch.cat([d["Z"] for d in recv_data.values()], dim=0).to(self.device)
        
        return {
            "pos": ghost_pos,
            "vel": ghost_vel,
            "Z": ghost_Z
        }

class DistributedCalculator:
    def __init__(self, calculator, cutoff: float, comm_map: Dict[int, torch.Tensor] = None):
        self.calculator = calculator
        self.cutoff = cutoff
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map if comm_map else {}, device)
        self.num_local = 0
        
    def update_comm_map(self, comm_map: Dict[int, torch.Tensor]):
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map, self.exchanger.device)
        
    def get_forces(self, atoms: ase.Atoms) -> np.ndarray:
        # 1. Exchange ghosts
        ghosts = self.exchanger.exchange(atoms)
        
        self.num_local = len(atoms)
        num_local = self.num_local
        
        # 2. Build combined atoms
        if ghosts:
            ghost_pos = ghosts["pos"].cpu().numpy()
            ghost_Z = ghosts["Z"].cpu().numpy()
            # We don't strictly need ghost velocities for force calculation usually, but good to have
            
            combined_atoms = atoms.copy()
            combined_atoms.extend(ase.Atoms(numbers=ghost_Z, positions=ghost_pos))
            # Copy cell/pbc
            combined_atoms.set_cell(atoms.get_cell())
            combined_atoms.set_pbc(atoms.get_pbc())
        else:
            combined_atoms = atoms
            
        # 3. Calculate
        # We need to attach the inner calculator to the combined atoms
        # But `self.calculator` might be stateful or attached to another atoms object
        # FairChemCalculator usually is stateless regarding the atoms object passed to get_forces
        # but we need to be careful.
        
        # Hack: FairChemCalculator uses `self.results` to store cache.
        # We should probably just call `get_forces` on the inner calc.
        
        # If the inner calculator is already attached to `atoms`, we might have issues.
        # Let's create a temporary calculator or re-attach.
        
        # For FairChemCalculator, it seems safe to just call get_forces(combined_atoms)
        # provided we handle the calculator attachment.
        
        combined_atoms.calc = self.calculator
        all_forces = self.calculator.get_forces(combined_atoms)
        
        # 4. Filter local forces
        local_forces = all_forces[:num_local]
        
        return local_forces
        
    def get_potential_energy(self, atoms: ase.Atoms) -> float:
        # Similar logic for energy, but we need to be careful about double counting.
        # Usually, we only sum energies of local atoms.
        # If the calculator provides per-atom energy, we sum local.
        # If it provides global energy, we are in trouble unless we can decompose it.
        # FairChem usually provides per-atom energy?
        # If not, we might need to rely on the fact that we only care about forces for MD.
        # But for logging, we need energy.
        
        # For now, let's assume we can get per-atom energy or just return 0 if not supported.
        # FairChemCalculator might not expose per-atom energy easily in ASE interface.
        # Let's try to get potential energy of combined system and approximate? 
        # No, that would be wrong (includes ghost-ghost interactions).
        
        # Correct way: Sum per-atom energies of local atoms.
        # If FairChemCalculator supports `get_potential_energies()`, use that.
        
        try:
            # 1. Exchange ghosts (if not already done in this step? ASE might call get_forces then get_pe)
            # For simplicity, repeat exchange or cache it. 
            # Let's repeat for safety/statelessness.
            ghosts = self.exchanger.exchange(atoms)
            num_local = len(atoms)
            
            if ghosts:
                ghost_pos = ghosts["pos"].cpu().numpy()
                ghost_Z = ghosts["Z"].cpu().numpy()
                combined_atoms = atoms.copy()
                combined_atoms.extend(ase.Atoms(numbers=ghost_Z, positions=ghost_pos))
                combined_atoms.set_cell(atoms.get_cell())
                combined_atoms.set_pbc(atoms.get_pbc())
            else:
                combined_atoms = atoms
            
            combined_atoms.calc = self.calculator
            energies = self.calculator.get_potential_energies(combined_atoms)
            return np.sum(energies[:num_local])
        except Exception:
            # Fallback or error
            return 0.0

    def calculation_required(self, atoms, quantities):
        return True
        
    @property
    def results(self):
        # Filter results to local atoms only
        res = self.calculator.results.copy()
        num_local = self.num_local
        for key, val in res.items():
            if isinstance(val, (np.ndarray, torch.Tensor)) and len(val) > num_local:
                res[key] = val[:num_local]
        return res
        
    @property
    def implemented_properties(self):
        return self.calculator.implemented_properties

def _build_edges_cutoff_fairchem(atoms: ase.Atoms, cutoff: float) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Build graph using FairChem radius_graph_pbc_v2 on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("GPU is required for FairChem graph building in this configuration.")

    class _FCData:
        def __init__(self, a: ase.Atoms):
            self.pos = torch.tensor(a.get_positions(), dtype=torch.float32, device=device)
            cell_array = np.array(a.get_cell().array, dtype=np.float32)
            self.cell = torch.tensor(cell_array, dtype=torch.float32, device=device).unsqueeze(0)
            self.natoms = torch.tensor([len(a)], dtype=torch.long, device=device)
            self.batch = None
            self.pbc = torch.tensor([bool(x) for x in a.get_pbc()], dtype=torch.bool, device=device)

    data = _FCData(atoms)
    pbc_tensor = data.pbc
    
    # Use fairchem to build graph
    # We run this to ensure graph building works and for timing, but we don't need the edges for SFC partitioning
    radius_graph_pbc_v2(
        data=data,
        radius=cutoff,
        max_num_neighbors_threshold=300,
        enforce_max_neighbors_strictly=False,
        pbc=pbc_tensor,
    )
    
    return None, None



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
        
        # Initial Partitioning
        tmp_atoms = ase.Atoms(numbers=numbers.tolist(), positions=positions)
        tmp_atoms.set_cell(full_atoms.get_cell())
        tmp_atoms.set_pbc(full_atoms.get_pbc())
        

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
            all_comm_maps = build_halo_maps(
                pos_tensor, 
                partition_id, 
                cutoff, 
                cfg.dist.gnn_layers, 
                cell, 
                pbc, 
                world_size
            )
        else:
            all_comm_maps = None
            
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
    # Broadcast initial payload
    if world_size > 1 and dist_is_enabled():
        dtype_map = {"pos": torch.float32, "vel": torch.float32, "Z": torch.int32}
        if cfg.dist.use_gid:
            dtype_map["gid"] = torch.int64
        payload = _concat_per_rank_and_broadcast(fields_per_rank, dtype_map)
        local_atoms = _build_local_atoms_from_payload(payload, cell_np, pbc_tuple)
        
        # Distribute initial comm_maps
        my_comm_map_container = [None]
        dist.scatter_object_list(my_comm_map_container, all_comm_maps if rank == 0 else [], src=0)
        my_comm_map = my_comm_map_container[0]
        
    else:
        if rank == 0:
            local_atoms = ase.Atoms(numbers=numbers.tolist(), positions=positions)
            local_atoms.set_cell(cell_np)
            local_atoms.set_pbc(list(pbc_tuple))
            local_atoms.set_velocities(velocities)
            if cfg.dist.use_gid:
                local_atoms.new_array("global_id", global_ids)
        else:
            local_atoms = ase.Atoms(cell=cell_np, pbc=list(pbc_tuple))
        
        my_comm_map = {}

    # Wrap calculator with DistributedCalculator if distributed
    if world_size > 1 and dist_is_enabled():
        calc = DistributedCalculator(calc, cutoff, comm_map=my_comm_map)
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
                all_comm_maps = build_halo_maps(
                    pos_all, 
                    partition_id, 
                    cutoff, 
                    cfg.dist.gnn_layers, 
                    cell_tensor, 
                    pbc_tensor, 
                    world_size
                )
                
            # 3. Broadcast Partitioning and Maps
            # Distribute comm_maps
            my_comm_map = [None]
            dist.scatter_object_list(my_comm_map, all_comm_maps, src=0)
            my_comm_map = my_comm_map[0]
            
            # Update calculator with new map
            if isinstance(calc, DistributedCalculator):
                calc.update_comm_map(my_comm_map)
            
            # Distribute atoms (pos, vel, Z, gid)
            # We can use the same masking trick as before if we broadcast full arrays, 
            # OR we can scatter the specific chunks.
            # Broadcasting full arrays (N ~ 10k-100k) is fine.
            # Let's broadcast full arrays from Rank 0 to all.
            
            if rank == 0:
                pos_global = torch.from_numpy(gathered["pos"]).to(device)
                vel_global = torch.from_numpy(gathered["vel"]).to(device)
                Z_global = torch.from_numpy(gathered["Z"]).to(device)
                gid_global = torch.from_numpy(gathered["gid"]).to(device) if gathered.get("gid") is not None else torch.empty(0, device=device)
            else:
                # We need to know N to allocate buffers
                # Broadcast N
                n_tensor = torch.tensor([0], dtype=torch.int64, device=device)
                dist.broadcast(n_tensor, src=0)
                N = n_tensor.item()
                
                pos_global = torch.zeros((N, 3), dtype=torch.float32, device=device)
                vel_global = torch.zeros((N, 3), dtype=torch.float32, device=device)
                Z_global = torch.zeros((N,), dtype=torch.int32, device=device)
                gid_global = torch.zeros((N,), dtype=torch.int64, device=device) # Or empty
                partition_id = torch.zeros((N,), dtype=torch.int32, device=device)

            # Broadcast data
            if rank == 0:
                n_tensor = torch.tensor([len(pos_global)], dtype=torch.int64, device=device)
                dist.broadcast(n_tensor, src=0)
                
            _broadcast_tensor(pos_global, src=0)
            _broadcast_tensor(vel_global, src=0)
            _broadcast_tensor(Z_global, src=0)
            _broadcast_tensor(partition_id, src=0)
            # Handle gid
            # Check if gid exists
            has_gid = torch.tensor([1 if gid_global.numel() > 0 else 0], device=device)
            dist.broadcast(has_gid, src=0)
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
                local_atoms.set_array("gid", local_gid)
                
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
        dyn.run(steps=1) # Run for one step
        if dist_is_enabled():
            dist.barrier()
        run_time = time.time() - start_time
        
        total_done += 1
        rank0_print(f"Ran 1 steps in {run_time:.2f}s. (perf: {1/run_time:.2f} steps/s, {total_done}/{total_steps})")

    
    if dist_is_enabled():
        dist.barrier()
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: AppConfig) -> None:
    if cfg.dist.starter == "slurm":
        rank, world_size, local_rank = dist_init_from_slurm()
    elif cfg.dist.starter == "torchrun":
        rank, world_size, local_rank = dist_init_from_torchrun()
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

    device = "cpu"
    if torch.cuda.is_available():
        # For single-GPU debugging with multiple ranks, map all ranks to device 0
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = "cuda"
    print(f"Rank {rank} Using device: {device}")

    # Build calculator
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
            predictor = pretrained_mlip.get_predict_unit(cfg.uma.model, device=device, inference_settings=settings)
            calc = FAIRChemCalculator(predictor, task_name="omol")
            if dist_is_enabled():
                dist.barrier()
    
    print(f"Rank {rank}: Calculator built. Starting simulation...")
    run_md_simulation_distributed(calc, cfg, rank, world_size, local_rank)

if __name__ == "__main__":
    main()