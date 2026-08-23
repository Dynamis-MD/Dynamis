from __future__ import annotations
import torch
import typing
__all__: list[str] = ['DistEnvironment', 'DistNNPState', 'KDTreePartitioner', 'MolecularInfo', 'PartitionMessagePassingInfo', 'SFCCurveType', 'SFCPartitioner', 'SimpleSpatialPartitioner', 'SimulationConfig', 'TrajectoryWriter', 'extract_halo_info', 'filter_edges_to_radius', 'radius_graph_pbc', 'radius_graph_pbc_counts_ex', 'radius_graph_pbc_ex']
class DistEnvironment:
    global_rank: int
    gpus_per_node: int
    local_rank: int
    num_nodes: int
    world_size: int
    def __init__(self) -> None:
        ...
class DistNNPState:
    dist_env: DistEnvironment
    global_molecular_info: MolecularInfo
    local_molecular_info: MolecularInfo
    local_partition_info: PartitionMessagePassingInfo
    sim_config: SimulationConfig
    def __init__(self) -> None:
        ...
class KDTreePartitioner:
    def __init__(self) -> None:
        ...
    def build_halo_maps(self, arg0: PartitionMessagePassingInfo, arg1: MolecularInfo, arg2: DistEnvironment, arg3: SimulationConfig) -> None:
        ...
    @typing.overload
    def partition(self, arg0: MolecularInfo, arg1: DistEnvironment) -> PartitionMessagePassingInfo:
        """
        Partition: unweighted (default) or user-weighted (if edge_counts set)
        """
    @typing.overload
    def partition(self, arg0: MolecularInfo, arg1: DistEnvironment, arg2: SimulationConfig) -> PartitionMessagePassingInfo:
        """
        Partition: auto-compute edge weights via radius_graph_pbc (CUDA)
        """
class MolecularInfo:
    atom_forces: torch.Tensor
    atom_positions: torch.Tensor
    atom_types: torch.Tensor
    atom_velocities: torch.Tensor
    atom_vertex_numbers: torch.Tensor
    edge_counts: torch.Tensor
    num_atoms: int
    def __init__(self) -> None:
        ...
class PartitionMessagePassingInfo:
    edge_index: torch.Tensor
    edge_index_halo: torch.Tensor
    global_send_recv_matrix: torch.Tensor
    interior_mask: torch.Tensor
    partition_id: torch.Tensor
    partition_indices: torch.Tensor
    recv_ghost_indices: torch.Tensor
    recv_offsets: torch.Tensor
    send_local_indices: torch.Tensor
    send_offsets: torch.Tensor
    def __init__(self) -> None:
        ...
class SFCCurveType:
    """
    Members:
    
      Morton
    
      Hilbert
    """
    Hilbert: typing.ClassVar[SFCCurveType]  # value = <SFCCurveType.Hilbert: 1>
    Morton: typing.ClassVar[SFCCurveType]  # value = <SFCCurveType.Morton: 0>
    __members__: typing.ClassVar[dict[str, SFCCurveType]]  # value = {'Morton': <SFCCurveType.Morton: 0>, 'Hilbert': <SFCCurveType.Hilbert: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SFCPartitioner:
    def __init__(self, curve_type: SFCCurveType = ...) -> None:
        ...
    def build_halo_maps(self, arg0: PartitionMessagePassingInfo, arg1: MolecularInfo, arg2: DistEnvironment, arg3: SimulationConfig) -> None:
        ...
    @typing.overload
    def partition(self, arg0: MolecularInfo, arg1: DistEnvironment) -> PartitionMessagePassingInfo:
        """
        Partition: unweighted (default) or user-weighted (if edge_counts set)
        """
    @typing.overload
    def partition(self, arg0: MolecularInfo, arg1: DistEnvironment, arg2: SimulationConfig) -> PartitionMessagePassingInfo:
        """
        Partition: auto-compute edge weights via radius_graph_pbc (CUDA)
        """
class SimpleSpatialPartitioner:
    def __init__(self) -> None:
        ...
    def build_halo_maps(self, arg0: PartitionMessagePassingInfo, arg1: MolecularInfo, arg2: DistEnvironment, arg3: SimulationConfig) -> None:
        ...
    def partition(self, arg0: MolecularInfo, arg1: DistEnvironment) -> PartitionMessagePassingInfo:
        ...
class SimulationConfig:
    cell: torch.Tensor
    cutoff: float
    ghost_cutoff_buffer: float
    pbc: torch.Tensor
    def __init__(self) -> None:
        ...
class TrajectoryWriter:
    def __enter__(self) -> TrajectoryWriter:
        ...
    def __exit__(self, *args) -> None:
        ...
    def __init__(self, global_rank: int, run_id: bytes, author: str, output_dir: str = './trajectory') -> None:
        ...
    def initialize(self, input_file_name: str, time_per_step: float, steps_per_write: int) -> MolecularInfo:
        """
        Writes front matter and initializes environment from an xyz file
        """
    def update_atom_counts(self, counts: list[int]) -> None:
        """
        Update the rank offsets. Should be called after every repartition
        """
    def write_frame(self, frame_idx: int, indices: torch.Tensor, positions: torch.Tensor, velocities: torch.Tensor) -> None:
        """
        Writes the frame to the binary file
        """
    def write_to_xyz(self, state: MolecularInfo, file_name: str) -> None:
        """
        Writes state to an xyz file
        """
def extract_halo_info(edges: torch.Tensor, pid_outer: torch.Tensor, my_outer_atoms: torch.Tensor, my_rank: int, world_size: int, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Derive halo send/recv CSR + interior mask from per-rank edges (dst in my_outer) via graph symmetry, grouped by peer and sorted by global atom id within each peer (no NCCL). Returns (send_offsets, send_local_indices, recv_offsets, recv_ghost_indices, interior_mask, recv_atom_global). step 7.
    """
def filter_edges_to_radius(edges: torch.Tensor, cell_shifts: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter edges built at r+skin down to true cutoff r: keep edge iff ||pos[src] + cell_shift @ cell - pos[dst]||^2 <= r^2. Order-preserving. Fused CUDA kernel + compaction; replaces the ATen recompute (step 8).
    """
def radius_graph_pbc(pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, radius: float, max_num_neighbors: int, enforce_max_strictly: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA cell-list radius graph with selective PBC tiling
    """
def radius_graph_pbc_counts_ex(pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, radius: float, target_indices: torch.Tensor | None = None) -> torch.Tensor:
    """
    Count-only variant: returns just neighbor_counts[N_target] int32. Skips edge buffer + prefix-sum + B4 write kernel (~50% faster).
    """
def radius_graph_pbc_ex(pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, radius: float, max_num_neighbors: int, enforce_max_strictly: bool, target_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extended radius graph: accepts optional target_indices subset, returns (edge_index, cell_shifts, num_neighbors_image, neighbor_counts).
    """
