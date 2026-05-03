from __future__ import annotations
import torch
import typing
__all__: list[str] = ['DistEnvironment', 'DistNNPState', 'KDTreePartitioner', 'MolecularInfo', 'PartitionMessagePassingInfo', 'SFCCurveType', 'SFCPartitioner', 'SimpleSpatialPartitioner', 'SimulationConfig', 'radius_graph_pbc']
class DistEnvironment:
    def __init__(self) -> None:
        ...
    @property
    def global_rank(self) -> int:
        ...
    @global_rank.setter
    def global_rank(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def gpus_per_node(self) -> int:
        ...
    @gpus_per_node.setter
    def gpus_per_node(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def local_rank(self) -> int:
        ...
    @local_rank.setter
    def local_rank(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def num_nodes(self) -> int:
        ...
    @num_nodes.setter
    def num_nodes(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def world_size(self) -> int:
        ...
    @world_size.setter
    def world_size(self, arg0: typing.SupportsInt) -> None:
        ...
class DistNNPState:
    dist_env: DistEnvironment
    global_molecular_info: MolecularInfo
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
    atom_vertex_numbers: torch.Tensor
    atom_types: torch.Tensor
    atom_positions: torch.Tensor
    atom_velocities: torch.Tensor
    edge_counts: torch.Tensor
    def __init__(self) -> None:
        ...
    @property
    def num_atoms(self) -> int:
        ...
    @num_atoms.setter
    def num_atoms(self, arg0: typing.SupportsInt) -> None:
        ...
class PartitionMessagePassingInfo:
    edge_index: torch.Tensor
    edge_index_halo: torch.Tensor
    global_send_recv_matrix: torch.Tensor
    interior_mask: torch.Tensor
    local_atom_positions: torch.Tensor
    local_atom_velocities: torch.Tensor
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
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
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
    pbc: torch.Tensor
    def __init__(self) -> None:
        ...
    @property
    def cutoff(self) -> float:
        ...
    @cutoff.setter
    def cutoff(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def ghost_cutoff_buffer(self) -> float:
        ...
    @ghost_cutoff_buffer.setter
    def ghost_cutoff_buffer(self, arg0: typing.SupportsFloat) -> None:
        ...
def radius_graph_pbc(pos: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, radius: typing.SupportsFloat, max_num_neighbors: typing.SupportsInt, enforce_max_strictly: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA cell-list radius graph with selective PBC tiling
    """

def edge_distances(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA pairwise atom distance calculation
    """
