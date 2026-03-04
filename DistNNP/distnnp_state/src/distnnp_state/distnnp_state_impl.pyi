from __future__ import annotations
import torch
import typing
__all__: list[str] = ['DistEnvironment', 'DistNNPState', 'MolecularInfo', 'PartitionMessagePassingInfo', 'SimulationConfig', 'SimpleSpatialPartitioner']
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
    local_partition_info: MolecularInfo
    def __init__(self) -> None:
        ...
class MolecularInfo:
    atom_indices: torch.Tensor
    atom_positions: torch.Tensor
    atom_velocities: torch.Tensor
    def __init__(self) -> None:
        ...
    @property
    def num_atoms(self) -> int:
        ...
    @num_atoms.setter
    def num_atoms(self, arg0: typing.SupportsInt) -> None:
        ...
class SimulationConfig:
    cutoff: float
    ghost_cutoff_buffer: float
    cell: torch.Tensor
    pbc: torch.Tensor
    def __init__(self) -> None:
        ...
class PartitionMessagePassingInfo:
    partition_id: torch.Tensor
    global_send_recv_matrix: torch.Tensor
    local_atom_positions: torch.Tensor
    local_atom_velocities: torch.Tensor
    partition_indices: torch.Tensor
    send_offsets: torch.Tensor
    send_local_indices: torch.Tensor
    recv_offsets: torch.Tensor
    recv_ghost_indices: torch.Tensor
    interior_mask: torch.Tensor
    def __init__(self) -> None:
        ...
class SimpleSpatialPartitioner:
    def __init__(self) -> None:
        ...
    def partition(self, state: MolecularInfo, env: DistEnvironment) -> PartitionMessagePassingInfo:
        ...
    def build_halo_maps(self, info: PartitionMessagePassingInfo, global_state: MolecularInfo, env: DistEnvironment, config: SimulationConfig) -> None:
        ...
