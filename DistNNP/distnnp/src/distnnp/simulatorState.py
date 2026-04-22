from distnnp_state import (
    DistNNPState,
    DistEnvironment,
    SimulationConfig,
    PartitionMessagePassingInfo,
    MolecularInfo,
    SFCPartitioner,
    SimpleSpatialPartitioner,
    KDTreePartitioner,
)
from distnnp_state.masses import TORCH_ATOMIC_MASSES
import torch
import torch.distributed as dist
from typing import Union, Literal, Optional


ABSTRACT_PARTITIONERS = Union[
    SFCPartitioner, SimpleSpatialPartitioner, KDTreePartitioner
]


class SimState:
    def __init__(
        self,
        molecular_system: MolecularInfo,
        partitioner: ABSTRACT_PARTITIONERS,
        simulation_config: SimulationConfig,
        distributed_environment: DistEnvironment,
        device: Optional[torch.device] = None,
    ):

        self.device = device

        # Initialize distributed environment and set device
        self.dist_env_obj = distributed_environment
        self.initialize_distributed_and_device()

        simulation_config.cell = simulation_config.cell.to(self.device)
        simulation_config.pbc = simulation_config.pbc.to(self.device)

        self.partitioner = partitioner

        partition_info = partitioner.partition(molecular_system, self.dist_env_obj)
        partitioner.build_halo_maps(
            partition_info, molecular_system, self.dist_env_obj, simulation_config
        )
        self.parition_info = partition_info

        self.global_molecular_info = molecular_system
        self._local_molecular_info = self.setup_mass_and_forces(
            partition_info=partition_info,
            molecular_system=molecular_system,
        )

        self._state = DistNNPState()
        self._state.dist_env = self.dist_env_obj
        self._state.sim_config = simulation_config
        self._state.global_molecular_info = molecular_system
        self._state.local_partition_info = partition_info

    def setup_mass_and_forces(self, partition_info, molecular_system):
        local_molecular_info = MolecularInfo()
        local_molecular_info.atom_positions = molecular_system.atom_positions[
            partition_info.partition_indices
        ]
        local_molecular_info.atom_velocities = molecular_system.atom_velocities[
            partition_info.partition_indices
        ]
        local_molecular_info.atom_masses = molecular_system.atom_masses[
            partition_info.partition_indices
        ]
        local_molecular_info.atom_types = molecular_system.atom_types[
            partition_info.partition_indices
        ]
        num_local_atoms = local_molecular_info.atom_positions.shape[0]
        local_molecular_info.atom_forces = torch.zeros(
            num_local_atoms, 3, device=self.device
        )
        local_molecular_info.num_atoms = num_local_atoms

        return local_molecular_info

    def initialize_distributed_and_device(self):
        local_rank = self.dist_env_obj.local_rank
        global_rank = self.dist_env_obj.global_rank
        world_size = self.dist_env_obj.world_size
        num_nodes = self.dist_env_obj.num_nodes
        gpus_per_node = self.dist_env_obj.gpus_per_node
        num_gpus = torch.cuda.device_count()

        device = (
            torch.device(f"cuda:{local_rank}" if num_gpus > 0 else "cpu")
            if self.device is None
            else self.device
        )

        self.device = device

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.is_distributed = world_size > 1

    @property
    def pos(self) -> torch.Tensor:
        return self._local_molecular_info.atom_positions

    @property
    def vel(self) -> torch.Tensor:
        return self._state.local_partition_info.local_atom_velocities

    @property
    def sim_config(self) -> SimulationConfig:
        return self._state.sim_config

    @property
    def dist_env(self) -> DistEnvironment:
        return self._state.dist_env

    @property
    def local_state(self) -> MolecularInfo:
        return self._local_molecular_info

    @property
    def local_partition_info(self) -> PartitionMessagePassingInfo:
        return self._state.local_partition_info

    def sync_and_repartition(
        self,
        paritioner: Optional[ABSTRACT_PARTITIONERS] = None,
        comm_backend: Union[Literal["nccl"], Literal["gloo"]] = "nccl",
    ) -> None:
        dist_env = self.dist_env
        partition_info = self._state.local_partition_info

        local_indices = (
            partition_info.partition_indices
        )  # [num_local] — global atom IDs
        local_positions = partition_info.local_atom_positions  # [num_local, 3]
        local_velocities = partition_info.local_atom_velocities  # [num_local, 3]

        global_mol_info = self._state.global_molecular_info

        if self.world_size == 1:
            # No communication needed; update global info in place.
            global_mol_info.atom_positions[local_indices] = local_positions
            global_mol_info.atom_velocities[local_indices] = local_velocities
        else:
            # Collect (indices, positions, velocities) from every rank.
            local_data = {
                "indices": local_indices,
                "positions": local_positions,
                "velocities": local_velocities,
            }
            com_device = self.device if comm_backend == "nccl" else torch.device("cpu")
            local_data = {k: v.to(com_device) for k, v in local_data.items()}
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local_data)

            assert all(
                [g is not None for g in gathered]
            ), "Gather failed: some ranks did not contribute data."

            all_indices = torch.cat([g["indices"] for g in gathered])  # type: ignore # [N]
            all_positions = torch.cat([g["positions"] for g in gathered])  # type: ignore  # [N, 3]
            all_velocities = torch.cat([g["velocities"] for g in gathered])  # type: ignore  # [N, 3]

            # Restore canonical atom order (0 … N-1).
            sort_order = torch.argsort(all_indices)
            global_mol_info.atom_positions = all_positions[sort_order].to(self.device)
            global_mol_info.atom_velocities = all_velocities[sort_order].to(self.device)

        self.partitioner = paritioner if paritioner is not None else self.partitioner
        # Re-partition on the updated positions and rebuild halo maps.
        new_partition_info = self.partitioner.partition(global_mol_info, dist_env)
        self.partitioner.build_halo_maps(
            new_partition_info, global_mol_info, dist_env, self.sim_config
        )

        self._state.local_partition_info = new_partition_info
        self._state.global_molecular_info = global_mol_info

        self.setup_mass_and_forces(
            partition_info=new_partition_info, molecular_system=global_mol_info
        )
