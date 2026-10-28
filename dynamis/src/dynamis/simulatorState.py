from dynamis_state import (
    DynamisState,
    DistEnvironment,
    SimulationConfig,
    PartitionMessagePassingInfo,
    MolecularInfo,
    deform_state,
    set_state_cell,
    SFCPartitioner,
    SimpleSpatialPartitioner,
    KDTreePartitioner,
)
import torch
import torch.distributed as dist
from typing import Union, Literal, Optional

# Bits of the per-rank drift word (see SimState.write_local_drift_flag).
DRIFT_TRIP = 1  # rebuild the halo before the next step
DRIFT_BREACH = 2  # this step's graph was built past the displacement guarantee.
# Debug only: never expected with the bit-0 rule; kept as a cheap
# check for future debugging (e.g. a new integrator or a larger dt).

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
        self.partition_info = partition_info

        self.global_molecular_info = molecular_system

        self._state = DynamisState()
        self._state.dist_env = self.dist_env_obj
        self._state.sim_config = simulation_config
        self._state.global_molecular_info = molecular_system
        self._state.local_partition_info = partition_info
        self._state.local_molecular_info = self.setup_mass_and_forces(
            partition_info=partition_info,
            molecular_system=molecular_system,
        )

        # Drift-watchdog reference: this rank's local positions at the last
        # repartition. The local atom set and its order stay fixed until the next
        # repartition (the integrator writes in place; nothing reorders), so drift is
        # an elementwise difference against this snapshot.
        self._ref_positions: Optional[torch.Tensor] = None
        self._prev_velocities: Optional[torch.Tensor] = None
        # Inverse of the cell the halo was built with (float64, on device): the
        # watchdog needs the affine map A = h_ref^-1 h_now to tell a deformation
        # of the box from motion of the atoms.
        self._ref_cell_inv: Optional[torch.Tensor] = None
        # Upper bound, in ||A - I||_F, on how much the cell may still change during
        # the next step. A barostat sets it so the watchdog's one-step look-ahead
        # covers its own box update; 0 for a fixed cell.
        self.cell_change_bound_per_step: float = 0.0
        self.snapshot_reference_positions()

    def setup_mass_and_forces(self, partition_info, molecular_system):
        local_molecular_info = MolecularInfo()
        local_molecular_info.atom_positions = molecular_system.atom_positions[
            partition_info.partition_indices
        ]
        local_molecular_info.atom_velocities = molecular_system.atom_velocities[
            partition_info.partition_indices
        ]
        local_molecular_info.atom_types = molecular_system.atom_types[
            partition_info.partition_indices
        ]
        num_local_atoms = local_molecular_info.atom_positions.shape[0]
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
        return self._state.local_molecular_info.atom_positions

    @property
    def vel(self) -> torch.Tensor:
        return self._state.local_molecular_info.atom_velocities

    @property
    def sim_config(self) -> SimulationConfig:
        return self._state.sim_config

    @property
    def dist_env(self) -> DistEnvironment:
        return self._state.dist_env

    @property
    def local_state(self) -> MolecularInfo:
        return self._state.local_molecular_info

    @property
    def local_partition_info(self) -> PartitionMessagePassingInfo:
        return self._state.local_partition_info

    def deform(self, transform: torch.Tensor) -> None:
        """Apply ``x -> x A`` to the local positions and ``h -> h A`` to the cell.
        See :func:`dynamis_state.deform_state`; the halo stays valid through the
        drift watchdog, and repartitioning stays on its own schedule."""
        deform_state(self._state, transform)

    def set_cell(self, cell: torch.Tensor) -> None:
        """Overwrite the cell without moving atoms (see
        :func:`dynamis_state.set_state_cell`)."""
        set_state_cell(self._state, cell)

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
        local_positions = (
            self._state.local_molecular_info.atom_positions
        )  # [num_local, 3]
        local_velocities = (
            self._state.local_molecular_info.atom_velocities
        )  # [num_local, 3]

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

            all_indices = torch.cat([g["indices"].to(self.device) for g in gathered])  # type: ignore # [N]
            all_positions = torch.cat([g["positions"].to(self.device) for g in gathered])  # type: ignore  # [N, 3]
            all_velocities = torch.cat([g["velocities"].to(self.device) for g in gathered])  # type: ignore  # [N, 3]

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
        self.parition_info = new_partition_info

        self._state.local_molecular_info = self.setup_mass_and_forces(
            partition_info=new_partition_info, molecular_system=global_mol_info
        )
        self.snapshot_reference_positions()

    def sync_positions_and_rebuild_halo(self) -> None:
        """Halo-only rebuild: refresh the ghost lists without repartitioning.

        Ownership, the local atom set and its order, and velocities are unchanged. Only
        the positions are all-gathered (one tensor collective, no pickling), and
        build_halo_maps() is re-run on the existing partition (it depends only on
        partition_id and the global positions). Load balance is left to the scheduled
        sync_and_repartition(). Afterwards global_molecular_info holds current positions
        but the velocities of the last full sync (nothing reads them in between; the
        next full sync overwrites both)."""
        info = self._state.local_partition_info
        global_mol_info = self._state.global_molecular_info
        local_positions = self._state.local_molecular_info.atom_positions.detach()

        if self.world_size == 1:
            global_mol_info.atom_positions[info.partition_indices] = local_positions
        else:
            owner = info.partition_id
            # Every rank's atoms in ascending global-ID order: one stable sort, one sync.
            order = torch.argsort(owner, stable=True)
            counts = torch.bincount(owner, minlength=self.world_size).tolist()
            starts = [0]
            for c in counts:
                starts.append(starts[-1] + c)
            ids = [order[starts[r] : starts[r + 1]] for r in range(self.world_size)]
            # Local arrays follow partition_indices; build_halo_maps indexes local atoms
            # in ascending-ID order. Both must agree for the send lists to be right.
            assert torch.equal(
                ids[self.global_rank], info.partition_indices
            ), "partition_indices is not in ascending global-ID order"
            m = max(counts)
            buf = local_positions.new_zeros((m, 3))
            buf[: counts[self.global_rank]] = local_positions
            gathered = local_positions.new_empty((self.world_size * m, 3))
            dist.all_gather_into_tensor(gathered, buf)
            full = local_positions.new_empty(
                (global_mol_info.atom_positions.shape[0], 3)
            )
            for r in range(self.world_size):
                full[ids[r]] = gathered[r * m : r * m + counts[r]]
            global_mol_info.atom_positions = full

        self.partitioner.build_halo_maps(
            info, global_mol_info, self.dist_env, self.sim_config
        )
        self.snapshot_reference_positions()

    def snapshot_reference_positions(self) -> None:
        """Latch the current local positions as the drift reference (the positions the
        halo maps were just built from). Must be a clone: the integrator updates
        atom_positions in place. Also restarts the velocity history of the one-step
        bound in write_local_drift_flag."""
        info = self._state.local_molecular_info
        self._ref_positions = info.atom_positions.detach().clone()
        self._prev_velocities = info.atom_velocities.detach().clone()
        cell = self._state.sim_config.cell.detach().reshape(3, 3)
        self._ref_cell_inv = torch.linalg.inv(cell.to(torch.float64))

    def write_local_drift_flag(
        self, flag: torch.Tensor, threshold: float, dt: float
    ) -> None:
        """Write this rank's 2-bit drift word into ``flag`` (int32 [1]); device-only.

        Called on the positions the model is about to push, i.e. the positions this
        step's graph is built on. With u_i the displacement of atom i since the last
        halo build *after removing the affine map of the cell* (below), and the next
        step's motion bounded per atom by dt * (|v_i| + |v_i - v_i^prev|) (velocity
        plus its change over the last call, standing in for the force update):

          bit 0 (DRIFT_TRIP):   max_i u_i + dt * (|v_i| + |v_i - v_i^prev|) + cell
                                look-ahead >= budget
                                -> the next step's graph could exceed it: rebuild first
          bit 1 (DRIFT_BREACH): max_i u_i > budget
                                -> THIS step's graph is built past the guarantee

        With a fixed cell, u_i = |x_i - x_i^ref| and budget = threshold = skin / 2, and
        no edge can be missing while no rank sets bit 1. Positions are never wrapped
        (not by the integrator, the models or the exchanger), so the raw difference is
        the true displacement: no minimum-image fold.

        When the cell has changed since the halo build, A = h_ref^-1 h_now maps the
        build-time lattice to the current one, and a pair vector (periodic image
        included) evolves as r(t) = A r(0) + (u_j - u_i), where x_i(t) = A x_i^ref + u_i.
        So u_i = |x_i - x_i^ref A| is the displacement beyond the affine part, and
        |r(0)| <= |A^-1| (cutoff + 2 max u). Using |A^-1| <= 1 / (1 - delta) with
        delta = ||A - I||_F (>= ||A - I||_2), no edge within the cutoff can lie outside
        the ghost radius (cutoff + skin) while

            max u < budget = threshold - delta * (cutoff + skin) / 2.

        This is also what catches a cell change that leaves the atoms in place, which a
        plain displacement test cannot see: neighbours across a periodic boundary move
        closer by the change of the lattice vector. ``cell_change_bound_per_step`` adds a
        look-ahead for the box update the integrator may still apply before the next
        graph: it enters the TRIP test only."""
        info = self._state.local_molecular_info
        cur = info.atom_positions.detach()
        ref = self._ref_positions
        if ref is None or ref.shape != cur.shape:
            flag.fill_(DRIFT_TRIP)  # the local set changed under us: force a rebuild
            return
        if cur.shape[0] == 0:
            flag.zero_()
            return
        vel = info.atom_velocities.detach()

        config = self._state.sim_config
        halo_radius = config.cutoff + config.ghost_cutoff_buffer
        cell = config.cell.detach().reshape(3, 3).to(torch.float64)
        affine = self._ref_cell_inv.to(cell.device) @ cell - torch.eye(
            3, dtype=torch.float64, device=cell.device
        )  # A - I, ~1e-16 for an unchanged cell
        delta = affine.norm()
        budget = threshold - 0.5 * halo_radius * delta

        # beyond-affine displacement: x - x_ref A, with the identity part cancelled
        moved = cur - ref - ref @ affine.to(device=cur.device, dtype=cur.dtype)
        disp = moved.norm(dim=1)
        reach = disp + dt * (
            vel.norm(dim=1) + (vel - self._prev_velocities).norm(dim=1)
        )
        look_ahead = 0.5 * halo_radius * self.cell_change_bound_per_step
        word = (reach.max() + look_ahead >= budget).to(torch.int32) * DRIFT_TRIP
        word += (disp.max() > budget).to(torch.int32) * DRIFT_BREACH
        flag.copy_(word)
        self._prev_velocities.copy_(vel)
