import os
import torch
import torch.distributed as dist

from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
    SimulationConfig,
    SFCPartitioner,
    SFCCurveType,
    SimState,
    DistNNPState,
)

# Side length (Å) of the synthetic simulation box.
# Positions in generate_synthetic_dataset are drawn from [0, BOX_SIZE).
BOX_SIZE = 20.0


def generate_synthetic_dataset(num_atoms: int) -> MolecularInfo:
    """
    Generates a random system of num_atoms size on all ranks. The system is the same on all ranks.

    Args:
        num_atom (int): number of atoms in the system
    Returns:
        (MolecularInfo)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed seed so every rank produces the identical positions/velocities.
    gen = torch.Generator()
    gen.manual_seed(42)

    mol_info = MolecularInfo()
    mol_info.num_atoms = num_atoms
    mol_info.atom_indices = torch.arange(num_atoms, dtype=torch.long, device=device)
    mol_info.atom_positions = (torch.rand(num_atoms, 3, generator=gen) * BOX_SIZE).to(
        device
    )
    mol_info.atom_velocities = (torch.randn(num_atoms, 3, generator=gen) * 0.1).to(
        device
    )

    return mol_info


def create_dist_nnp_state(num_atoms: int) -> SimState:
    """
    This helper function creates a starting DistNNPState to use with the Simulator.
    Given the number of atoms, it:

    (0) Initializes a distributed system with Torch and make a DistEnvironment
    (1) Creates a synthetic system on all ranks to create a MolecularInfo
    (2) Uses a partitioner to generate a partitioning and make a PartitionMessagePassingInfo
    (3) Use the aforementioned DistEnvironment, MolecularInfo, and PartitionMessagePassingInfo to create a DistNNPState

    Args:
        num_atoms (int): number of atoms in the system
    Returns:
        (DistNNPState)

    """

    # (0) Initialize distributed if not already done.
    # Single-GPU / single-process runs skip init_process_group entirely —
    # torchrun sets RANK; SLURM sets SLURM_PROCID. If neither is present we
    # are running standalone and treat this as a world_size=1 job.
    is_distributed_launch = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    if is_distributed_launch and not dist.is_initialized():
        # backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend="cuda:nccl,cpu:gloo")
        # dist.init_process_group(backend=backend)

    if dist.is_initialized():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    num_gpus = torch.cuda.device_count()
    gpus_per_node = num_gpus if num_gpus > 0 else 1
    local_rank = global_rank % gpus_per_node
    num_nodes = max(1, (world_size + gpus_per_node - 1) // gpus_per_node)

    if num_gpus > 0:
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if num_gpus > 0 else "cpu")

    dist_env = DistEnvironment()
    dist_env.global_rank = global_rank
    dist_env.world_size = world_size
    dist_env.gpus_per_node = gpus_per_node
    dist_env.num_nodes = num_nodes
    dist_env.local_rank = local_rank

    # (1) Create the same synthetic MolecularInfo on every rank.
    mol_info = generate_synthetic_dataset(num_atoms)

    # (2) Partition atoms and build halo communication maps.
    sim_config = SimulationConfig()
    sim_config.cutoff = 6.0
    sim_config.ghost_cutoff_buffer = 0.5
    sim_config.cell = torch.eye(3, device=device) * BOX_SIZE
    sim_config.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    partitioner = SFCPartitioner(SFCCurveType.Morton)
    partition_info = partitioner.partition(mol_info, dist_env)
    partitioner.build_halo_maps(partition_info, mol_info, dist_env, sim_config)

    # (3) Assemble DistNNPState.
    nnp_state = DistNNPState()
    nnp_state.dist_env = dist_env
    nnp_state.sim_config = sim_config
    nnp_state.global_molecular_info = mol_info
    nnp_state.local_partition_info = partition_info
    nnp_state = SimState(
        nnp_state
    )  # Wrap in SimState for use with Simulator and potential models.

    return nnp_state


def sync_and_repartition(state: SimState) -> SimState:
    """
    Gathers updated atom positions and velocities from all ranks, rebuilds the
    global MolecularInfo, re-partitions, and returns a new DistNNPState with
    atoms redistributed to their new owning ranks.

    For world_size=1 (single GPU), skips all communication and re-partitions
    in place.

    Args:
        state (SimState): current simulation state after one or more steps
    Returns:
        (SimState): new state with updated partitioning and halo maps
    """
    dist_env = state.dist_env
    world_size = dist_env.world_size
    device = state.local_partition_info.local_atom_positions.device

    pi = state.local_partition_info
    local_indices = pi.partition_indices  # [num_local] — global atom IDs
    local_positions = pi.local_atom_positions  # [num_local, 3]
    local_velocities = pi.local_atom_velocities  # [num_local, 3]

    global_mol_info = state.global_molecular_info

    if world_size == 1:
        # No communication needed; update global info in place.
        global_mol_info.atom_positions[local_indices] = local_positions
        global_mol_info.atom_velocities[local_indices] = local_velocities
    else:
        # Collect (indices, positions, velocities) from every rank.
        local_data = {
            "indices": local_indices.cpu(),
            "positions": local_positions.cpu(),
            "velocities": local_velocities.cpu(),
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_data)

        assert all(
            [g is not None for g in gathered]
        ), "Gather failed: some ranks did not contribute data."

        all_indices = torch.cat([g["indices"] for g in gathered])  # type: ignore # [N]
        all_positions = torch.cat([g["positions"] for g in gathered])  # type: ignore  # [N, 3]
        all_velocities = torch.cat([g["velocities"] for g in gathered])  # type: ignore  # [N, 3]

        # Restore canonical atom order (0 … N-1).
        sort_order = torch.argsort(all_indices)
        global_mol_info.atom_positions = all_positions[sort_order].to(device)
        global_mol_info.atom_velocities = all_velocities[sort_order].to(device)

    # Re-partition on the updated positions and rebuild halo maps.
    partitioner = SFCPartitioner(SFCCurveType.Morton)
    partition_info = partitioner.partition(global_mol_info, dist_env)
    partitioner.build_halo_maps(
        partition_info, global_mol_info, dist_env, state.sim_config
    )

    new_state = DistNNPState()
    new_state.dist_env = dist_env
    new_state.sim_config = state.sim_config
    new_state.global_molecular_info = global_mol_info
    new_state.local_partition_info = partition_info

    new_state = SimState(
        new_state
    )  # Wrap in SimState for use with Simulator and potential models.

    return new_state
