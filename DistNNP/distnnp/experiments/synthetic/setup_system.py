import os
import torch
import torch.distributed as dist

from distnnp_state import (
    MolecularInfo,
    SimulationConfig,
    SFCPartitioner,
    SFCCurveType,
    DistNNPState,
    TorchDistEnv,
)

from distnnp_state.masses import TORCH_ATOMIC_MASSES
from distnnp import SimState

import json

# Side length (Å) of the synthetic simulation box.
# Positions in generate_synthetic_dataset are drawn from [0, BOX_SIZE).
BOX_SIZE = 20.0


def generate_synthetic_dataset(
    num_atoms: int, device, box_size: float = BOX_SIZE
) -> MolecularInfo:
    """
    Generates a random system of num_atoms size on all ranks. The system is the same on all ranks.

    Args:
        num_atom (int): number of atoms in the system
    Returns:
        (MolecularInfo)

    """
    # Fixed seed so every rank produces the identical positions/velocities.
    gen = torch.Generator()
    gen.manual_seed(42)

    mol_info = MolecularInfo()
    mol_info.num_atoms = num_atoms
    mol_info.atom_vertex_numbers = torch.arange(
        num_atoms, dtype=torch.long, device=device
    )
    mol_info.atom_positions = (torch.rand(num_atoms, 3, generator=gen) * box_size).to(
        device
    )
    mol_info.atom_velocities = (torch.randn(num_atoms, 3, generator=gen) * 0.1).to(
        device
    )
    atom_masses_on_device = TORCH_ATOMIC_MASSES.to(device)
    mol_info.atom_types = torch.randint(
        1, 100, (num_atoms,), dtype=torch.long, device=device
    )
    mol_info.atom_masses = atom_masses_on_device[mol_info.atom_types].to(device)
    mol_info.atom_forces = torch.zeros_like(mol_info.atom_positions)

    return mol_info


def create_dist_sim_state(num_atoms: int) -> SimState:
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
        (SimState)

    """

    # (0) Initialize distributed if not already done.
    # Single-GPU / single-process runs skip init_process_group entirely —
    # torchrun sets RANK; SLURM sets SLURM_PROCID. If neither is present we
    # are running standalone and treat this as a world_size=1 job.
    torch_dist_env = TorchDistEnv()
    dist_env = torch_dist_env.dist_env

    # (1) Create the same synthetic MolecularInfo on every rank.
    mol_info = generate_synthetic_dataset(num_atoms, device=torch_dist_env.device)

    # (2) Partition atoms and build halo communication maps.
    sim_config = SimulationConfig()
    sim_config.cutoff = 6.0
    sim_config.ghost_cutoff_buffer = 0.5
    sim_config.cell = torch.eye(3) * BOX_SIZE
    sim_config.pbc = torch.tensor([True, True, True], dtype=torch.bool)

    nnp_state = SimState(
        molecular_system=mol_info,
        partitioner=SFCPartitioner(SFCCurveType.Morton),
        simulation_config=sim_config,
        distributed_environment=dist_env,
        device=torch_dist_env.device,
    )

    return nnp_state


class Timer:
    # Class-level dictionary to aggregate timings globally
    timing_keys = {}

    def __init__(self, key, metadata=None):
        self.key = key
        self.start_event = None
        self.end_event = None

        # Initialize the global registry for this key if it's the first time
        if self.key not in Timer.timing_keys:
            Timer.timing_keys[self.key] = {
                "total_time_ms": 0.0,
                "count": 0,
                "elapsed_times_ms": [],
            }
        if metadata is not None:
            Timer.timing_keys[self.key] = Timer.timing_keys[self.key] | metadata

    def start(self):
        """Manually starts the CUDA timer."""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self):
        """Manually stops the CUDA timer, syncs, and records the time."""
        if self.start_event is None:
            raise RuntimeError(
                "Timer was stopped before being started. Call start() first."
            )

        self.end_event.record()
        torch.cuda.synchronize()

        # Calculate time in milliseconds
        elapsed = self.start_event.elapsed_time(self.end_event)

        # Update the global registry
        Timer.timing_keys[self.key]["elapsed_times_ms"].append(elapsed)
        Timer.timing_keys[self.key]["total_time_ms"] += elapsed
        Timer.timing_keys[self.key]["count"] += 1

        # Clear events to prevent accidental reuse without starting again
        self.start_event = None
        self.end_event = None

        return elapsed

    def __enter__(self):
        """Hook for the 'with' statement. Automatically calls start()."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Hook for the 'with' statement. Automatically calls stop()."""
        self.stop()

    # --- Local Object Helpers ---

    def __getitem__(self, key):
        """Allows you to access the list of elapsed times for this specific key from the local object."""
        if key != Timer.timing_keys:
            raise KeyError(f"Key '{key}' not found in Timer registry.")
        return Timer.timing_keys[self.key]["elapsed_times_ms"]

    @property
    def local_total_time(self):
        """Allows you to check the total time for this specific key from the local object."""
        return Timer.timing_keys[self.key]["total_time_ms"]

    @property
    def local_count(self):
        """Allows you to check the run count for this specific key from the local object."""
        return Timer.timing_keys[self.key]["count"]

    # --- Global Class Methods ---

    @classmethod
    def json(cls):
        """Returns the dictionary as a JSON string."""
        return json.dumps(cls.timing_keys, indent=4)

    @classmethod
    def clear(cls):
        """Resets all recorded timings globally."""
        cls.timing_keys.clear()

    @classmethod
    def add_meta_data(cls, meta_dict):
        """Allows adding arbitrary metadata to the timing dictionary."""
        cls.timing_keys["meta_data"] = meta_dict
