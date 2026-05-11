"""
conftest.py — shared fixtures for distributed simulation tests.

Run via:
    torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v
"""

import os
import pytest
import torch
import torch.distributed as dist
from distnnp_model.empirical.lj_potential_model import LennardJonesPotentialModel
from distnnp.simulator import Simulator, SimulatorConfig
from distnnp_integrator.langevin import Langevin
from distnnp_integrator.verlet import Verlet
from distnnp_integrator.euler import Euler

from distnnp_state import (
    MolecularInfo,
    DistEnvironment,
)

# ---------------------------------------------------------------------------
# Session-scoped autouse fixture: initialize torch.distributed.
#
# pytestmark with dist.is_initialized() at module import time does not work
# because init_process_group hasn't been called yet. Instead we use an
# autouse session fixture that either initializes NCCL (when running under
# torchrun) or skips the entire session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def dist_init():
    """Initialize NCCL process group from torchrun environment variables."""
    if os.environ.get("LOCAL_RANK") is None:
        pytest.skip("Not running under torchrun — distributed tests require torchrun")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    if dist.get_world_size() < 2:
        dist.destroy_process_group()
        pytest.skip("Requires world_size >= 2")

    yield

    dist.barrier()
    dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _make_global_state(num_atoms: int, seed: int, device: torch.device) -> MolecularInfo:
    """
    Create a MolecularInfo with random positions in an 8x8x8 box.
    We make this with a fixed seed but set that fixed seed as a parameter.
    """
    torch.manual_seed(seed)
    state = MolecularInfo()
    state.num_atoms = num_atoms
    
    # Generate on CPU to ensure all ranks start with the SAME universe
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long)
    state.atom_positions = torch.rand(num_atoms, 3) * 8.0
    state.atom_velocities = torch.randn(num_atoms, 3) * 0.1
    state.atom_masses = torch.ones(num_atoms, dtype=torch.float32)
    state.atom_types = torch.zeros(num_atoms, dtype=torch.long)

    # Now move the consistent data to the GPU
    state.atom_vertex_numbers = state.atom_vertex_numbers.to(device)
    state.atom_positions = state.atom_positions.to(device)
    state.atom_velocities = state.atom_velocities.to(device)
    state.atom_masses = state.atom_masses.to(device)
    state.atom_types = state.atom_types.to(device)
    return state

def _make_env(rank: int, world_size: int, local_rank: int) -> DistEnvironment:
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.gpus_per_node = gpus_per_node
    env.num_nodes = world_size // gpus_per_node
    env.local_rank = local_rank
    return env

# ---------------------------------------------------------------------------
# simulation_factory fixture
#
# Parametrize over all available integrators so every test runs once per
# integrator. Returns a callable make() -> Simulator so each test
# controls when construction happens.
# ---------------------------------------------------------------------------
@pytest.fixture(params=["euler", "verlet", "langevin"])
def simulation_factory(request):
    integrator_name = request.param
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # For teardown
    created = []

    def make():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        molecular_info = _make_global_state(16, 100, device)
        env = _make_env(rank, world_size, local_rank)

        epsilon = torch.tensor([1.0], dtype=torch.float32, device=device)
        sigma = torch.tensor([1.0], dtype=torch.float32, device=device)
        lj_model = LennardJonesPotentialModel(epsilon, sigma)

        if integrator_name == "euler":
            integrator = Euler(dt=1e-3)
        elif integrator_name == "verlet":
            integrator = Verlet(dt=1e-3)
        else:  # langevin
            integrator = Langevin(dt=1e-3, gamma=1.0, kT=0.6)

        # Constants are fairly arbitrary
        # Not sure if the partition algo should be adjusted
        config = SimulatorConfig(
            cutoff_angstrom=3.5,
            ghost_buffer_angstrom=0.5,
            side_length_angstrom=8.0,
            partition_algorithm="simple",
        )

        simulator = Simulator(
            model=lj_model,
            integrator=integrator,
            dist_environment=env,
            config=config,
            device=device,
        )
        simulator.initialize_simulator_state(molecular_info, device=device)

        # Store exchangers for teardown
        created.append(lj_model.position_exchanger.exchanger)
        created.append(lj_model.atom_type_exchanger.exchanger)
        created.append(lj_model.global_index_exchanger.exchanger)

        return simulator

    yield make

    # Teardown everything in created
    for exchanger in created:
        exchanger.teardown()
        
    created.clear()
