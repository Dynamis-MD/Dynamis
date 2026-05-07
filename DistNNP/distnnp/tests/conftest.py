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
from distnnp_model.distnnp_model_impl import FeatureExchanger
from distnnp.simulation import SimState, Simulation
from distnnp_integrator.langevin import Langevin
from distnnp_integrator.verlet import Verlet
from distnnp_integrator.euler import Euler

from distnnp_state import (
    MolecularInfo,
    SimulationConfig,
    DistEnvironment,
    SimpleSpatialPartitioner,
    DistNNPState,
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
    Create a MolecularInfo with random positions in a 10x10x10 box.
    We make this with a fixed seed but set that fixed seed as a parameter.
    """
    torch.manual_seed(seed)
    state = MolecularInfo()
    state.num_atoms = num_atoms
    
    # Generate on CPU to ensure all ranks start with the SAME universe
    state.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long)
    state.atom_positions = torch.rand(num_atoms, 3) * 8.0
    state.atom_velocities = torch.randn(num_atoms, 3) * 0.1
    
    # Now move the consistent data to the GPU
    state.atom_vertex_numbers = state.atom_vertex_numbers.to(device)
    state.atom_positions = state.atom_positions.to(device)
    state.atom_velocities = state.atom_velocities.to(device)
    return state

def _make_sim_config(cutoff: float, device: torch.device) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = 0.5
    cfg.cell = torch.eye(3, device=device) * 8.0
    cfg.pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    return cfg

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
# integrator. Returns a callable make() -> Simulation so each test
# controls when construction happens.
#
# Construction steps inside make():
#   1. Build MolecularInfo  (small synthetic system, fixed seed)
#   2. Build SimulationConfig
#   3. Build DistEnvironment from dist.get_rank() / get_world_size()
#   4. Partition with SimpleSpatialPartitioner + build_halo_maps
#   5. Assemble DistNNPState
#   6. Create LennardJonesPotentialModel (stand-in for UMAModel)
#   7. Set up FeatureExchanger IPC on model._feature_exchanger
#      (see distnnp_model/tests/test_ipc_exchange.py for the protocol)
#   8. Build SimState(dist_state, sim_config, forces, mass)
#   9. Return Simulation(sim_state, model, integrator)
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

        # Step 1: Build MolecularInfo (16-atom system, fixed seed, same on all ranks)
        molecular_info = _make_global_state(16, 100, device)

        # Step 2: Build Simulation Config
        # Note; I'm not sure if 3.5 is a proper value here
        sim_config = _make_sim_config(3.5, device)

        # Step 3: Build DistEnviornment
        env = _make_env(rank, world_size, local_rank)

        # Step 4: Partition with SimpleSpatialPartitioner + build_halo_maps
        partitioner = SimpleSpatialPartitioner()
        partition_info = partitioner.partition(molecular_info, env)
        partitioner.build_halo_maps(partition_info, molecular_info, env, sim_config)

        # Step 5: Assemble DistNNPState
        distnnp_state = DistNNPState()
        distnnp_state.dist_env = env
        distnnp_state.global_molecular_info = molecular_info
        distnnp_state.local_partition_info = partition_info

        # Step 6: Create LennardJonesPotentialModel (stand-in for UMAModel)
        # These 1.0 values are fairly arbitrary, not sure if they should be changed
        epsilon = torch.tensor([1.0], dtype=torch.float32, device=device)
        sigma = torch.tensor([1.0], dtype=torch.float32, device=device)
        lj_model = LennardJonesPotentialModel(epsilon, sigma)

        # Step 7: Build SimState
        F = 3
        n_local = partition_info.local_atom_positions.shape[0]
        forces = torch.zeros(n_local, F, dtype=torch.float32, device=device)
        mass = torch.full((n_local,), 1.0, dtype=torch.float32, device=device)
        sim_state = SimState(distnnp_state, sim_config, forces, mass)

        # Step 8: Return Simulation
        # The constants below are arbitrary, not sure if they should be changed
        if integrator_name == "euler":
            integrator = Euler(dt=1e-3)
        elif integrator_name == "verlet":
            integrator = Verlet(dt=1e-3)
        else: # integrator_name == "langevin"
            integrator = Langevin(dt=1e-3, gamma=1.0, kT=0.6)

        # Store exchangers for teardown
        created.append(lj_model.position_exchanger.exchanger)
        created.append(lj_model.atom_type_exchanger.exchanger)
        created.append(lj_model.global_index_exchanger.exchanger)

        return Simulation(sim_state, lj_model, integrator)

    yield make

    # Teardown everything in created
    for exchanger in created:
        exchanger.teardown()
        
    created.clear()
