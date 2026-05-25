import os
import torch
import torch.distributed as dist
import pytest

from distnnp_model.nnp import UMAModel
from distnnp_state import (
    DistNNPState, 
    DistEnvironment, 
    MolecularInfo, 
    SimulationConfig,
    SimpleSpatialPartitioner,
    radius_graph_pbc_cuda
)

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from ase import Atoms
    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False

def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    nnodes = int(os.environ.get("SLURM_NNODES", 1))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    return rank, local_rank, world_size, nnodes

@pytest.mark.skipif(not HAS_FAIRCHEM, reason="fairchem-core is not installed")
def test_uma_distributed_parity():
    rank, local_rank, world_size, nnodes = setup_distributed()
    if world_size < 2:
        pytest.skip("Parity test requires at least 2 GPUs. Run with torchrun or mpirun.")
    
    # device = f"cuda:{local_rank}"
    device = "cuda"
    
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
    standard_calc = FAIRChemCalculator(predictor, task_name="omol")
    dist_model = UMAModel("uma-s-1p2", device=device)

    # ==========================================
    # 2. Setup Global Data (Ground Truth)
    # ==========================================
    total_atoms = 16

    # Use a fixed seed on all ranks so they generate the same global structure
    torch.manual_seed(42)
    # Give enough spatial spread to ensure cutoffs divide the space reasonably
    global_positions = torch.rand((total_atoms, 3), device=device) * 10.0
    global_atomic_numbers = torch.randint(1, 10, (total_atoms,), device=device, dtype=torch.long)
    global_cell = torch.eye(3, device=device) * 10.0
    global_pbc = torch.tensor([True, True, True], device=device)
    cutoff = 5.0

    ase_system = Atoms(
        numbers=global_atomic_numbers.cpu().clone().detach(),
        positions=global_positions.cpu().clone().detach(),
        cell=global_cell.cpu().clone().detach(),
        pbc=global_pbc.cpu().clone().detach()
    )

    # ==========================================
    # 3. Setup Distributed State (Partitioning)
    # ==========================================
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.local_rank = local_rank
    env.num_nodes = nnodes
    env.gpus_per_node = world_size // nnodes
    
    global_mol = MolecularInfo()
    global_mol.num_atoms = total_atoms
    global_mol.atom_vertex_numbers = torch.arange(total_atoms, device=device, dtype=torch.long)
    global_mol.atom_types = global_atomic_numbers
    global_mol.atom_positions = global_positions
    global_mol.atom_velocities = torch.zeros((total_atoms, 3), device=device)
    global_mol.atom_forces = torch.zeros((total_atoms, 3), device=device)

    sim_config = SimulationConfig()
    sim_config.cell = global_cell
    sim_config.pbc = global_pbc
    sim_config.cutoff = cutoff
    sim_config.ghost_cutoff_buffer = 0.5
    
    partitioner = SimpleSpatialPartitioner()
    partition_info = partitioner.partition(global_mol, env)
    partitioner.build_halo_maps(partition_info, global_mol, env, sim_config)

    local_mol = MolecularInfo()
    local_mol.atom_positions = global_mol.atom_positions[
        partition_info.partition_indices
    ]
    local_mol.atom_velocities = global_mol.atom_velocities[
        partition_info.partition_indices
    ]
    local_mol.atom_types = global_mol.atom_types[
        partition_info.partition_indices
    ]
    n_local = local_mol.atom_positions.shape[0]
    local_mol.atom_forces = global_mol.atom_forces[
        partition_info.partition_indices
    ]
    local_mol.num_atoms = n_local
    
    state = DistNNPState()
    state.dist_env = env
    state.local_molecular_info = local_mol
    state.global_molecular_info = global_mol
    state.local_partition_info = partition_info
    state.sim_config = sim_config

    # ==========================================
    # 4. Forward & Backward Pass: STANDARD MODEL
    # ==========================================
    if rank == 0:
        global_pos_std = global_positions.clone().detach().requires_grad_(True)
        
        data = type("Data", (object,), {
            "pos": global_pos_std,
            "atomic_numbers": global_atomic_numbers,
            "cell": sim_config.cell.unsqueeze(0),
            "natoms": torch.tensor([total_atoms], device=device),
            "pbc": sim_config.pbc.unsqueeze(0),
            "batch": torch.zeros(total_atoms, dtype=torch.long, device=device),
            "tags": torch.zeros(total_atoms, dtype=torch.long, device=device),
            # "edge_index": global_edge_index,
        })()

        # Build strict Graph for Fairchem if needed
        with torch.no_grad():
            global_edge_index, _, _ = radius_graph_pbc_cuda(
                data,
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )

        data.edge_index = global_edge_index

        # Fairchem Backbone Output
        global_energy = standard_calc.get_potential_energy(ase_system)
        global_forces = standard_calc.get_forces(ase_system)

    # ==========================================
    # 5. Forward & Backward Pass: DISTRIBUTED MODEL
    # ==========================================
    batch = torch.zeros(n_local, dtype=torch.long, device=device)

    n_total_with_ghosts = (
        state.local_molecular_info.num_atoms +
        state.local_partition_info.recv_ghost_indices.shape[0]
    )
    batch_full = torch.zeros(n_total_with_ghosts, dtype=torch.long, device=device)

    local_energy, local_forces = dist_model(state, batch, batch_full, "omol")
    
    # ==========================================
    # 6. Parity Assertions
    # ==========================================
    
    # A. Check Energy Parity (Sum of local energies == global energy)
    local_energy_sum = local_energy.clone()
    dist.all_reduce(local_energy_sum, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        torch.testing.assert_close(
            local_energy_sum.item(), global_energy, 
            rtol=1e-3, atol=1e-3, # Use generous tolerance for reduction ordering differences
            msg="Distributed total energy does not match Standard model energy!"
        )
        print("Energy Parity Passed!")

    # B. Check Forces Parity (Local forces must exactly match their slice of global forces)
    # The dist_model only computes grad for n_local atoms
    n_local = state.local_molecular_info.num_atoms
    
    # Each rank gathers back correctly to global
    local_indices = state.local_partition_info.partition_indices[:n_local].long()
    
    gathered_forces = [torch.zeros((total_atoms, 3), device=device) for _ in range(world_size)] if rank == 0 else None
    
    rank_force = torch.zeros((total_atoms, 3), device=device)
    rank_force.scatter_(0, local_indices.unsqueeze(-1).expand(-1, 3), local_forces[:n_local])

    if world_size > 1:
        dist.gather(rank_force, gather_list=gathered_forces, dst=0)
    else:
        gathered_forces = [rank_force]
        
    if rank == 0:
        reconstructed_global_forces = torch.sum(torch.stack(gathered_forces), dim=0)
        torch.testing.assert_close(
            reconstructed_global_forces, global_forces,
            rtol=1e-3, atol=1e-3,
            msg="Distributed forces do not match Standard model forces!"
        )
        print("Forces Parity Passed!")

    dist.destroy_process_group()

if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        print("This file should be run via torchrun to test distributed functionality.")
    test_uma_distributed_parity()

