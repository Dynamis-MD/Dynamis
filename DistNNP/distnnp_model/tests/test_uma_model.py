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
    KDTreePartitioner,
    radius_graph_pbc_cuda
)

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets.atomic_data import AtomicData
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

    if rank == 0:
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
    dist_model = UMAModel("uma-s-1p2", dataset="omol", device=device)

    # ==========================================
    # 2. Setup Global Data (Ground Truth)
    # ==========================================
    total_atoms = 200
    cell_length = 20
    torch.manual_seed(42)
    # Give enough spatial spread to ensure cutoffs divide the space reasonably
    global_positions = torch.rand((total_atoms, 3), device=device) * cell_length
    global_atomic_numbers = torch.randint(1, 10, (total_atoms,), device=device, dtype=torch.long)
    print(f"global_positions: {global_positions}, global_atomic_numbers: {global_atomic_numbers}")
    global_cell = torch.eye(3, device=device) * cell_length
    global_pbc = torch.tensor([True, True, True], device=device)
    cutoff = dist_model.main_model.backbone.cutoff

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

    sim_config = SimulationConfig()
    sim_config.cell = global_cell
    sim_config.pbc = global_pbc
    sim_config.cutoff = cutoff
    sim_config.ghost_cutoff_buffer = 0.5
    
    partitioner = KDTreePartitioner()
    global_mol.atom_positions = global_mol.atom_positions
    partition_info = partitioner.partition(global_mol, env)
    partitioner.build_halo_maps(partition_info, global_mol, env, sim_config)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(
            partition_info.global_send_recv_matrix, 
            op=dist.ReduceOp.SUM
        )
    global_mol.atom_positions = global_mol.atom_positions

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
    local_mol.num_atoms = n_local
    
    state = DistNNPState()
    state.dist_env = env
    state.local_molecular_info = local_mol
    state.global_molecular_info = global_mol
    state.local_partition_info = partition_info
    state.sim_config = sim_config

    dist_model.initialize_buffers(partition_info, device)

    # ==========================================
    # 4. Forward & Backward Pass: STANDARD MODEL
    # ==========================================
    if rank == 0:
        cell = sim_config.cell.unsqueeze(0)
        pbc = sim_config.pbc.unsqueeze(0)
        # Build strict Graph for Fairchem if needed
        with torch.no_grad():
            global_edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                type("Data", (object,), {
                    "pos": global_positions,
                    "cell": cell,
                    "natoms": torch.tensor([total_atoms], device=device),
                    "pbc": pbc,
                })(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )
        data = AtomicData(
            global_positions,
            global_atomic_numbers,
            cell,
            pbc,
            torch.tensor([total_atoms], dtype=torch.long, device=device),
            global_edge_index,
            -cell_offsets,
            torch.tensor([global_edge_index.shape[1]], dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device), # charge
            torch.ones(1, dtype=torch.long, device=device), # spin
            torch.zeros(total_atoms, dtype=torch.long, device=device), # fixed
            torch.zeros(total_atoms, dtype=torch.long, device=device), # tags
            dataset="omol",
        )
        data.to(device)
        # Fairchem Backbone Output
        predictor.move_to_device()
        pred = predictor.predict(data)
        from distnnp_model.nnp.uma import _EV_PER_A_TO_AMU_A_PER_FS2
        global_energy, global_forces = pred["energy"] * _EV_PER_A_TO_AMU_A_PER_FS2, pred["forces"] * _EV_PER_A_TO_AMU_A_PER_FS2

    # ==========================================
    # 5. Forward & Backward Pass: DISTRIBUTED MODEL
    # ==========================================
    local_energy, local_forces = dist_model(state)
    
    # ==========================================
    # 6. Parity Assertions
    # ==========================================
    
    # A. Check Energy Parity (Sum of local energies == global energy)
    dist.all_reduce(local_energy, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"Energy from distributed model: {local_energy.item()}, global energy: {global_energy}")
        torch.testing.assert_close(
            local_energy.item(), global_energy.item(),
            rtol=1e-6, atol=1e-6, # Use generous tolerance for reduction ordering differences
            msg="Distributed total energy does not match Standard model energy!"
        )
        print("Energy Parity Passed!")

    # B. Check Forces Parity (Local forces must exactly match their slice of global forces)
    n_local = state.local_molecular_info.num_atoms
    local_indices = state.local_partition_info.partition_indices[:n_local].long().to(device)

    local_forces = local_forces.to(device)
    gathered_forces = [torch.zeros((total_atoms, 3), device=device) for _ in range(world_size)] if rank == 0 else None
    rank_force = torch.zeros((total_atoms, 3), device=device)
    rank_force.scatter_(0, local_indices.unsqueeze(-1).expand(-1, 3), local_forces[:n_local])
    if world_size > 1:
        dist.gather(rank_force, gather_list=gathered_forces, dst=0)
    else:
        gathered_forces = [rank_force]
        
    if rank == 0:
        reconstructed_global_forces = torch.sum(torch.stack(gathered_forces), dim=0)
        print(f"Forces from distributed model: {reconstructed_global_forces}, global forces: {global_forces}")

        max_diff = torch.max(torch.abs(reconstructed_global_forces - global_forces))
        print(f"Maximum Force Difference: {max_diff.item():.6f}")

        torch.testing.assert_close(
            reconstructed_global_forces, global_forces,
            rtol=1e-6, atol=5e-6,
            msg="Distributed forces do not match Standard model forces!"
        )
        print("Forces Parity Passed!")

    dist.destroy_process_group()

if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        print("This file should be run via torchrun to test distributed functionality.")
    test_uma_distributed_parity()

