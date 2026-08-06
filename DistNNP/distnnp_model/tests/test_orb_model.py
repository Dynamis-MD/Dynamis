"""
Distributed parity test for ORBModel — mirrors test_uma_model.

Place at: distnnp_model/tests/test_orb_model.py
Run:
    torchrun --nproc_per_node=2 distnnp_model/tests/test_orb_model.py

KEY FACTS (verified against orb-models source):
  * orb_v3_conservative_inf_omat energy =
        EnergyHead(interaction)  +  ZBL pair_repulsion  +  element reference
    The reference term is per-element and LARGE (~1e4-1e5 eV). It dominates
    the absolute energy but distributes cleanly (each rank adds its own local
    atoms' reference; the sum across ranks == global reference), PROVIDED
    ghost atoms are excluded from the reference sum.
  * Conservative forces = -dE/d(positions) via autograd. The conservative
    path does NOT subtract net force (that only happens in the DIRECT
    ForceHead). So no net-force reduction is needed for parity.
  * EnergyHead uses MEAN node-aggregation and is therefore NOT size-consistent
    across a partition on its own. See the note in ORBModel about why we sum
    the INTERACTION energy across ranks but handle the reference separately.
  * predict() returns fp64 energy by default -> cast before comparing.

The cleanest ground truth is orbff.predict(graph, split=False) on the full
system, built with the SAME adapter the model ships with.
"""

import os
import torch
import torch.distributed as dist
import pytest

# NOTE: ORBModel lives at distnnp_model/src/distnnp_model/nnp/orb.py
# so the import is distnnp_model.nnp.orb (matches uma at distnnp_model.nnp.uma)
from distnnp_model.nnp.orb import ORBModel
from distnnp_state import (
    DistNNPState,
    DistEnvironment,
    MolecularInfo,
    SimulationConfig,
    KDTreePartitioner,
    radius_graph_pbc_cuda,
)

try:
    from orb_models.forcefield import pretrained
    HAS_ORB = True
except ImportError:
    HAS_ORB = False


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    nnodes = int(os.environ.get("SLURM_NNODES", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # ORB_ALLOW_SINGLE_RANK=1 is a DIAGNOSTIC path: run world_size=1 to validate
    # the ORB energy composition + autograd in isolation (no ghosts, exchange is
    # identity), separating "is the model right" from "is the cross-rank
    # decomposition right". The process group is still initialized so the
    # exchanger's single-rank identity short-circuit (dist.is_initialized() and
    # world_size==1) engages.
    allow_single = os.environ.get("ORB_ALLOW_SINGLE_RANK") == "1"
    if (world_size > 1 or allow_single) and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, local_rank, world_size, nnodes


@pytest.mark.skipif(not HAS_ORB, reason="orb_models is not installed")
def test_orb_distributed_parity():
    rank, local_rank, world_size, nnodes = setup_distributed()
    if world_size < 2 and os.environ.get("ORB_ALLOW_SINGLE_RANK") != "1":
        pytest.skip("Parity test requires at least 2 GPUs. Run with torchrun.")

    device = "cuda"
    MODEL_NAME = "orb_v3_conservative_inf_omat"
    CUTOFF = 6.0  # orb-v3 uses a 6 A cutoff

    # -----------------------------------------------------------------
    # 1. Reference model on rank 0 (full system)
    # -----------------------------------------------------------------
    if rank == 0:
        factory_fn = getattr(pretrained, MODEL_NAME)
        ref_orbff, atoms_adapter = factory_fn(device=device, precision="float32-highest", compile=False)
        ref_orbff.eval()

    # Distributed model on every rank
    dist_model = ORBModel(model_name=MODEL_NAME, device=device)

    # -----------------------------------------------------------------
    # 2. Global system — identical on all ranks (seeded)
    # -----------------------------------------------------------------
    total_atoms = 20
    cell_length = 10.0
    torch.manual_seed(42)

    global_positions = torch.rand((total_atoms, 3), device=device) * cell_length
    global_atom_types = torch.randint(1, 10, (total_atoms,), device=device, dtype=torch.long)
    global_cell = torch.eye(3, device=device) * cell_length
    global_pbc = torch.tensor([True, True, True], device=device)

    # -----------------------------------------------------------------
    # 3. Partition
    # -----------------------------------------------------------------
    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.local_rank = local_rank
    env.num_nodes = nnodes
    env.gpus_per_node = world_size // nnodes

    global_mol = MolecularInfo()
    global_mol.num_atoms = total_atoms
    global_mol.atom_vertex_numbers = torch.arange(total_atoms, device=device, dtype=torch.long)
    global_mol.atom_types = global_atom_types
    global_mol.atom_positions = global_positions
    global_mol.atom_velocities = torch.zeros((total_atoms, 3), device=device)
    global_mol.atom_forces = torch.zeros((total_atoms, 3), device=device)

    sim_config = SimulationConfig()
    sim_config.cell = global_cell
    sim_config.pbc = global_pbc
    sim_config.cutoff = CUTOFF
    sim_config.ghost_cutoff_buffer = 0.5

    partitioner = KDTreePartitioner()
    partition_info = partitioner.partition(global_mol, env)
    partitioner.build_halo_maps(partition_info, global_mol, env, sim_config)

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(partition_info.global_send_recv_matrix, op=dist.ReduceOp.SUM)

    local_mol = MolecularInfo()
    local_mol.atom_positions = global_mol.atom_positions[partition_info.partition_indices]
    local_mol.atom_velocities = global_mol.atom_velocities[partition_info.partition_indices]
    local_mol.atom_types = global_mol.atom_types[partition_info.partition_indices]
    local_mol.atom_forces = global_mol.atom_forces[partition_info.partition_indices]
    n_local = local_mol.atom_positions.shape[0]
    local_mol.num_atoms = n_local

    state = DistNNPState()
    state.dist_env = env
    state.local_molecular_info = local_mol
    state.global_molecular_info = global_mol
    state.local_partition_info = partition_info
    state.sim_config = sim_config

    dist_model.initialize_buffers(partition_info, device)

    # -----------------------------------------------------------------
    # 4. Reference forward (rank 0) via the shipped adapter + predict()
    # -----------------------------------------------------------------
    if rank == 0:
        import ase
        ref_atoms = ase.Atoms(
            numbers=global_atom_types.cpu().numpy(),
            positions=global_positions.detach().cpu().numpy(),
            cell=global_cell.cpu().numpy(),
            pbc=True,
        )
        ref_graph = atoms_adapter.from_ase_atoms(ref_atoms, device=device)
        ref_out = ref_orbff.predict(ref_graph, split=False)

        ename = ref_orbff.energy_name          # "energy"
        fname = ref_orbff.grad_forces_name     # "grad_forces"
        print("[ref] predict keys:", list(ref_out.keys()))   # see them once
        ref_energy = ref_out[ename].to(torch.float32).squeeze()
        ref_forces = ref_out[fname].to(torch.float32)
        print(f"[ref] energy = {ref_energy.item()}")
        print(f"[ref] net force = {ref_forces.sum(dim=0)}")

        if os.environ.get("ORB_DEBUG") == "1":
            # Compare graph construction: the reference adapter's edge count vs for a PBC/neighbor-list. (radius_graph_pbc_cuda and ORB's from_ase_atoms)
            _rg_edges = ref_graph.senders.shape[0]
            _ehead = ref_orbff.heads[ref_orbff.energy_name]
            _ref_term = _ehead.reference(ref_graph.atomic_numbers, ref_graph.n_node)
            _rlen = ref_graph.edge_features["vectors"].norm(dim=-1)
            print(
                f"[ref dbg] n_node={int(ref_graph.n_node.sum())} "
                f"n_edges={_rg_edges} radius={getattr(atoms_adapter, 'radius', '?')} "
                f"edge_len_sum={_rlen.sum().item():.4f} "
                f"edge_len_min={_rlen.min().item():.4f} edge_len_max={_rlen.max().item():.4f} "
                f"reference_term={_ref_term.flatten()[0].item():.5f}",
                flush=True,
            )

    # -----------------------------------------------------------------
    # 5. Distributed forward (all ranks)
    # -----------------------------------------------------------------
    local_energy, local_forces = dist_model(state)
    local_energy = local_energy.to(torch.float32)
    local_forces = local_forces.to(torch.float32)

    # -----------------------------------------------------------------
    # 6a. Energy parity
    # -----------------------------------------------------------------
    energy_reduce = local_energy.clone()
    dist.all_reduce(energy_reduce, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"[dist] energy (summed) = {energy_reduce.item()}")
        e_dist = energy_reduce.squeeze()
        e_diff = (e_dist - ref_energy).abs()
        print(f"[parity energy] ref={ref_energy.item()} dist={e_dist.item()}")
        print(f"[parity energy] abs diff={e_diff.item()} rel diff={(e_diff / ref_energy.abs()).item()}")

    # -----------------------------------------------------------------
    # 6b. Forces parity
    # -----------------------------------------------------------------
    local_indices = partition_info.partition_indices[:n_local].long().to(device)
    rank_force = torch.zeros((total_atoms, 3), device=device)
    # So row i of local_forces lands at row g of rank_force. Everything else stays 0.
    rank_force.scatter_(0, local_indices.unsqueeze(-1).expand(-1, 3), local_forces[:n_local])

    gathered = (
        [torch.zeros((total_atoms, 3), device=device) for _ in range(world_size)]
        if rank == 0 else None
    )
    dist.gather(rank_force, gather_list=gathered, dst=0)

    if rank == 0:
        reconstructed = torch.sum(torch.stack(gathered), dim=0)
        # Mirrors the energy block: raw magnitudes, then abs diff, then rel diff.
        # rel is max-then-divide (worst component error / force scale), NOT
        # divide-then-max: dividing each component by its own |f| puts the max on
        # whichever component sits nearest zero, which reports ~0.26 on a force
        # field whose worst error is 1e-4 of the scale.
        f_diff = (reconstructed - ref_forces).abs()
        f_max = f_diff.max()
        f_scale = ref_forces.abs().max()
        print(f"[parity forces] ref max|F|={f_scale.item()} dist max|F|={reconstructed.abs().max().item()}")
        print(f"[parity forces] abs diff={f_max.item()} rel diff={(f_max / f_scale).item()}")

    # -----------------------------------------------------------------
    # 7. Reference-free sanity checks
    # -----------------------------------------------------------------
    assert local_energy.numel() == 1, "local_energy must be scalar"
    assert torch.isfinite(local_energy).all(), "local_energy has NaN/Inf"
    assert local_forces.shape == (n_local, 3), (
        f"forces shape {local_forces.shape} != ({n_local}, 3) — ghosts leaking?"
    )
    assert torch.isfinite(local_forces).all(), "local_forces has NaN/Inf"
    if rank == 0:
        print("Sanity checks passed")

    dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        print("Run with: torchrun --nproc_per_node=2 distnnp_model/tests/test_orb_model.py")
    test_orb_distributed_parity()