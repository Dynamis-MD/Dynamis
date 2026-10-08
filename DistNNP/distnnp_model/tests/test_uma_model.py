"""Distributed UMAModel parity tests across fairchem inference settings.

Run with torchrun, e.g.:
    torchrun --nproc_per_node=2 -m pytest -s distnnp_model/tests/test_uma_model.py

Every configuration is compared against a float64, eager, fairchem-native
ground truth. The tolerance for each configuration is calibrated from the
fairchem reference model itself: it is run several times with the *same*
inference settings on randomly permuted copies of the system (which changes
reduction order the way partitioning does), and its worst deviation from the
ground truth, scaled by a safety factor, is the error budget the distributed
model has to meet. This lets TF32/compiled configurations pass with a
tolerance that is still tight enough to catch real bugs (e.g. missing
cross-rank gradient contributions, which show up as O(0.1-1) force errors).
"""

import dataclasses
import datetime
import os

import pytest
import torch
import torch.distributed as dist

from distnnp_model.nnp import UMAModel
from distnnp_state import (
    DistEnvironment,
    DistNNPState,
    KDTreePartitioner,
    MolecularInfo,
    SimulationConfig,
    radius_graph_pbc_cuda,
)

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import guess_inference_settings

    from distnnp_model.nnp.uma import _EV_PER_A_TO_AMU_A_PER_FS2

    HAS_FAIRCHEM = True
except ImportError:
    HAS_FAIRCHEM = False

pytestmark = [
    pytest.mark.skipif(not HAS_FAIRCHEM, reason="fairchem-core is not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

torch._inductor.config.cpp_cache_precompile_headers = False

MODEL_NAME = "uma-s-1p2"
DATASET = "omol"

TOTAL_ATOMS = 200
CELL_LENGTH = 20.0
# Atoms sit on a jittered LATTICE_SITES^3 lattice (spacing 3.33 A, jitter
# +-0.8 A => min separation >= 1.7 A). Uniformly random positions put atoms
# ~0.5 A apart, whose ~100 eV/A forces dominate max-abs errors and inflate
# the calibrated TF32 tolerance for every other atom.
LATTICE_SITES = 6
LATTICE_JITTER = 0.8
SEED = 42

# Number of fairchem reference evaluations (on permuted atom orderings) used to
# estimate the numerical noise of a given inference setting.
NUM_REFERENCE_SAMPLES = 4
NUM_DIST_CALLS = 2
# Distributed error may be at most SAFETY x the reference's worst error ...
TOL_SAFETY = 5.0
# ... but never tighter than these floors (fp32 accumulation-order noise).
ENERGY_TOL_FLOOR = 1e-3  # eV (total energy ~ -5e4 eV)
FORCE_TOL_FLOOR = 1e-4  # eV/A


def _jittered_lattice_positions(generator, device):
    spacing = CELL_LENGTH / LATTICE_SITES
    axis = torch.arange(LATTICE_SITES, dtype=torch.float32)
    sites = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    sites = sites[torch.randperm(sites.shape[0], generator=generator)[:TOTAL_ATOMS]]
    jitter = (torch.rand((TOTAL_ATOMS, 3), generator=generator) * 2 - 1) * LATTICE_JITTER
    return ((sites + 0.5) * spacing + jitter).to(device)


def _settings(base="default", **changes):
    return dataclasses.replace(guess_inference_settings(base), **changes)


# Inference-setting combinations that change UMAModel's code path. UMAModel
# always forces external_graph_gen=True, so that axis is not varied here.
# Settings with activation_checkpointing=False and merge_mole=True resolve to
# the Triton "umas_fast_gpu" backend unless execution_mode is given.
INFERENCE_CONFIGS = {
    "default": lambda: _settings("default"),
    "tf32": lambda: _settings("default", tf32=True),
    "no_act_ckpt": lambda: _settings("default", activation_checkpointing=False),
    "merge_mole": lambda: _settings("default", merge_mole=True),
    "merge_mole_tf32": lambda: _settings("default", merge_mole=True, tf32=True),
    "compile": lambda: _settings("default", compile=True),
    "compile_merge_mole": lambda: _settings("default", compile=True, merge_mole=True),
    "fast_gpu_eager": lambda: _settings("turbo", compile=False, tf32=False),
    "turbo_no_tf32": lambda: _settings("turbo", tf32=False),
    "turbo": lambda: _settings("turbo"),
    "turbo_fast_pytorch": lambda: _settings("turbo", execution_mode="umas_fast_pytorch"),
    "turbo_general": lambda: _settings("turbo", execution_mode="general"),
}

# Known upstream limitations. Strict, so we notice when fairchem fixes them.
EXPECTED_FAILURES = {
    # fairchem's MOLE layers cannot run under torch.compile unless merged
    # (MOLE.forward asserts on data-dependent mole_sizes); fairchem's own
    # predictor fails the same way.
    "compile": "fairchem MOLE does not support compile without merge_mole",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_atomic_data(positions, atomic_numbers, cell, pbc, cutoff, device):
    """Single-system AtomicData for the fairchem reference predictor."""
    num_atoms = positions.shape[0]
    with torch.no_grad():
        edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
            type(
                "Data",
                (object,),
                {
                    "pos": positions,
                    "cell": cell,
                    "natoms": torch.tensor([num_atoms], device=device),
                    "pbc": pbc,
                },
            )(),
            radius=cutoff,
            max_num_neighbors_threshold=0,
            enforce_max_neighbors_strictly=False,
        )
    data = AtomicData(
        positions,
        atomic_numbers,
        cell,
        pbc,
        torch.tensor([num_atoms], dtype=torch.long, device=device),
        edge_index,
        cell_offsets,
        torch.tensor([edge_index.shape[1]], dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),  # charge
        torch.ones(1, dtype=torch.long, device=device),  # spin
        torch.zeros(num_atoms, dtype=torch.long, device=device),  # fixed
        torch.zeros(num_atoms, dtype=torch.long, device=device),  # tags
        dataset=DATASET,
    )
    return data.to(device)


def _predict_reference(predictor, system, perm):
    """Run the fairchem predictor on a permuted copy; return (E [eV], F [eV/A]) in original order."""
    data = _build_atomic_data(
        system["positions"][perm],
        system["atomic_numbers"][perm],
        system["cell"],
        system["pbc"],
        system["cutoff"],
        system["device"],
    )
    pred = predictor.predict(data)
    energy = pred["energy"].detach().double().reshape(())
    forces = torch.empty_like(pred["forces"], dtype=torch.float64)
    forces[perm] = pred["forces"].detach().double()
    return energy, forces


def _free_compiled_state():
    """Drop compiled graphs/cached memory from models the caller has deleted."""
    torch._dynamo.reset()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dist_env():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    nnodes = int(os.environ.get("SLURM_NNODES", 1))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    owns_pg = not dist.is_initialized()
    if owns_pg:
        # Generous timeout: compiling the reference/distributed models can take
        # several minutes, during which peers wait in a collective.
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30),
        )

    env = DistEnvironment()
    env.global_rank = rank
    env.world_size = world_size
    env.local_rank = local_rank
    env.num_nodes = nnodes
    env.gpus_per_node = world_size // nnodes
    yield env

    if owns_pg:
        dist.barrier()
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ground_truth(dist_env):
    """System definition plus a float64/eager fairchem-native ground truth.

    Computed independently on every rank so that each rank can evaluate the
    same pass/fail decision without extra collectives.
    """
    device = "cuda"
    truth_settings = _settings(
        "default", base_precision_dtype=torch.float64, execution_mode="general"
    )
    predictor = pretrained_mlip.get_predict_unit(
        MODEL_NAME, device=device, inference_settings=truth_settings
    )
    # fairchem merges MOLE before moving the model; parameters must already be on device.
    predictor.move_to_device()

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    system = {
        "device": device,
        "positions": _jittered_lattice_positions(gen, device),
        "atomic_numbers": torch.randint(1, 10, (TOTAL_ATOMS,), generator=gen).to(device),
        "cell": (torch.eye(3, device=device) * CELL_LENGTH).unsqueeze(0),
        "pbc": torch.tensor([[True, True, True]], device=device),
        "cutoff": predictor.model.module.backbone.cutoff,
    }
    identity = torch.arange(TOTAL_ATOMS, device=device)
    energy, forces = _predict_reference(predictor, system, identity)
    del predictor
    _free_compiled_state()
    return system, energy, forces


@pytest.fixture(scope="module")
def dist_state(dist_env, ground_truth):
    system, _, _ = ground_truth
    device = system["device"]

    global_mol = MolecularInfo()
    global_mol.num_atoms = TOTAL_ATOMS
    global_mol.atom_vertex_numbers = torch.arange(TOTAL_ATOMS, device=device, dtype=torch.long)
    global_mol.atom_types = system["atomic_numbers"]
    global_mol.atom_positions = system["positions"]
    global_mol.atom_velocities = torch.zeros((TOTAL_ATOMS, 3), device=device)

    sim_config = SimulationConfig()
    sim_config.cell = system["cell"].squeeze(0)
    sim_config.pbc = system["pbc"].squeeze(0)
    sim_config.cutoff = system["cutoff"]
    sim_config.ghost_cutoff_buffer = 0.5

    partitioner = KDTreePartitioner()
    partition_info = partitioner.partition(global_mol, dist_env)
    partitioner.build_halo_maps(partition_info, global_mol, dist_env, sim_config)
    if dist_env.world_size > 1:
        dist.all_reduce(partition_info.global_send_recv_matrix, op=dist.ReduceOp.SUM)

    idx = partition_info.partition_indices
    local_mol = MolecularInfo()
    local_mol.atom_positions = global_mol.atom_positions[idx]
    local_mol.atom_velocities = global_mol.atom_velocities[idx]
    local_mol.atom_types = global_mol.atom_types[idx]
    local_mol.num_atoms = local_mol.atom_positions.shape[0]

    state = DistNNPState()
    state.dist_env = dist_env
    state.local_molecular_info = local_mol
    state.global_molecular_info = global_mol
    state.local_partition_info = partition_info
    state.sim_config = sim_config
    return state


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def _calibrate_tolerances(settings, system, true_energy, true_forces):
    """Worst-case deviation of the fairchem reference (same settings) from the truth."""
    ref_settings = settings
    if settings.compile and not settings.merge_mole:
        # fairchem cannot compile un-merged MOLE models (MOLE.forward asserts on
        # dynamic mole_sizes), so calibrate against its eager equivalent.
        ref_settings = dataclasses.replace(settings, compile=False)
    predictor = pretrained_mlip.get_predict_unit(
        MODEL_NAME, device=system["device"], inference_settings=ref_settings
    )
    predictor.move_to_device()
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
    energy_err = torch.zeros((), dtype=torch.float64, device=system["device"])
    force_err = torch.zeros((), dtype=torch.float64, device=system["device"])
    for i in range(NUM_REFERENCE_SAMPLES):
        perm = (
            torch.arange(TOTAL_ATOMS)
            if i == 0
            else torch.randperm(TOTAL_ATOMS, generator=gen)
        ).to(system["device"])
        energy, forces = _predict_reference(predictor, system, perm)
        energy_err = torch.maximum(energy_err, (energy - true_energy).abs())
        force_err = torch.maximum(force_err, (forces - true_forces).abs().max())
    del predictor
    _free_compiled_state()
    return energy_err.item(), force_err.item()


def _gather_distributed(dist_model, state):
    """Evaluate the distributed model; return global (E [eV], F [eV/A]) on every rank."""
    energy, local_forces = dist_model(state)
    energy = energy.detach().double().reshape(()).clone()
    n_local = state.local_molecular_info.num_atoms
    local_idx = state.local_partition_info.partition_indices[:n_local].long()
    forces = torch.zeros((TOTAL_ATOMS, 3), dtype=torch.float64, device=energy.device)
    forces[local_idx] = local_forces[:n_local].double()
    if dist.get_world_size() > 1:
        dist.all_reduce(energy)
        dist.all_reduce(forces)
    return energy / _EV_PER_A_TO_AMU_A_PER_FS2, forces / _EV_PER_A_TO_AMU_A_PER_FS2


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(
            name,
            marks=[pytest.mark.xfail(reason=EXPECTED_FAILURES[name], raises=AssertionError, strict=True)]
            if name in EXPECTED_FAILURES
            else [],
        )
        for name in INFERENCE_CONFIGS
    ],
)
def test_uma_distributed_parity(config_name, dist_env, ground_truth, dist_state):
    system, true_energy, true_forces = ground_truth
    settings = INFERENCE_CONFIGS[config_name]()

    ref_energy_err, ref_force_err = _calibrate_tolerances(
        settings, system, true_energy, true_forces
    )
    energy_tol = max(TOL_SAFETY * ref_energy_err, ENERGY_TOL_FLOOR)
    force_tol = max(TOL_SAFETY * ref_force_err, FORCE_TOL_FLOOR)

    dist_model = UMAModel(
        MODEL_NAME, dataset=DATASET, device=system["device"], inference_settings=settings
    )
    dist_model.initialize_buffers(dist_state.local_partition_info, system["device"])

    energy_errs, force_errs = [], []
    for _ in range(NUM_DIST_CALLS):
        energy, forces = _gather_distributed(dist_model, dist_state)
        energy_errs.append((energy - true_energy).abs().item())
        force_errs.append((forces - true_forces).abs().max().item())
    del dist_model
    _free_compiled_state()

    if dist_env.global_rank == 0:
        print(
            f"\n[{config_name}] reference err: E={ref_energy_err:.3e} F={ref_force_err:.3e} | "
            f"distributed err: E={max(energy_errs):.3e} F={max(force_errs):.3e} | "
            f"tol: E={energy_tol:.3e} F={force_tol:.3e}"
        )
    # Every rank holds identical all-reduced results, so every rank reaches the
    # same verdict and no rank is left waiting in a collective.
    for step, (e_err, f_err) in enumerate(zip(energy_errs, force_errs)):
        assert e_err <= energy_tol, (
            f"[{config_name}] call {step}: energy error {e_err:.3e} eV exceeds "
            f"tolerance {energy_tol:.3e} eV (reference noise {ref_energy_err:.3e})"
        )
        assert f_err <= force_tol, (
            f"[{config_name}] call {step}: max force error {f_err:.3e} eV/A exceeds "
            f"tolerance {force_tol:.3e} eV/A (reference noise {ref_force_err:.3e})"
        )
