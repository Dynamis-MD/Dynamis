"""Phase 1 distributed-input MACE adapter for DistNNP.

Phase 1 validates only the first distributed bridge:

local DistNNP partition
-> exchange raw positions / atomic numbers
-> local + halo molecular view
-> standard MACE forward

It intentionally does not solve ownership-aware energy or distributed force
correctness yet.

Step 1 baseline role
--------------------
Treat this file as the frozen distributed structural-wrapper baseline. It
should remain the outer-shell reference for raw structural exchange and local
plus halo reconstruction while ownership-aware energy, autograd-correct local
forces, and internal learned-feature exchange are prototyped separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.potential import DistNNPModel
from distnnp_state import (
    DistEnvironment,
    DistNNPState,
    MolecularInfo,
    PartitionMessagePassingInfo,
    SFCCurveType,
    SFCPartitioner,
    SimulationConfig,
    TorchDistEnv,
)
from distnnp_state.masses import TORCH_ATOMIC_MASSES

try:
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None
    _MACE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "MACE imports failed. Install the public 'mace-torch' package in the "
            "runtime environment before using the Phase 1 adapter."
        ) from _MACE_IMPORT_ERROR


def load_raw_mace_model(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    _require_mace()
    common = {"model": model, "device": device, "return_raw_model": True}
    if model_family == "mp":
        return mace_mp(**common)
    if model_family == "off":
        return mace_off(**common)
    if model_family == "omol":
        return mace_omol(**common)
    if model_family == "polar":
        return mace_polar(**common)
    raise ValueError(f"Unsupported MACE model family: {model_family}")


def align_batch_dtype(batch_dict: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    model_dtype = next(model.parameters()).dtype
    aligned: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            aligned[key] = value.to(dtype=model_dtype)
        else:
            aligned[key] = value
    return aligned


BUILTIN_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "adapter_xyz6": {
        "kind": "file",
        "description": "6-atom non-overlapping xyz fixture stored inside the adaptation folder.",
    },
    "adapter_boundary_halo": {
        "kind": "file",
        "description": (
            "8-atom boundary-straddling xyz fixture laid out to create cross-partition "
            "neighbors and therefore nonzero halo traffic in a 2-rank run."
        ),
    },
    "adapter_xyz6_halo": {
        "kind": "file",
        "description": (
            "Compatibility alias for the boundary-straddling halo fixture. "
            "Prefer --builtin-system adapter_boundary_halo in new runs."
        ),
    },
    "distnnp_xyz6": {
        "kind": "file",
        "description": "6-atom DistNNP xyz fixture with overlapping coordinates.",
    },
}

DEFAULT_ADAPTER_STRUCTURE_PATH = Path(__file__).resolve().with_name("sample_nonoverlap.xyz")
DEFAULT_HALO_STRUCTURE_PATH = Path(__file__).resolve().with_name("sample_halo_boundary.xyz")
DEFAULT_REAL_STRUCTURE_PATH = (
    Path(__file__).resolve().parents[5] / "distnnp_state" / "tests" / "xyz.xyz"
)

ELEMENT_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 distributed-input MACE adapter: exchange raw positions and "
            "atomic numbers, then run standard MACE forward on local+halo atoms."
        )
    )
    parser.add_argument(
        "--builtin-system",
        choices=sorted(BUILTIN_SYSTEMS.keys()),
        default="adapter_boundary_halo",
        help=(
            "Built-in structure to use when --xyz-path is not provided. "
            "Defaults to the halo-producing Phase 1 verification fixture."
        ),
    )
    parser.add_argument(
        "--xyz-path",
        help="Optional path to a simple XYZ/extxyz structure file.",
    )
    parser.add_argument(
        "--model-family",
        choices=["mp", "off", "omol", "polar"],
        default="mp",
        help="Public MACE foundation-model family to load.",
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help="Foundation-model key or checkpoint path understood by the selected loader.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device for the loaded model.",
    )
    parser.add_argument(
        "--partitioner",
        choices=["sfc"],
        default="sfc",
        help="Partitioner to use for constructing the distributed state.",
    )
    parser.add_argument(
        "--ghost-cutoff-buffer",
        type=float,
        default=0.5,
        help="Ghost cutoff buffer used when building halo maps.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save a JSON summary from rank 0.",
    )
    return parser


def parse_xyz_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    if len(text) < 2:
        raise ValueError(f"XYZ file is too short: {path}")

    num_atoms = int(text[0].strip())
    comment = text[1].strip()
    atom_lines = text[2 : 2 + num_atoms]
    if len(atom_lines) != num_atoms:
        raise ValueError(
            f"Expected {num_atoms} atom lines in {path}, found {len(atom_lines)}"
        )

    atomic_numbers = []
    positions = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ atom line in {path}: {line}")
        symbol = fields[0]
        if symbol not in ELEMENT_TO_Z:
            raise ValueError(f"Unsupported element symbol '{symbol}' in {path}")
        atomic_numbers.append(ELEMENT_TO_Z[symbol])
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])

    cell = np.zeros((3, 3), dtype=np.float64)
    pbc = [False, False, False]
    lattice_match = re.search(r'Lattice="([^"]+)"', comment)
    if lattice_match:
        lattice_values = [float(x) for x in lattice_match.group(1).split()]
        if len(lattice_values) == 9:
            cell = np.asarray(lattice_values, dtype=np.float64).reshape(3, 3)

    pbc_match = re.search(r'pbc="([^"]+)"', comment)
    if pbc_match:
        pbc_tokens = pbc_match.group(1).split()
        if len(pbc_tokens) == 3:
            pbc = [token.upper().startswith("T") for token in pbc_tokens]

    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": cell.tolist(),
        "pbc": pbc,
        "source_path": str(path),
    }


def parse_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    if args.xyz_path:
        return parse_xyz_file(Path(args.xyz_path))
    if args.builtin_system == "adapter_xyz6":
        return parse_xyz_file(DEFAULT_ADAPTER_STRUCTURE_PATH)
    if args.builtin_system == "adapter_boundary_halo":
        return parse_xyz_file(DEFAULT_HALO_STRUCTURE_PATH)
    if args.builtin_system == "adapter_xyz6_halo":
        return parse_xyz_file(DEFAULT_HALO_STRUCTURE_PATH)
    if args.builtin_system == "distnnp_xyz6":
        return parse_xyz_file(DEFAULT_REAL_STRUCTURE_PATH)
    raise ValueError(f"Unsupported built-in system: {args.builtin_system}")


def analyze_structure(raw_structure: Dict[str, Any]) -> Dict[str, Any]:
    positions = np.asarray(raw_structure["positions"], dtype=np.float64)
    diagnostics: Dict[str, Any] = {
        "has_duplicate_positions": False,
        "duplicate_pairs": [],
        "min_pair_distance_A": None,
        "min_pair_pair": None,
    }
    if len(positions) < 2:
        return diagnostics

    min_distance = float("inf")
    min_pair = None
    duplicate_pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < min_distance:
                min_distance = distance
                min_pair = [i, j]
            if np.allclose(positions[i], positions[j], atol=0.0, rtol=0.0):
                duplicate_pairs.append([i, j])

    diagnostics["has_duplicate_positions"] = len(duplicate_pairs) > 0
    diagnostics["duplicate_pairs"] = duplicate_pairs
    diagnostics["min_pair_distance_A"] = min_distance
    diagnostics["min_pair_pair"] = min_pair
    return diagnostics


def _csr_counts(offsets: torch.Tensor) -> list[int]:
    offsets_long = offsets.to(torch.int64)
    if offsets_long.ndim != 1 or offsets_long.numel() < 2:
        raise ValueError(
            f"Expected a 1D CSR offsets tensor with at least 2 entries, got shape {list(offsets_long.shape)}"
        )
    return (
        offsets_long[1:] - offsets_long[:-1]
    ).detach().cpu().to(torch.int64).tolist()


def validate_phase1_partition_info(
    partition_info: PartitionMessagePassingInfo,
    world_size: int,
) -> Dict[str, Any]:
    """Validate the partition/halo metadata required by the outer wrapper.

    Step 2 depends on this metadata being structurally sound before we even try
    to build a local-plus-halo MACE view. These checks stay deliberately
    wrapper-focused: they validate tensor presence, CSR layout, and local/halo
    accounting, but do not impose ownership-aware energy semantics yet.
    """

    required_names = [
        "partition_indices",
        "send_offsets",
        "send_local_indices",
        "recv_offsets",
        "recv_ghost_indices",
        "partition_id",
    ]
    missing = [
        name
        for name in required_names
        if not hasattr(partition_info, name) or getattr(partition_info, name) is None
    ]
    if missing:
        raise ValueError(
            "PartitionMessagePassingInfo is missing required fields for the Phase 1 wrapper: "
            + ", ".join(missing)
        )

    partition_indices = partition_info.partition_indices.to(torch.long)
    send_offsets = partition_info.send_offsets.to(torch.int64)
    recv_offsets = partition_info.recv_offsets.to(torch.int64)
    send_local_indices = partition_info.send_local_indices.to(torch.int64)
    recv_ghost_indices = partition_info.recv_ghost_indices.to(torch.int64)

    if send_offsets.numel() != world_size + 1:
        raise ValueError(
            f"send_offsets must have shape [world_size + 1] = [{world_size + 1}], got {list(send_offsets.shape)}"
        )
    if recv_offsets.numel() != world_size + 1:
        raise ValueError(
            f"recv_offsets must have shape [world_size + 1] = [{world_size + 1}], got {list(recv_offsets.shape)}"
        )

    if not torch.all(send_offsets[1:] >= send_offsets[:-1]):
        raise ValueError("send_offsets is not monotonic nondecreasing.")
    if not torch.all(recv_offsets[1:] >= recv_offsets[:-1]):
        raise ValueError("recv_offsets is not monotonic nondecreasing.")
    if int(send_offsets[0].item()) != 0:
        raise ValueError(f"send_offsets must start at 0, got {int(send_offsets[0].item())}")
    if int(recv_offsets[0].item()) != 0:
        raise ValueError(f"recv_offsets must start at 0, got {int(recv_offsets[0].item())}")

    total_send = int(send_offsets[-1].item())
    total_recv = int(recv_offsets[-1].item())
    if send_local_indices.numel() != total_send:
        raise ValueError(
            f"send_local_indices has {send_local_indices.numel()} entries but send_offsets encodes total_send={total_send}."
        )
    if recv_ghost_indices.numel() != total_recv:
        raise ValueError(
            f"recv_ghost_indices has {recv_ghost_indices.numel()} entries but recv_offsets encodes total_recv={total_recv}."
        )

    n_local = int(partition_indices.numel())
    if total_send > 0:
        if int(send_local_indices.min().item()) < 0 or int(send_local_indices.max().item()) >= n_local:
            raise ValueError(
                "send_local_indices contains entries outside the local atom range "
                f"[0, {max(0, n_local - 1)}]."
            )

    expected_ghost_indices = torch.arange(
        n_local,
        n_local + total_recv,
        dtype=torch.int64,
        device=recv_ghost_indices.device,
    )
    if total_recv > 0 and not torch.equal(recv_ghost_indices, expected_ghost_indices):
        raise ValueError(
            "recv_ghost_indices is expected to enumerate contiguous ghost slots "
            f"starting at num_local={n_local}, but got {recv_ghost_indices.detach().cpu().tolist()}."
        )

    send_counts = _csr_counts(send_offsets)
    recv_counts = _csr_counts(recv_offsets)
    return {
        "n_local": n_local,
        "total_send": total_send,
        "total_recv": total_recv,
        "send_counts_by_peer": send_counts,
        "recv_counts_by_peer": recv_counts,
    }


def validate_phase1_state(
    distributed_state: DistNNPState,
) -> Dict[str, Any]:
    """Validate that the local view still matches the owned partition indices."""

    partition_validation = validate_phase1_partition_info(
        distributed_state.local_partition_info,
        int(distributed_state.dist_env.world_size),
    )
    local_info = distributed_state.local_molecular_info
    global_info = distributed_state.global_molecular_info
    local_indices = distributed_state.local_partition_info.partition_indices.to(torch.long)

    if int(local_info.num_atoms) != partition_validation["n_local"]:
        raise ValueError(
            f"local_molecular_info.num_atoms={int(local_info.num_atoms)} does not match "
            f"partition_indices size {partition_validation['n_local']}."
        )

    expected_positions = global_info.atom_positions.index_select(0, local_indices)
    expected_atom_types = global_info.atom_types.index_select(0, local_indices)
    if local_info.atom_positions.shape != expected_positions.shape:
        raise ValueError(
            "local_molecular_info.atom_positions shape does not match the owned partition shape: "
            f"{list(local_info.atom_positions.shape)} vs {list(expected_positions.shape)}."
        )
    if local_info.atom_types.shape != expected_atom_types.shape:
        raise ValueError(
            "local_molecular_info.atom_types shape does not match the owned partition shape: "
            f"{list(local_info.atom_types.shape)} vs {list(expected_atom_types.shape)}."
        )
    if not torch.equal(local_info.atom_positions, expected_positions):
        raise ValueError(
            "local_molecular_info.atom_positions no longer matches global_molecular_info[partition_indices]."
        )
    if not torch.equal(local_info.atom_types, expected_atom_types):
        raise ValueError(
            "local_molecular_info.atom_types no longer matches global_molecular_info[partition_indices]."
        )

    return partition_validation


def build_global_molecular_info(
    raw_structure: Dict[str, Any],
    device: torch.device,
) -> MolecularInfo:
    def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    num_atoms = len(raw_structure["atomic_numbers"])
    atomic_numbers = torch.tensor(
        raw_structure["atomic_numbers"], dtype=torch.long, device=device
    )

    mol_info = MolecularInfo()
    mol_info.num_atoms = num_atoms
    mol_info.atom_positions = torch.tensor(
        raw_structure["positions"], dtype=torch.float32, device=device
    )
    mol_info.atom_types = atomic_numbers
    mol_info.atom_velocities = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)
    mol_info.atom_forces = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)
    _assign_if_present(
        mol_info,
        "atom_vertex_numbers",
        # Partitioners propagate atom_vertex_numbers into partition_indices,
        # and SimState later uses partition_indices as tensor indices.
        # PyTorch accepts int64/long indexing, but not uint64 indexing.
        # For Phase 1 we therefore prefer long here, even though some IO tests
        # elsewhere compare atom_vertex_numbers against uint64.
        torch.arange(num_atoms, dtype=torch.long, device=device),
    )
    _assign_if_present(
        mol_info,
        "atom_masses",
        TORCH_ATOMIC_MASSES.to(device)[atomic_numbers].to(torch.float32),
    )
    return mol_info


def build_sim_config(
    raw_structure: Dict[str, Any],
    cutoff: float,
    ghost_cutoff_buffer: float,
    device: torch.device,
) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.cutoff = cutoff
    cfg.ghost_cutoff_buffer = ghost_cutoff_buffer
    cfg.cell = torch.tensor(raw_structure["cell"], dtype=torch.float32, device=device)
    cfg.pbc = torch.tensor(raw_structure["pbc"], dtype=torch.bool, device=device)
    return cfg


def build_phase1_distnnp_state(
    global_molecular_info: MolecularInfo,
    partitioner,
    sim_config: SimulationConfig,
    dist_env: DistEnvironment,
    device: torch.device,
) -> DistNNPState:
    """Build a partitioned DistNNPState without relying on SimState's mass assumptions.

    The cluster build currently in use appears to expose a MolecularInfo
    without an ``atom_masses`` attribute, while ``SimState`` unconditionally
    accesses that field. Phase 1 only needs positions/types/partition info, so
    we construct the local state directly here and assign optional fields only
    when the loaded extension exposes them.
    """

    def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    partition_info = partitioner.partition(global_molecular_info, dist_env)
    partitioner.build_halo_maps(
        partition_info,
        global_molecular_info,
        dist_env,
        sim_config,
    )

    local_indices = partition_info.partition_indices.to(torch.long)
    local_molecular_info = MolecularInfo()
    local_molecular_info.num_atoms = int(local_indices.numel())
    local_molecular_info.atom_positions = global_molecular_info.atom_positions.index_select(
        0, local_indices
    )
    local_molecular_info.atom_velocities = global_molecular_info.atom_velocities.index_select(
        0, local_indices
    )
    local_molecular_info.atom_types = global_molecular_info.atom_types.index_select(
        0, local_indices
    )
    local_molecular_info.atom_forces = torch.zeros(
        local_indices.numel(), 3, dtype=torch.float32, device=device
    )

    if hasattr(global_molecular_info, "atom_vertex_numbers"):
        _assign_if_present(
            local_molecular_info,
            "atom_vertex_numbers",
            global_molecular_info.atom_vertex_numbers.index_select(0, local_indices),
        )
    if hasattr(global_molecular_info, "atom_masses"):
        _assign_if_present(
            local_molecular_info,
            "atom_masses",
            global_molecular_info.atom_masses.index_select(0, local_indices),
        )

    state = DistNNPState()
    state.dist_env = dist_env
    state.sim_config = sim_config
    state.global_molecular_info = global_molecular_info
    state.local_partition_info = partition_info
    state.local_molecular_info = local_molecular_info
    validate_phase1_state(state)
    return state


class MACEPotentialModelWithRawExchange(DistNNPModel):
    def __init__(
        self,
        model,
        atomic_number_map: Optional[Mapping[int, int]] = None,
    ) -> None:
        _require_mace()
        super().__init__()
        self.model = model
        self.model.eval()
        self.atomic_number_map = dict(atomic_number_map or {})
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        self.z_table = AtomicNumberTable(
            [int(z) for z in model.atomic_numbers.detach().cpu().tolist()]
        )
        self.cutoff = float(model.r_max.detach().cpu().item())
        self.heads = getattr(model, "heads", ["Default"])
        self.last_observation: Dict[str, Any] = {}
        self.exchange_mode = "uninitialized"
        self.exchange_reason = "uninitialized"
        self.total_send = 0
        self.total_recv = 0
        self.partition_validation: Dict[str, Any] = {}
        self.world_size = 1

    def initialize_buffers(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
        world_size: int,
    ) -> None:
        validation = validate_phase1_partition_info(
            partition_info,
            world_size,
        )
        self.world_size = int(world_size)
        self.partition_validation = validation
        self.total_send = validation["total_send"]
        self.total_recv = validation["total_recv"]

        # Zero-sized IPC buffers cannot be exported with cudaIpcGetMemHandle.
        # For Phase 1 we fall back to a graph-derived local+halo reconstruction
        # if one side of the exchange would be empty, and we skip communication
        # entirely when there is no halo traffic at all.
        if self.total_send == 0 and self.total_recv == 0:
            self.exchange_mode = "no_halo_identity"
            self.exchange_reason = "No halo traffic was requested by build_halo_maps."
            return
        if self.total_send == 0 or self.total_recv == 0:
            self.exchange_mode = "graph_fallback"
            self.exchange_reason = (
                "Asymmetric halo traffic would create a zero-sized IPC buffer; "
                "using deterministic global-state halo reconstruction instead."
            )
            return

        try:
            self.position_exchanger.setup_exchanger(
                feature_dim=3,
                partition_info=partition_info,
                device=device,
            )
            self.atom_type_exchanger.setup_exchanger(
                feature_dim=1,
                partition_info=partition_info,
                device=device,
            )
            self.exchange_mode = "ipc"
            self.exchange_reason = "CUDA IPC halo exchange setup succeeded."
        except Exception as exc:
            self.position_exchanger.reset_buffers()
            self.atom_type_exchanger.reset_buffers()
            self.exchange_mode = "graph_fallback"
            self.exchange_reason = (
                "IPC setup failed; falling back to deterministic global-state "
                f"halo reconstruction. Root cause: {exc}"
            )

    @staticmethod
    def _derive_halo_global_indices(distributed_state: DistNNPState) -> torch.Tensor:
        partition_info = distributed_state.local_partition_info
        edge_index_halo = partition_info.edge_index_halo
        if edge_index_halo is None or edge_index_halo.numel() == 0:
            return torch.empty(
                0,
                dtype=torch.long,
                device=distributed_state.local_molecular_info.atom_positions.device,
            )

        src = edge_index_halo[0].to(torch.long)
        dst = edge_index_halo[1].to(torch.long)
        partition_id = partition_info.partition_id.to(torch.long)
        my_rank = int(distributed_state.dist_env.global_rank)
        world_size = int(distributed_state.dist_env.world_size)

        dst_is_mine = partition_id.index_select(0, dst).eq(my_rank)
        src_is_remote = partition_id.index_select(0, src).ne(my_rank)
        remote_src = src[dst_is_mine & src_is_remote]
        if remote_src.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=src.device)

        ordered_groups = []
        src_owner = partition_id.index_select(0, remote_src)
        for peer in range(world_size):
            if peer == my_rank:
                continue
            peer_src = remote_src[src_owner.eq(peer)]
            if peer_src.numel() == 0:
                continue
            ordered_groups.append(torch.unique(peer_src.sort()[0]))

        if not ordered_groups:
            return torch.empty(0, dtype=torch.long, device=src.device)
        return torch.cat(ordered_groups, dim=0)

    def _expand_via_global_halo(
        self,
        local_positions: torch.Tensor,
        local_atom_types: torch.Tensor,
        distributed_state: DistNNPState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        halo_global_indices = self._derive_halo_global_indices(distributed_state)
        expected_total_recv = self.total_recv
        if halo_global_indices.numel() != expected_total_recv:
            raise ValueError(
                "Graph-derived halo reconstruction did not match the halo CSR metadata: "
                f"derived {int(halo_global_indices.numel())} ghost atoms but recv_offsets encodes {expected_total_recv}. "
                "This usually means the outer wrapper's halo ordering assumptions are inconsistent."
            )
        if halo_global_indices.numel() == 0:
            return local_positions, local_atom_types.to(torch.long)

        global_info = distributed_state.global_molecular_info
        halo_positions = global_info.atom_positions.index_select(
            0, halo_global_indices
        ).detach()
        halo_atom_types = global_info.atom_types.index_select(
            0, halo_global_indices
        ).to(torch.long)
        expanded_positions = torch.cat([local_positions, halo_positions], dim=0)
        expanded_atom_types = torch.cat(
            [local_atom_types.to(torch.long), halo_atom_types], dim=0
        )
        return expanded_positions, expanded_atom_types

    def _expand_local_plus_halo_view(
        self,
        local_positions: torch.Tensor,
        local_atom_types: torch.Tensor,
        distributed_state: DistNNPState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.exchange_mode == "ipc":
            exchanged_positions = FeaturePushToExchange.apply(
                local_positions,
                distributed_state,
                self.position_exchanger,
            )
            with torch.no_grad():
                exchanged_atom_types = FeaturePushToExchange.apply(
                    local_atom_types.float().unsqueeze(1),
                    distributed_state,
                    self.atom_type_exchanger,
                ).squeeze(1).long()
            return exchanged_positions, exchanged_atom_types

        if self.exchange_mode == "no_halo_identity":
            return local_positions, local_atom_types.to(torch.long)

        return self._expand_via_global_halo(
            local_positions,
            local_atom_types,
            distributed_state,
        )

    def _map_atomic_numbers(self, atom_types: torch.Tensor) -> torch.Tensor:
        atom_types_long = atom_types.to(torch.long)
        if not self.atomic_number_map:
            return atom_types_long
        mapped = [self.atomic_number_map[int(atom_type)] for atom_type in atom_types_long.detach().cpu()]
        return torch.tensor(mapped, dtype=torch.long, device=atom_types.device)

    @staticmethod
    def _cell_to_numpy(cell: torch.Tensor) -> np.ndarray:
        cell_cpu = cell.detach().cpu()
        if cell_cpu.ndim == 3 and cell_cpu.shape[0] == 1:
            cell_cpu = cell_cpu[0]
        return cell_cpu.to(torch.float64).numpy()

    @staticmethod
    def _pbc_to_tuple(pbc: torch.Tensor) -> tuple[bool, bool, bool]:
        pbc_flat = pbc.detach().cpu().reshape(-1).tolist()
        if len(pbc_flat) != 3:
            raise ValueError(f"Expected pbc to contain 3 values, got {pbc_flat}")
        return tuple(bool(value) for value in pbc_flat)

    def _build_configuration(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        distributed_state: DistNNPState,
    ) -> Configuration:
        sim_config = distributed_state.sim_config
        return Configuration(
            atomic_numbers=atomic_numbers.detach().cpu().to(torch.int64).numpy(),
            positions=positions.detach().cpu().to(torch.float64).numpy(),
            properties={},
            property_weights={},
            cell=self._cell_to_numpy(sim_config.cell),
            pbc=self._pbc_to_tuple(sim_config.pbc),
            weight=1.0,
            config_type="DistNNP_Phase1_RawExchange",
            head="Default",
        )

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        validation = validate_phase1_state(distributed_state)
        self.partition_validation = validation
        self.total_send = validation["total_send"]
        self.total_recv = validation["total_recv"]

        local_positions = (
            distributed_state.local_molecular_info.atom_positions.clone().detach().requires_grad_(True)
        )
        local_atom_types = distributed_state.local_molecular_info.atom_types

        exchanged_positions, exchanged_atom_types = self._expand_local_plus_halo_view(
            local_positions,
            local_atom_types,
            distributed_state,
        )

        exchanged_atomic_numbers = self._map_atomic_numbers(exchanged_atom_types)
        n_local = int(local_positions.shape[0])
        n_total = int(exchanged_positions.shape[0])
        n_halo = int(n_total - n_local)

        config = self._build_configuration(
            exchanged_atomic_numbers,
            exchanged_positions,
            distributed_state,
        )
        atomic_data = AtomicData.from_config(
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.heads,
        )
        edge_count = int(atomic_data.edge_index.shape[1])

        batch = Batch.from_data_list([atomic_data]).to(local_positions.device)
        batch_dict = align_batch_dtype(batch.to_dict(), self.model)
        outputs = self.model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_stress=False,
            compute_hessian=False,
        )

        forces = outputs["forces"]
        energy = outputs["energy"]
        if torch.is_tensor(energy) and energy.numel() == 1:
            energy = energy.reshape(())

        self.last_observation = {
            "rank": int(distributed_state.dist_env.global_rank),
            "n_local": n_local,
            "n_total_after_exchange": n_total,
            "n_halo": n_halo,
            "expected_total_recv": self.total_recv,
            "local_position_shape": list(local_positions.shape),
            "exchanged_position_shape": list(exchanged_positions.shape),
            "local_atom_type_shape": list(local_atom_types.shape),
            "exchanged_atom_type_shape": list(exchanged_atomic_numbers.shape),
            "cell": self._cell_to_numpy(distributed_state.sim_config.cell).tolist(),
            "pbc": list(self._pbc_to_tuple(distributed_state.sim_config.pbc)),
            "cutoff": self.cutoff,
            "exchange_mode": self.exchange_mode,
            "exchange_reason": self.exchange_reason,
            "total_send": self.total_send,
            "total_recv": self.total_recv,
            "send_counts_by_peer": validation["send_counts_by_peer"],
            "recv_counts_by_peer": validation["recv_counts_by_peer"],
            "edge_count_after_atomic_data": edge_count,
            "local_atomic_numbers": exchanged_atomic_numbers[:n_local].detach().cpu().tolist(),
            "halo_atomic_numbers": exchanged_atomic_numbers[n_local:].detach().cpu().tolist(),
            "position_min": exchanged_positions.min(dim=0).values.detach().cpu().tolist(),
            "position_max": exchanged_positions.max(dim=0).values.detach().cpu().tolist(),
            "energy": float(energy.detach().cpu().item()) if torch.is_tensor(energy) else float(energy),
        }

        return energy.detach(), forces[:n_local].detach()


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    wrapper_checks = {
        "all_expected_halo_counts_match": all(
            record["n_halo"] == record["expected_total_recv"] for record in rank_records_sorted
        ),
        "all_force_shapes_match_owned_atoms": all(
            record["returned_force_shape"] == [record["n_local"], 3]
            for record in rank_records_sorted
        ),
        "all_energies_finite": all(np.isfinite(record["energy"]) for record in rank_records_sorted),
        "exchange_modes": [record["exchange_mode"] for record in rank_records_sorted],
    }
    return {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "wrapper_checks": wrapper_checks,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("Phase 1 Raw-Exchange MACE Adapter")
    print("============================================================")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print()
    print("------------------------------------------------------------")
    print("Global Input")
    print("------------------------------------------------------------")
    print(f"source_path: {summary['source_path']}")
    print(f"num_atoms: {summary['num_atoms']}")
    print(f"atomic_numbers: {summary['atomic_numbers']}")
    print(f"positions_A: {summary['positions_A']}")
    print()
    print("------------------------------------------------------------")
    print("Structure Diagnostics")
    print("------------------------------------------------------------")
    diagnostics = summary["structure_diagnostics"]
    print(f"min_pair_distance_A: {diagnostics['min_pair_distance_A']}")
    print(f"min_pair_pair: {diagnostics['min_pair_pair']}")
    print(f"has_duplicate_positions: {diagnostics['has_duplicate_positions']}")
    print(f"duplicate_pairs: {diagnostics['duplicate_pairs']}")
    print()
    print("------------------------------------------------------------")
    print("Per-Rank Observation Points")
    print("------------------------------------------------------------")
    wrapper_checks = summary["wrapper_checks"]
    print(
        f"all_expected_halo_counts_match: {wrapper_checks['all_expected_halo_counts_match']}"
    )
    print(
        f"all_force_shapes_match_owned_atoms: {wrapper_checks['all_force_shapes_match_owned_atoms']}"
    )
    print(f"all_energies_finite: {wrapper_checks['all_energies_finite']}")
    print(f"exchange_modes: {wrapper_checks['exchange_modes']}")
    print()
    for record in summary["rank_records"]:
        print(f"rank: {record['rank']}")
        print(f"  n_local: {record['n_local']}")
        print(f"  n_total_after_exchange: {record['n_total_after_exchange']}")
        print(f"  n_halo: {record['n_halo']}")
        print(f"  expected_total_recv: {record['expected_total_recv']}")
        print(f"  local_position_shape: {record['local_position_shape']}")
        print(f"  exchanged_position_shape: {record['exchanged_position_shape']}")
        print(f"  local_atom_type_shape: {record['local_atom_type_shape']}")
        print(f"  exchanged_atom_type_shape: {record['exchanged_atom_type_shape']}")
        print(f"  cutoff: {record['cutoff']}")
        print(f"  exchange_mode: {record['exchange_mode']}")
        print(f"  exchange_reason: {record['exchange_reason']}")
        print(f"  total_send: {record['total_send']}")
        print(f"  total_recv: {record['total_recv']}")
        print(f"  send_counts_by_peer: {record['send_counts_by_peer']}")
        print(f"  recv_counts_by_peer: {record['recv_counts_by_peer']}")
        print(f"  pbc: {record['pbc']}")
        print(f"  edge_count_after_atomic_data: {record['edge_count_after_atomic_data']}")
        print(f"  local_atomic_numbers: {record['local_atomic_numbers']}")
        print(f"  halo_atomic_numbers: {record['halo_atomic_numbers']}")
        print(f"  position_min: {record['position_min']}")
        print(f"  position_max: {record['position_max']}")
        print(f"  energy: {record['energy']}")
        print()


def maybe_gather_records(local_record: Dict[str, Any]) -> list[Dict[str, Any]]:
    if dist.is_initialized():
        gathered: list[Optional[Dict[str, Any]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_record)
        return [record for record in gathered if record is not None]
    return [local_record]


def finalize_distributed() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def main() -> None:
    try:
        args = build_parser().parse_args()
        raw_structure = parse_raw_structure(args)
        structure_diagnostics = analyze_structure(raw_structure)

        is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
        torch_dist_env = TorchDistEnv(is_distributed=is_distributed)
        device = torch_dist_env.device if args.device == "cuda" else torch.device(args.device)

        if not dist.is_initialized():
            raise RuntimeError(
                "Phase 1 raw-exchange adapter requires a distributed launch. "
                "Run it under torchrun or an equivalent SLURM multi-rank launch."
            )

        raw_model = load_raw_mace_model(
            model_family=args.model_family,
            model=args.model,
            device=str(device),
        )
        raw_model.to(device)
        raw_model.eval()

        global_mol_info = build_global_molecular_info(raw_structure, device)
        sim_config = build_sim_config(
            raw_structure,
            cutoff=float(raw_model.r_max.detach().cpu().item()),
            ghost_cutoff_buffer=args.ghost_cutoff_buffer,
            device=device,
        )

        if args.partitioner == "sfc":
            partitioner = SFCPartitioner(SFCCurveType.Morton)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported partitioner: {args.partitioner}")

        distributed_state = build_phase1_distnnp_state(
            global_molecular_info=global_mol_info,
            partitioner=partitioner,
            sim_config=sim_config,
            dist_env=torch_dist_env.dist_env,
            device=device,
        )

        adapter = MACEPotentialModelWithRawExchange(raw_model)
        adapter.initialize_buffers(
            distributed_state.local_partition_info,
            device,
            int(distributed_state.dist_env.world_size),
        )
        energy, forces = adapter(distributed_state)

        local_record = adapter.last_observation | {
            "returned_force_shape": list(forces.shape),
            "returned_energy": float(energy.detach().cpu().item()),
        }
        gathered_records = maybe_gather_records(local_record)

        if not dist.is_initialized() or dist.get_rank() == 0:
            summary = summarize_rank_results(
                raw_structure=raw_structure,
                structure_diagnostics=structure_diagnostics,
                observation_records=gathered_records,
            )
            print_summary(summary, args)

            if args.output_json:
                output_path = Path(args.output_json)
                output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print()
                print(f"Wrote JSON summary to {output_path}")
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
