"""Top-down single-file DistMACE adapter for DistNNP.

This adapter keeps DistNNP as the authority for partitioning and halo
construction while packaging one coherent MACE-facing execution path:

1. DistNNP partition + halo maps
2. local-plus-halo structural reconstruction
3. public MACE graph build over that structural view
4. exactly one learned-feature exchange after interaction/product layer 0
5. owned-local energy inside the subclassed MACE forward
6. owned-local force recovery through autograd
7. stage-wise validation and monolithic comparison
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    from mace.modules import ScaleShiftMACE
    from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
    from mace.tools import AtomicNumberTable
    from mace.tools.scatter import scatter_sum
    from mace.tools.torch_geometric.batch import Batch

    _MACE_MODEL_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None
    ScaleShiftMACE = None  # type: ignore[assignment]
    get_atomic_virials_stresses = None  # type: ignore[assignment]
    get_outputs = None  # type: ignore[assignment]
    prepare_graph = None  # type: ignore[assignment]
    scatter_sum = None  # type: ignore[assignment]
    _MACE_MODEL_IMPORT_ERROR = exc


def _require_mace_model_ops() -> None:
    if _MACE_MODEL_IMPORT_ERROR is not None:
        raise ImportError(
            "This adapter requires public MACE model symbols from 'mace.modules'."
        ) from _MACE_MODEL_IMPORT_ERROR


def load_raw_mace_model(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    _require_mace_model_ops()
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


@dataclass
class StructuralHaloView:
    local_positions: torch.Tensor
    local_atom_types: torch.Tensor
    exchanged_positions: torch.Tensor
    exchanged_atom_types: torch.Tensor
    exchanged_atomic_numbers: torch.Tensor
    local_global_indices: torch.Tensor
    halo_global_indices: torch.Tensor
    expanded_global_indices: torch.Tensor
    num_local: int
    num_total: int
    num_halo: int


@dataclass
class MonolithicReference:
    total_energy: float
    forces: torch.Tensor
    node_energy: torch.Tensor
    edge_count: int
    num_interactions: int
    global_edge_pairs: List[List[int]]


BUILTIN_SYSTEMS: Dict[str, Dict[str, str]] = {
    "adapter_boundary_halo": {
        "kind": "phase1_builtin",
        "description": "Existing small halo-producing boundary fixture.",
    },
    "adapter_xyz6": {
        "kind": "phase1_builtin",
        "description": "Existing small non-halo six-atom fixture.",
    },
    "adapter_natural_chain12": {
        "kind": "natural",
        "description": (
            "A 12-atom chain-like hydrocarbon fixture with realistic local spacing. "
            "Designed to create an interpretable cross-partition bond instead of a "
            "dense periodic stress cube."
        ),
    },
    "adapter_natural_stress_bundle48": {
        "kind": "natural_stress",
        "description": (
            "A 48-atom nonperiodic multi-chain hydrocarbon bundle with realistic "
            "local spacing and moderate cross-chain density. Designed to stress "
            "halo stability more naturally than the dense periodic cube."
        ),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Top-down single-file DistMACE adapter: DistNNP-driven structural halo, "
            "single post-layer-0 learned-feature exchange, owned-local energy, and "
            "autograd-based owned-local forces."
        )
    )
    parser.add_argument(
        "--builtin-system",
        choices=sorted(BUILTIN_SYSTEMS.keys()),
        default="adapter_natural_chain12",
        help="Built-in structure to use when --xyz-path is not provided.",
    )
    parser.add_argument("--xyz-path", help="Optional path to an XYZ/extxyz structure.")
    parser.add_argument(
        "--model-family",
        choices=["mp", "off", "omol", "polar"],
        default="mp",
        help="Public MACE foundation-model family to load.",
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help="Foundation-model key or checkpoint path understood by the loader.",
    )
    parser.add_argument(
        "--comparison-model",
        default="",
        help="Optional model override for monolithic parity checks.",
    )
    parser.add_argument("--device", default="cuda", help="Execution device.")
    parser.add_argument(
        "--partitioner",
        choices=["sfc"],
        default="sfc",
        help="DistNNP partitioner to use.",
    )
    parser.add_argument(
        "--ghost-cutoff-buffer",
        type=float,
        default=0.5,
        help="Ghost cutoff buffer passed to DistNNP halo construction.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save a JSON summary from rank 0.",
    )
    parser.add_argument(
        "--verbose-diagnostics",
        action="store_true",
        help="Print the full per-rank diagnostics instead of a compact summary.",
    )
    return parser


def build_natural_chain12_structure() -> Dict[str, Any]:
    atomic_numbers = [
        6,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    positions = [
        [0.00, 0.00, 0.00],
        [1.52, 0.05, 0.00],
        [3.04, -0.03, 0.02],
        [4.56, 0.04, -0.01],
        [-0.63, 0.93, 0.02],
        [-0.58, -0.91, -0.04],
        [1.49, 1.10, -0.02],
        [1.55, -1.01, 0.07],
        [3.02, 1.02, 0.05],
        [3.06, -1.09, -0.03],
        [5.13, 0.92, 0.02],
        [5.16, -0.89, -0.05],
    ]
    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0],
        ],
        "pbc": [False, False, False],
        "source_path": "builtin://adapter_natural_chain12",
    }


def build_natural_stress_bundle48_structure() -> Dict[str, Any]:
    atomic_numbers: List[int] = []
    positions: List[List[float]] = []
    chain_y_offsets = [-4.6, -1.5, 1.6, 4.7]
    backbone_spacing = 1.46
    hydrogen_offset_y = 1.02
    hydrogen_offset_z = 0.82
    for chain_index, y0 in enumerate(chain_y_offsets):
        z0 = -1.35 if chain_index % 2 == 0 else 1.35
        for i in range(6):
            x = round(i * backbone_spacing, 6)
            y = round(y0 + (0.12 if i % 2 else -0.12), 6)
            z = round(z0 + (0.08 if i % 2 else -0.08), 6)
            atomic_numbers.append(6)
            positions.append([x, y, z])

            h_y = y + (hydrogen_offset_y if (i + chain_index) % 2 == 0 else -hydrogen_offset_y)
            h_z = z + (hydrogen_offset_z if i % 2 == 0 else -hydrogen_offset_z)
            atomic_numbers.append(1)
            positions.append([round(x, 6), round(h_y, 6), round(h_z, 6)])

    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": [
            [30.0, 0.0, 0.0],
            [0.0, 24.0, 0.0],
            [0.0, 0.0, 24.0],
        ],
        "pbc": [False, False, False],
        "source_path": "builtin://adapter_natural_stress_bundle48",
    }


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
    if args.builtin_system == "adapter_natural_chain12":
        return build_natural_chain12_structure()
    if args.builtin_system == "adapter_natural_stress_bundle48":
        return build_natural_stress_bundle48_structure()
    if args.builtin_system == "adapter_xyz6":
        return parse_xyz_file(DEFAULT_ADAPTER_STRUCTURE_PATH)
    if args.builtin_system == "adapter_boundary_halo":
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


def _csr_counts(offsets: torch.Tensor) -> List[int]:
    offsets_long = offsets.to(torch.int64)
    if offsets_long.ndim != 1 or offsets_long.numel() < 2:
        raise ValueError(
            f"Expected a 1D CSR offsets tensor with at least 2 entries, got shape {list(offsets_long.shape)}"
        )
    return (
        offsets_long[1:] - offsets_long[:-1]
    ).detach().cpu().to(torch.int64).tolist()


def validate_partition_info(
    partition_info: PartitionMessagePassingInfo,
    world_size: int,
) -> Dict[str, Any]:
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
            "PartitionMessagePassingInfo is missing required fields: " + ", ".join(missing)
        )

    partition_indices = partition_info.partition_indices.to(torch.long)
    send_offsets = partition_info.send_offsets.to(torch.int64)
    recv_offsets = partition_info.recv_offsets.to(torch.int64)
    send_local_indices = partition_info.send_local_indices.to(torch.int64)
    recv_ghost_indices = partition_info.recv_ghost_indices.to(torch.int64)

    if send_offsets.numel() != world_size + 1:
        raise ValueError(
            f"send_offsets must have shape [world_size + 1], got {list(send_offsets.shape)}"
        )
    if recv_offsets.numel() != world_size + 1:
        raise ValueError(
            f"recv_offsets must have shape [world_size + 1], got {list(recv_offsets.shape)}"
        )
    if not torch.all(send_offsets[1:] >= send_offsets[:-1]):
        raise ValueError("send_offsets is not monotonic nondecreasing.")
    if not torch.all(recv_offsets[1:] >= recv_offsets[:-1]):
        raise ValueError("recv_offsets is not monotonic nondecreasing.")
    if int(send_offsets[0].item()) != 0 or int(recv_offsets[0].item()) != 0:
        raise ValueError("CSR offsets must start at zero.")

    total_send = int(send_offsets[-1].item())
    total_recv = int(recv_offsets[-1].item())
    if send_local_indices.numel() != total_send:
        raise ValueError(
            f"send_local_indices has {send_local_indices.numel()} entries but total_send={total_send}."
        )
    if recv_ghost_indices.numel() != total_recv:
        raise ValueError(
            f"recv_ghost_indices has {recv_ghost_indices.numel()} entries but total_recv={total_recv}."
        )

    n_local = int(partition_indices.numel())
    if total_send > 0:
        if int(send_local_indices.min().item()) < 0 or int(send_local_indices.max().item()) >= n_local:
            raise ValueError("send_local_indices contains entries outside the local atom range.")

    expected_ghost_indices = torch.arange(
        n_local,
        n_local + total_recv,
        dtype=torch.int64,
        device=recv_ghost_indices.device,
    )
    if total_recv > 0 and not torch.equal(recv_ghost_indices, expected_ghost_indices):
        raise ValueError("recv_ghost_indices does not enumerate contiguous appended ghost slots.")

    return {
        "n_local": n_local,
        "total_send": total_send,
        "total_recv": total_recv,
        "send_counts_by_peer": _csr_counts(send_offsets),
        "recv_counts_by_peer": _csr_counts(recv_offsets),
    }


def validate_state(distributed_state: DistNNPState) -> Dict[str, Any]:
    partition_validation = validate_partition_info(
        distributed_state.local_partition_info,
        int(distributed_state.dist_env.world_size),
    )
    local_info = distributed_state.local_molecular_info
    global_info = distributed_state.global_molecular_info
    local_indices = distributed_state.local_partition_info.partition_indices.to(torch.long)

    if int(local_info.num_atoms) != partition_validation["n_local"]:
        raise ValueError("local_molecular_info.num_atoms does not match partition_indices size.")

    expected_positions = global_info.atom_positions.index_select(0, local_indices)
    expected_atom_types = global_info.atom_types.index_select(0, local_indices)
    if local_info.atom_positions.shape != expected_positions.shape:
        raise ValueError("local_molecular_info.atom_positions shape mismatch.")
    if local_info.atom_types.shape != expected_atom_types.shape:
        raise ValueError("local_molecular_info.atom_types shape mismatch.")
    if not torch.equal(local_info.atom_positions, expected_positions):
        raise ValueError("local_molecular_info.atom_positions no longer matches global_molecular_info[partition_indices].")
    if not torch.equal(local_info.atom_types, expected_atom_types):
        raise ValueError("local_molecular_info.atom_types no longer matches global_molecular_info[partition_indices].")
    return partition_validation


def build_global_molecular_info(
    raw_structure: Dict[str, Any],
    device: torch.device,
) -> MolecularInfo:
    def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
        if hasattr(obj, name):
            setattr(obj, name, value)

    num_atoms = len(raw_structure["atomic_numbers"])
    atomic_numbers = torch.tensor(raw_structure["atomic_numbers"], dtype=torch.long, device=device)

    mol_info = MolecularInfo()
    mol_info.num_atoms = num_atoms
    mol_info.atom_positions = torch.tensor(raw_structure["positions"], dtype=torch.float32, device=device)
    mol_info.atom_types = atomic_numbers
    mol_info.atom_velocities = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)
    mol_info.atom_forces = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)
    _assign_if_present(
        mol_info,
        "atom_vertex_numbers",
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


def build_distnnp_state(
    global_molecular_info: MolecularInfo,
    partitioner,
    sim_config: SimulationConfig,
    dist_env: DistEnvironment,
    device: torch.device,
) -> DistNNPState:
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
    local_molecular_info.atom_positions = global_molecular_info.atom_positions.index_select(0, local_indices)
    local_molecular_info.atom_velocities = global_molecular_info.atom_velocities.index_select(0, local_indices)
    local_molecular_info.atom_types = global_molecular_info.atom_types.index_select(0, local_indices)
    local_molecular_info.atom_forces = torch.zeros(local_indices.numel(), 3, dtype=torch.float32, device=device)
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
    validate_state(state)
    return state


def maybe_gather_records(local_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    if dist.is_initialized():
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_record)
        return [record for record in gathered if record is not None]
    return [local_record]


def finalize_distributed() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()


def resolve_comparison_model(args: argparse.Namespace) -> str:
    return args.comparison_model.strip() or args.model


class DistScaleShiftMACERuntimeMixin:
    __distmace_top_down__ = True

    def _configure_dist_runtime(self) -> None:
        self._distmace_context: Dict[str, Any] = {
            "distributed_state": None,
            "exchange_mode": "uninitialized",
            "observation": None,
        }
        self._distmace_feature_exchanger = GPUDirectExchanger()

    def _begin_dist_forward(
        self,
        distributed_state: DistNNPState,
        exchange_mode: str,
        observation: Dict[str, Any],
    ) -> None:
        self._distmace_context["distributed_state"] = distributed_state
        self._distmace_context["exchange_mode"] = exchange_mode
        self._distmace_context["observation"] = observation

    def _end_dist_forward(self) -> None:
        self._distmace_context["distributed_state"] = None
        self._distmace_context["exchange_mode"] = "uninitialized"
        self._distmace_context["observation"] = None

    def _exchange_after_first_layer(
        self,
        node_feats: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        observation = self._distmace_context["observation"]
        if layer_index != 0:
            if observation is not None:
                observation.setdefault("layer_records", []).append(
                    {
                        "layer_index": layer_index,
                        "boundary": "post_product_layer0_only",
                        "executed": False,
                        "reason": "only_layer0_is_exchanged",
                        "node_feature_shape": list(node_feats.shape),
                    }
                )
            return node_feats

        if self._distmace_context["exchange_mode"] != "ipc":
            if observation is not None:
                observation.setdefault("layer_records", []).append(
                    {
                        "layer_index": layer_index,
                        "boundary": "post_product_layer0_only",
                        "executed": False,
                        "reason": f"exchange_mode={self._distmace_context['exchange_mode']}",
                        "node_feature_shape": list(node_feats.shape),
                    }
                )
            return node_feats

        distributed_state = self._distmace_context["distributed_state"]
        if distributed_state is None:
            raise RuntimeError(
                "Distributed MACE forward reached the layer-0 exchange without an active DistNNPState."
            )

        num_local = int(
            distributed_state.local_partition_info.partition_indices.shape[0]
        )
        expected_total_rows = int(
            num_local + distributed_state.local_partition_info.recv_offsets[-1].item()
        )
        input_dtype = node_feats.dtype
        local_node_feats = node_feats[:num_local].contiguous()
        exchanged_node_feats = FeaturePushToExchange.apply(
            local_node_feats,
            distributed_state,
            self._distmace_feature_exchanger,
        )
        if exchanged_node_feats.dtype != input_dtype:
            exchanged_node_feats = exchanged_node_feats.to(dtype=input_dtype)

        if observation is not None:
            observation["feature_exchange_used_ipc"] = True
            observation["feature_exchange_boundary"] = "post_product_layer0_only"
            observation["feature_exchange_layer"] = 0
            observation["executed_exchange_layers"] = [0]
            observation.setdefault("layer_records", []).append(
                {
                    "layer_index": 0,
                    "boundary": "post_product_layer0_only",
                    "executed": True,
                    "reason": "ipc",
                    "expected_total_rows": expected_total_rows,
                    "actual_rows_before_exchange": int(node_feats.shape[0]),
                    "actual_rows_after_exchange": int(exchanged_node_feats.shape[0]),
                    "row_count_matches_expected_total": int(exchanged_node_feats.shape[0])
                    == expected_total_rows,
                    "owned_local_feature_shape": list(local_node_feats.shape),
                    "exchanged_feature_shape": list(exchanged_node_feats.shape),
                    "input_feature_dtype": str(input_dtype),
                    "exchanged_feature_dtype": str(exchanged_node_feats.dtype),
                }
            )
        return exchanged_node_feats

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        _require_mace_model_ops()

        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        observation = self._distmace_context["observation"]
        if observation is not None:
            observation["num_interactions"] = int(
                self.num_interactions.detach().cpu().item()
            )
            observation["model_parameter_dtype"] = str(next(self.parameters()).dtype)
            observation["batch_positions_dtype"] = str(data["positions"].dtype)
            observation["graph_vectors_dtype"] = str(vectors.dtype)
            observation["graph_lengths_dtype"] = str(lengths.dtype)
            observation["node_embedding_dtype"] = str(node_feats.dtype)
            observation["edge_attr_dtype"] = str(edge_attrs.dtype)
            observation["edge_feat_dtype"] = str(edge_feats.dtype)

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )
            node_feats = self._exchange_after_first_layer(node_feats, i)
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()
        owned_node_energy = None
        owned_local_energy = None

        distributed_state = self._distmace_context["distributed_state"]
        if distributed_state is not None:
            num_local = int(
                distributed_state.local_partition_info.partition_indices.shape[0]
            )
            owned_node_energy = node_energy.reshape(-1)[:num_local]
            owned_local_energy = owned_node_energy.sum()
            if observation is not None:
                observation["ownership_boundary"] = "model_internal_node_energy"
                observation["ownership_rule"] = (
                    "owned_local_energy = node_energy[:num_local].sum() inside the "
                    "subclassed MACE forward."
                )
                observation["node_energy_shape"] = list(node_energy.shape)
                observation["owned_node_energy_shape"] = list(owned_node_energy.shape)
                observation["node_energy_dtype"] = str(node_energy.dtype)
                observation["total_energy_dtype"] = str(total_energy.dtype)
                observation["owned_local_energy_dtype"] = str(owned_local_energy.dtype)

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "owned_node_energy": owned_node_energy,
            "owned_local_energy": owned_local_energy,
        }


class DistMACEBaseAdapter(DistNNPModel):
    def __init__(
        self,
        model,
        atomic_number_map: Optional[Mapping[int, int]] = None,
    ) -> None:
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
        validation = validate_partition_info(partition_info, world_size)
        self.world_size = int(world_size)
        self.partition_validation = validation
        self.total_send = validation["total_send"]
        self.total_recv = validation["total_recv"]

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        halo_global_indices = self._derive_halo_global_indices(distributed_state)
        expected_total_recv = self.total_recv
        if halo_global_indices.numel() != expected_total_recv:
            raise ValueError(
                "Graph-derived halo reconstruction did not match the halo CSR metadata: "
                f"derived {int(halo_global_indices.numel())} ghost atoms but recv_offsets encodes {expected_total_recv}."
            )
        if halo_global_indices.numel() == 0:
            return local_positions, local_atom_types.to(torch.long)

        global_info = distributed_state.global_molecular_info
        halo_positions = global_info.atom_positions.index_select(0, halo_global_indices).detach()
        halo_atom_types = global_info.atom_types.index_select(0, halo_global_indices).to(torch.long)
        expanded_positions = torch.cat([local_positions, halo_positions], dim=0)
        expanded_atom_types = torch.cat([local_atom_types.to(torch.long), halo_atom_types], dim=0)
        return expanded_positions, expanded_atom_types

    def _expand_local_plus_halo_view(
        self,
        local_positions: torch.Tensor,
        local_atom_types: torch.Tensor,
        distributed_state: DistNNPState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def _pbc_to_tuple(pbc: torch.Tensor) -> Tuple[bool, bool, bool]:
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
            config_type="DistMACE_TopDown",
            head="Default",
        )


class DistMACETopDownAdapter(DistMACEBaseAdapter):
    adapter_target = (
        "distnnp_structural_halo + public_mace_graph_bridge + "
        "post_layer0_feature_exchange + owned_local_energy + autograd_forces"
    )
    adapter_status = "top_down_single_file_prototype"

    def __init__(self, model, atomic_number_map: Optional[Dict[int, int]] = None) -> None:
        super().__init__(model, atomic_number_map=atomic_number_map)
        self._install_model_subclass()

    def _install_model_subclass(self) -> None:
        _require_mace_model_ops()
        if ScaleShiftMACE is None or not isinstance(self.model, ScaleShiftMACE):
            raise TypeError(
                "This adapter currently targets public ScaleShiftMACE instances returned "
                "by the MACE foundation-model loaders."
            )
        if not getattr(self.model.__class__, "__distmace_top_down__", False):
            subclass_name = f"DistMACE{self.model.__class__.__name__}"
            subclass_cls = type(
                subclass_name,
                (DistScaleShiftMACERuntimeMixin, self.model.__class__),
                {
                    "__distmace_top_down__": True,
                    "__distmace_subclass_role__": "top_down_single_file_model_loop",
                },
            )
            self.model.__class__ = subclass_cls
        self.model._configure_dist_runtime()

    def _build_structural_halo_view(
        self,
        distributed_state: DistNNPState,
    ) -> StructuralHaloView:
        validation = validate_state(distributed_state)
        self.partition_validation = validation
        self.total_send = validation["total_send"]
        self.total_recv = validation["total_recv"]

        local_positions = (
            distributed_state.local_molecular_info.atom_positions.clone()
            .detach()
            .requires_grad_(True)
        )
        local_atom_types = distributed_state.local_molecular_info.atom_types
        exchanged_positions, exchanged_atom_types = self._expand_local_plus_halo_view(
            local_positions,
            local_atom_types,
            distributed_state,
        )
        exchanged_positions.retain_grad()
        exchanged_atomic_numbers = self._map_atomic_numbers(exchanged_atom_types)
        local_global_indices = (
            distributed_state.local_partition_info.partition_indices.to(torch.long).detach()
        )
        halo_global_indices = self._derive_halo_global_indices(distributed_state).to(
            torch.long
        )
        expanded_global_indices = torch.cat(
            [local_global_indices, halo_global_indices], dim=0
        )

        num_local = int(local_positions.shape[0])
        num_total = int(exchanged_positions.shape[0])
        expected_total = int(num_local + self.total_recv)
        if num_total != expected_total:
            raise ValueError(
                "Expanded structural view row count does not match DistNNP halo metadata: "
                f"{num_total} vs expected {expected_total}."
            )
        if int(halo_global_indices.numel()) != self.total_recv:
            raise ValueError(
                "Derived halo global indices do not match DistNNP recv metadata: "
                f"{int(halo_global_indices.numel())} vs expected {self.total_recv}."
            )

        return StructuralHaloView(
            local_positions=local_positions,
            local_atom_types=local_atom_types,
            exchanged_positions=exchanged_positions,
            exchanged_atom_types=exchanged_atom_types,
            exchanged_atomic_numbers=exchanged_atomic_numbers,
            local_global_indices=local_global_indices,
            halo_global_indices=halo_global_indices,
            expanded_global_indices=expanded_global_indices,
            num_local=num_local,
            num_total=num_total,
            num_halo=int(num_total - num_local),
        )

    def _build_graph_input(
        self,
        structural_view: StructuralHaloView,
        distributed_state: DistNNPState,
    ) -> tuple[Dict[str, torch.Tensor], int, torch.Tensor]:
        config = self._build_configuration(
            structural_view.exchanged_atomic_numbers,
            structural_view.exchanged_positions,
            distributed_state,
        )
        atomic_data = AtomicData.from_config(
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.heads,
        )
        edge_count = int(atomic_data.edge_index.shape[1])
        batch = Batch.from_data_list([atomic_data]).to(
            structural_view.exchanged_positions.device
        )
        batch_dict = batch.to_dict()
        batch_dict["positions"] = structural_view.exchanged_positions
        batch_dict = align_batch_dtype(batch_dict, self.model)
        return batch_dict, edge_count, atomic_data.edge_index.detach().clone()

    def _recover_owned_local_forces(
        self,
        owned_local_energy: torch.Tensor,
        structural_view: StructuralHaloView,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        local_position_grads, exchanged_position_grads = torch.autograd.grad(
            owned_local_energy,
            [structural_view.local_positions, structural_view.exchanged_positions],
            retain_graph=False,
            allow_unused=False,
        )
        owned_local_forces = -local_position_grads
        halo_position_grads = exchanged_position_grads[structural_view.num_local :]
        local_grad_is_finite = bool(torch.isfinite(local_position_grads).all().item())
        exchanged_grad_is_finite = bool(
            torch.isfinite(exchanged_position_grads).all().item()
        )
        halo_grad_norm = (
            float(halo_position_grads.norm().detach().cpu().item())
            if structural_view.num_halo > 0
            else 0.0
        )
        local_grad_norm = float(local_position_grads.norm().detach().cpu().item())
        local_force_norm = float(owned_local_forces.norm().detach().cpu().item())
        return owned_local_forces, {
            "recovered_force_shape": list(owned_local_forces.shape),
            "local_grad_is_finite": local_grad_is_finite,
            "exchanged_grad_is_finite": exchanged_grad_is_finite,
            "local_grad_norm": local_grad_norm,
            "halo_grad_norm": halo_grad_norm,
            "local_force_norm": local_force_norm,
            "halo_gradient_active": bool(halo_grad_norm > 0.0)
            if structural_view.num_halo > 0
            else False,
        }

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        self.last_observation = {
            "feature_exchange_used_ipc": False,
            "feature_exchange_boundary": "post_product_layer0_only",
            "feature_exchange_layer": 0,
            "executed_exchange_layers": [],
            "layer_records": [],
        }
        self.model._begin_dist_forward(
            distributed_state=distributed_state,
            exchange_mode=self.exchange_mode,
            observation=self.last_observation,
        )
        try:
            structural_view = self._build_structural_halo_view(distributed_state)
            batch_dict, edge_count, edge_index = self._build_graph_input(
                structural_view,
                distributed_state,
            )
            outputs = self.model(
                batch_dict,
                training=False,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
            owned_local_energy = outputs.get("owned_local_energy")
            if owned_local_energy is None:
                raise KeyError(
                    "The subclassed MACE forward did not return 'owned_local_energy'."
                )
            owned_local_forces, force_diagnostics = self._recover_owned_local_forces(
                owned_local_energy,
                structural_view,
            )
        finally:
            self.model._end_dist_forward()

        self.last_observation.update(
            {
                "rank": int(distributed_state.dist_env.global_rank),
                "adapter_target": self.adapter_target,
                "adapter_status": self.adapter_status,
                "n_local": structural_view.num_local,
                "n_total_after_exchange": structural_view.num_total,
                "n_halo": structural_view.num_halo,
                "expected_total_recv": self.total_recv,
                "exchange_mode": self.exchange_mode,
                "exchange_reason": self.exchange_reason,
                "total_send": self.total_send,
                "total_recv": self.total_recv,
                "send_counts_by_peer": self.partition_validation.get(
                    "send_counts_by_peer", []
                ),
                "recv_counts_by_peer": self.partition_validation.get(
                    "recv_counts_by_peer", []
                ),
                "edge_count_after_atomic_data": edge_count,
                "local_atomic_numbers": structural_view.exchanged_atomic_numbers[
                    : structural_view.num_local
                ]
                .detach()
                .cpu()
                .tolist(),
                "halo_atomic_numbers": structural_view.exchanged_atomic_numbers[
                    structural_view.num_local :
                ]
                .detach()
                .cpu()
                .tolist(),
                "local_global_indices": structural_view.local_global_indices.detach()
                .cpu()
                .tolist(),
                "halo_global_indices": structural_view.halo_global_indices.detach()
                .cpu()
                .tolist(),
                "owned_local_energy": float(owned_local_energy.detach().cpu().item()),
                "expanded_rank_graph_energy": float(
                    outputs["energy"].reshape(()).detach().cpu().item()
                )
                if torch.is_tensor(outputs["energy"]) and outputs["energy"].numel() == 1
                else float(outputs["energy"].detach().cpu().item()),
                "owned_node_energy_shape": self.last_observation.get(
                    "owned_node_energy_shape"
                ),
                "node_energy_shape": self.last_observation.get("node_energy_shape"),
                "owned_node_energy_values": outputs["owned_node_energy"].reshape(-1)
                .detach()
                .cpu()
                .tolist()
                if outputs.get("owned_node_energy") is not None
                else [],
                "global_edge_pairs": edge_index_to_global_pairs(
                    edge_index,
                    structural_view.expanded_global_indices.to(edge_index.device),
                ),
                "owned_destination_global_edge_pairs": filter_owned_destination_pairs(
                    edge_index_to_global_pairs(
                        edge_index,
                        structural_view.expanded_global_indices.to(edge_index.device),
                    ),
                    structural_view.local_global_indices.detach().cpu().tolist(),
                ),
                "owned_local_force_values": owned_local_forces.detach().cpu().tolist(),
                **force_diagnostics,
            }
        )
        return owned_local_energy.detach(), owned_local_forces.detach()


def build_adapter_distnnp_state(
    raw_structure: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
):
    global_mol_info = build_global_molecular_info(raw_structure, device)
    sim_config = build_sim_config(
        raw_structure=raw_structure,
        cutoff=float(args._raw_model.r_max.detach().cpu().item()),
        ghost_cutoff_buffer=args.ghost_cutoff_buffer,
        device=device,
    )
    if args.partitioner != "sfc":
        raise ValueError(f"Unsupported partitioner: {args.partitioner}")
    partitioner = SFCPartitioner(SFCCurveType.Morton)
    distributed_state = build_distnnp_state(
        global_molecular_info=global_mol_info,
        partitioner=partitioner,
        sim_config=sim_config,
        dist_env=args._torch_dist_env.dist_env,
        device=device,
    )
    return distributed_state, sim_config


def build_monolithic_reference(
    raw_structure: Dict[str, Any],
    sim_config,
    args: argparse.Namespace,
    device: torch.device,
) -> MonolithicReference:
    reference_model = load_raw_mace_model(
        model_family=args.model_family,
        model=resolve_comparison_model(args),
        device=str(device),
    )
    reference_model.to(device)
    reference_model.eval()

    positions = torch.tensor(
        raw_structure["positions"],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor(
        raw_structure["atomic_numbers"],
        dtype=torch.long,
        device=device,
    )
    config = Configuration(
        atomic_numbers=atomic_numbers.detach().cpu().numpy(),
        positions=positions.detach().cpu().to(torch.float64).numpy(),
        properties={},
        property_weights={},
        cell=DistMACEBaseAdapter._cell_to_numpy(sim_config.cell),
        pbc=DistMACEBaseAdapter._pbc_to_tuple(sim_config.pbc),
        weight=1.0,
        config_type="DistMACE_TopDown_Monolithic",
        head="Default",
    )
    z_table = AtomicNumberTable(
        [int(z) for z in reference_model.atomic_numbers.detach().cpu().tolist()]
    )
    heads = getattr(reference_model, "heads", ["Default"])
    cutoff = float(reference_model.r_max.detach().cpu().item())
    atomic_data = AtomicData.from_config(
        config=config,
        z_table=z_table,
        cutoff=cutoff,
        heads=heads,
    )
    batch = Batch.from_data_list([atomic_data]).to(device)
    batch_dict = batch.to_dict()
    batch_dict["positions"] = positions
    batch_dict = align_batch_dtype(batch_dict, reference_model)
    outputs = reference_model(
        batch_dict,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    total_energy = outputs["energy"]
    if torch.is_tensor(total_energy) and total_energy.numel() == 1:
        total_energy = total_energy.reshape(())
    return MonolithicReference(
        total_energy=float(total_energy.detach().cpu().item()),
        forces=outputs["forces"].detach().cpu(),
        node_energy=outputs["node_energy"].reshape(-1).detach().cpu(),
        edge_count=int(atomic_data.edge_index.shape[1]),
        num_interactions=int(reference_model.num_interactions.detach().cpu().item()),
        global_edge_pairs=edge_index_to_global_pairs(
            atomic_data.edge_index.detach(),
            torch.arange(len(raw_structure["atomic_numbers"]), device=atomic_data.edge_index.device),
        ),
    )


def edge_index_to_global_pairs(
    edge_index: torch.Tensor,
    expanded_global_indices: torch.Tensor,
) -> List[List[int]]:
    src = expanded_global_indices.index_select(0, edge_index[0].to(torch.long))
    dst = expanded_global_indices.index_select(0, edge_index[1].to(torch.long))
    pairs = torch.stack([src, dst], dim=1).detach().cpu().tolist()
    return sorted([list(map(int, pair)) for pair in pairs])


def filter_owned_destination_pairs(
    edge_pairs: List[List[int]],
    owned_global_indices: List[int],
) -> List[List[int]]:
    owned_set = {int(index) for index in owned_global_indices}
    return [pair for pair in edge_pairs if int(pair[1]) in owned_set]


def compare_edge_pair_sets(
    lhs: List[List[int]],
    rhs: List[List[int]],
) -> Dict[str, Any]:
    lhs_set = {tuple(pair) for pair in lhs}
    rhs_set = {tuple(pair) for pair in rhs}
    lhs_only = sorted([list(pair) for pair in lhs_set - rhs_set])
    rhs_only = sorted([list(pair) for pair in rhs_set - lhs_set])
    return {
        "exact_match": lhs_set == rhs_set,
        "lhs_count": len(lhs_set),
        "rhs_count": len(rhs_set),
        "lhs_only_count": len(lhs_only),
        "rhs_only_count": len(rhs_only),
        "lhs_only_preview": lhs_only[:12],
        "rhs_only_preview": rhs_only[:12],
    }


def build_structural_halo_sufficiency_records(
    rank_records: List[Dict[str, Any]],
    monolithic: MonolithicReference,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    monolithic_edge_pairs = monolithic.global_edge_pairs
    for record in rank_records:
        owned_indices = record["local_global_indices"]
        actual_halo = sorted(int(idx) for idx in record["halo_global_indices"])
        owned_set = {int(idx) for idx in owned_indices}
        required_remote = sorted(
            {
                int(src)
                for src, dst in monolithic_edge_pairs
                if int(dst) in owned_set and int(src) not in owned_set
            }
        )
        required_remote_set = set(required_remote)
        actual_halo_set = set(actual_halo)
        missing = sorted(list(required_remote_set - actual_halo_set))
        extra = sorted(list(actual_halo_set - required_remote_set))
        records.append(
            {
                "rank": record["rank"],
                "required_remote_atom_count_from_monolithic": len(required_remote),
                "actual_halo_atom_count": len(actual_halo),
                "halo_contains_all_required_remote_atoms": len(missing) == 0,
                "required_remote_atoms_missing_from_halo": missing,
                "extra_halo_atoms_not_required_by_monolithic": extra,
            }
        )
    return records


def build_partition_verification(
    rank_records: List[Dict[str, Any]],
    num_atoms: int,
) -> Dict[str, Any]:
    owned_indices = [idx for record in rank_records for idx in record["local_global_indices"]]
    owned_tensor = torch.tensor(owned_indices, dtype=torch.long)
    expected = torch.arange(num_atoms, dtype=torch.long)
    sorted_owned = owned_tensor.sort()[0] if owned_tensor.numel() > 0 else owned_tensor

    send_matrix = torch.tensor(
        [record["send_counts_by_peer"] for record in rank_records],
        dtype=torch.long,
    )
    recv_matrix = torch.tensor(
        [record["recv_counts_by_peer"] for record in rank_records],
        dtype=torch.long,
    )
    peer_symmetry = bool(torch.equal(send_matrix, recv_matrix.T))

    duplicate_count = int(owned_tensor.numel() - torch.unique(owned_tensor).numel())
    coverage_ok = bool(
        owned_tensor.numel() == num_atoms and torch.equal(sorted_owned, expected)
    )
    return {
        "all_atoms_owned_exactly_once": coverage_ok and duplicate_count == 0,
        "owned_index_count": int(owned_tensor.numel()),
        "unique_owned_index_count": int(torch.unique(owned_tensor).numel())
        if owned_tensor.numel() > 0
        else 0,
        "duplicate_owned_index_count": duplicate_count,
        "min_owned_index": int(owned_tensor.min().item()) if owned_tensor.numel() > 0 else None,
        "max_owned_index": int(owned_tensor.max().item()) if owned_tensor.numel() > 0 else None,
        "send_recv_peer_symmetry": peer_symmetry,
        "send_matrix": send_matrix.tolist(),
        "recv_matrix": recv_matrix.tolist(),
    }


def build_rank_parity_records(
    rank_records: List[Dict[str, Any]],
    monolithic: MonolithicReference,
    structural_halo_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    parity_records: List[Dict[str, Any]] = []
    monolithic_node_energy = monolithic.node_energy
    structural_by_rank = {record["rank"]: record for record in structural_halo_records}
    for record in rank_records:
        owned_indices = record["local_global_indices"]
        owned_node_energy_values = torch.tensor(
            record.get("owned_node_energy_values", []),
            dtype=torch.float64,
        )
        monolithic_owned_node_energy = monolithic_node_energy[owned_indices].to(torch.float64)
        node_energy_delta = owned_node_energy_values - monolithic_owned_node_energy

        monolithic_owned_edges = filter_owned_destination_pairs(
            monolithic.global_edge_pairs,
            owned_indices,
        )
        edge_parity = compare_edge_pair_sets(
            record.get("owned_destination_global_edge_pairs", []),
            monolithic_owned_edges,
        )
        structural_record = structural_by_rank[record["rank"]]
        mismatch_cause = "parity_ok"
        if not structural_record["halo_contains_all_required_remote_atoms"]:
            mismatch_cause = "halo_selection_or_halo_derivation"
        elif not edge_parity["exact_match"]:
            mismatch_cause = "graph_builder_disagreement_after_halo"
        parity_records.append(
            {
                "rank": record["rank"],
                "owned_node_energy_mae": float(node_energy_delta.abs().mean().item())
                if node_energy_delta.numel() > 0
                else 0.0,
                "owned_node_energy_max_abs": float(node_energy_delta.abs().max().item())
                if node_energy_delta.numel() > 0
                else 0.0,
                "owned_energy_sum_delta": float(node_energy_delta.sum().item()),
                "parity_gate_category": {
                    "structural_halo": structural_record["halo_contains_all_required_remote_atoms"],
                    "owned_destination_edge_set": edge_parity["exact_match"],
                    "owned_node_energy": bool(torch.allclose(
                        owned_node_energy_values,
                        monolithic_owned_node_energy,
                        atol=1e-6,
                        rtol=1e-6,
                    )) if node_energy_delta.numel() > 0 else True,
                },
                "primary_mismatch_cause": mismatch_cause,
                "owned_destination_edge_parity": edge_parity,
            }
        )
    return parity_records


def assemble_global_owned_forces(rank_records: List[Dict[str, Any]], num_atoms: int) -> torch.Tensor:
    full_forces = torch.zeros(num_atoms, 3, dtype=torch.float32)
    for record in rank_records:
        partition_indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        owned_forces = torch.tensor(record["owned_local_force_values"], dtype=torch.float32)
        full_forces.index_copy_(0, partition_indices, owned_forces)
    return full_forces


def build_summary(
    raw_structure: Dict[str, Any],
    diagnostics: Dict[str, Any],
    rank_records: List[Dict[str, Any]],
    monolithic: Optional[MonolithicReference],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(rank_records, key=lambda item: item["rank"])
    partition_verification = build_partition_verification(
        rank_records_sorted,
        len(raw_structure["atomic_numbers"]),
    )
    structural_halo_records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": diagnostics,
        "rank_records": rank_records_sorted,
        "partition_verification": partition_verification,
        "checks": {
            "all_expected_halo_counts_match": all(
                record["n_halo"] == record["expected_total_recv"]
                for record in rank_records_sorted
            ),
            "all_recovered_force_shapes_match_owned_atoms": all(
                record["recovered_force_shape"] == [record["n_local"], 3]
                for record in rank_records_sorted
            ),
            "all_local_grads_finite": all(
                record["local_grad_is_finite"] for record in rank_records_sorted
            ),
            "all_exchanged_grads_finite": all(
                record["exchanged_grad_is_finite"] for record in rank_records_sorted
            ),
            "layer0_exchange_executed": any(
                record.get("feature_exchange_used_ipc", False)
                for record in rank_records_sorted
            ),
            "summed_owned_local_energy": float(
                sum(record["owned_local_energy"] for record in rank_records_sorted)
            ),
            "all_atoms_owned_exactly_once": partition_verification[
                "all_atoms_owned_exactly_once"
            ],
            "send_recv_peer_symmetry": partition_verification["send_recv_peer_symmetry"],
        },
    }
    if monolithic is not None:
        structural_halo_records = build_structural_halo_sufficiency_records(
            rank_records_sorted,
            monolithic,
        )
        assembled_forces = assemble_global_owned_forces(
            rank_records_sorted,
            len(raw_structure["atomic_numbers"]),
        )
        force_delta = assembled_forces - monolithic.forces
        rank_parity_records = build_rank_parity_records(
            rank_records_sorted,
            monolithic,
            structural_halo_records,
        )
        summary["monolithic_reference"] = {
            "total_energy": monolithic.total_energy,
            "edge_count": monolithic.edge_count,
            "num_interactions": monolithic.num_interactions,
            "structural_halo_verification": structural_halo_records,
            "distributed_minus_monolithic_energy": (
                summary["checks"]["summed_owned_local_energy"] - monolithic.total_energy
            ),
            "assembled_force_mae": float(force_delta.abs().mean().item()),
            "assembled_force_max_abs": float(force_delta.abs().max().item()),
            "assembled_force_l2": float(force_delta.norm().item()),
            "rank_parity_records": rank_parity_records,
            "all_owned_destination_edge_sets_match": all(
                item["owned_destination_edge_parity"]["exact_match"]
                for item in rank_parity_records
            ),
            "all_structural_halos_satisfy_monolithic_owned_destinations": all(
                item["halo_contains_all_required_remote_atoms"]
                for item in structural_halo_records
            ),
            "max_owned_node_energy_abs_delta": max(
                item["owned_node_energy_max_abs"] for item in rank_parity_records
            )
            if rank_parity_records
            else 0.0,
        }
    return summary


def print_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("Top-Down Single-File DistMACE Adapter")
    print("============================================================")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"comparison_model: {resolve_comparison_model(args)}")
    print(f"device: {args.device}")
    print(f"builtin_system: {args.builtin_system if not args.xyz_path else 'custom_xyz'}")
    print()
    print("------------------------------------------------------------")
    print("Global Input")
    print("------------------------------------------------------------")
    print(f"source_path: {summary['source_path']}")
    print(f"num_atoms: {summary['num_atoms']}")
    print(f"atomic_numbers: {summary['atomic_numbers']}")
    print()
    print("------------------------------------------------------------")
    print("Checks")
    print("------------------------------------------------------------")
    checks = summary["checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(
        "all_recovered_force_shapes_match_owned_atoms: "
        f"{checks['all_recovered_force_shapes_match_owned_atoms']}"
    )
    print(f"all_local_grads_finite: {checks['all_local_grads_finite']}")
    print(f"all_exchanged_grads_finite: {checks['all_exchanged_grads_finite']}")
    print(f"layer0_exchange_executed: {checks['layer0_exchange_executed']}")
    print(f"summed_owned_local_energy: {checks['summed_owned_local_energy']}")
    print(f"all_atoms_owned_exactly_once: {checks['all_atoms_owned_exactly_once']}")
    print(f"send_recv_peer_symmetry: {checks['send_recv_peer_symmetry']}")
    if "monolithic_reference" in summary:
        mono = summary["monolithic_reference"]
        print(f"monolithic_total_energy: {mono['total_energy']}")
        print(
            "all_structural_halos_satisfy_monolithic_owned_destinations: "
            f"{mono['all_structural_halos_satisfy_monolithic_owned_destinations']}"
        )
        print(
            "distributed_minus_monolithic_energy: "
            f"{mono['distributed_minus_monolithic_energy']}"
        )
        print(f"assembled_force_mae: {mono['assembled_force_mae']}")
        print(f"assembled_force_max_abs: {mono['assembled_force_max_abs']}")
        print(f"assembled_force_l2: {mono['assembled_force_l2']}")
        print(f"monolithic_edge_count: {mono['edge_count']}")
        print(f"monolithic_num_interactions: {mono['num_interactions']}")
        print(
            "all_owned_destination_edge_sets_match: "
            f"{mono['all_owned_destination_edge_sets_match']}"
        )
        print(
            "max_owned_node_energy_abs_delta: "
            f"{mono['max_owned_node_energy_abs_delta']}"
        )
    print()
    print("------------------------------------------------------------")
    print("Per-Rank Summary")
    print("------------------------------------------------------------")
    for record in summary["rank_records"]:
        print(f"rank: {record['rank']}")
        print(f"  n_local: {record['n_local']}")
        print(f"  n_total_after_exchange: {record['n_total_after_exchange']}")
        print(f"  n_halo: {record['n_halo']}")
        print(f"  exchange_mode: {record['exchange_mode']}")
        print(f"  feature_exchange_used_ipc: {record.get('feature_exchange_used_ipc')}")
        print(f"  edge_count_after_atomic_data: {record['edge_count_after_atomic_data']}")
        print(
            "  owned_destination_edge_count: "
            f"{len(record.get('owned_destination_global_edge_pairs', []))}"
        )
        print(f"  owned_local_energy: {record['owned_local_energy']}")
        if args.verbose_diagnostics:
            print(f"  send_counts_by_peer: {record['send_counts_by_peer']}")
            print(f"  recv_counts_by_peer: {record['recv_counts_by_peer']}")
            print(f"  local_global_indices: {record['local_global_indices']}")
            print(f"  halo_global_indices: {record['halo_global_indices']}")
            print(f"  owned_node_energy_values: {record.get('owned_node_energy_values', [])}")
            print(f"  layer_records: {record.get('layer_records', [])}")
            print(
                "  owned_destination_global_edge_pairs: "
                f"{record.get('owned_destination_global_edge_pairs', [])}"
            )
        print()

    if "monolithic_reference" in summary and args.verbose_diagnostics:
        print("------------------------------------------------------------")
        print("Structural Halo Sufficiency")
        print("------------------------------------------------------------")
        for item in summary["monolithic_reference"]["structural_halo_verification"]:
            print(f"rank: {item['rank']}")
            print(
                "  required_remote_atom_count_from_monolithic: "
                f"{item['required_remote_atom_count_from_monolithic']}"
            )
            print(f"  actual_halo_atom_count: {item['actual_halo_atom_count']}")
            print(
                "  halo_contains_all_required_remote_atoms: "
                f"{item['halo_contains_all_required_remote_atoms']}"
            )
            print(
                "  required_remote_atoms_missing_from_halo: "
                f"{item['required_remote_atoms_missing_from_halo']}"
            )
            print(
                "  extra_halo_atoms_not_required_by_monolithic: "
                f"{item['extra_halo_atoms_not_required_by_monolithic']}"
            )
            print()

        print("------------------------------------------------------------")
        print("Detailed Parity")
        print("------------------------------------------------------------")
        for item in summary["monolithic_reference"]["rank_parity_records"]:
            print(f"rank: {item['rank']}")
            print(f"  owned_node_energy_mae: {item['owned_node_energy_mae']}")
            print(f"  owned_node_energy_max_abs: {item['owned_node_energy_max_abs']}")
            print(f"  owned_energy_sum_delta: {item['owned_energy_sum_delta']}")
            print(f"  parity_gate_category: {item['parity_gate_category']}")
            print(f"  primary_mismatch_cause: {item['primary_mismatch_cause']}")
            print(
                "  owned_destination_edge_parity: "
                f"{item['owned_destination_edge_parity']}"
            )
            print()


def main() -> None:
    try:
        args = build_parser().parse_args()
        raw_structure = parse_raw_structure(args)
        structure_diagnostics = analyze_structure(raw_structure)

        is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
        torch_dist_env = TorchDistEnv(is_distributed=is_distributed)
        device = (
            torch_dist_env.device if args.device == "cuda" else torch.device(args.device)
        )
        args._torch_dist_env = torch_dist_env

        if not dist.is_initialized():
            raise RuntimeError(
                "This adapter requires a distributed launch. Run it under torchrun or "
                "an equivalent SLURM multi-rank launch."
            )

        raw_model = load_raw_mace_model(
            model_family=args.model_family,
            model=args.model,
            device=str(device),
        )
        raw_model.to(device)
        raw_model.eval()
        args._raw_model = raw_model

        distributed_state, sim_config = build_adapter_distnnp_state(
            raw_structure=raw_structure,
            args=args,
            device=device,
        )

        adapter = DistMACETopDownAdapter(raw_model)
        adapter.initialize_buffers(
            distributed_state.local_partition_info,
            device,
            int(distributed_state.dist_env.world_size),
        )
        owned_local_energy, owned_local_forces = adapter(distributed_state)

        local_record = adapter.last_observation | {
            "returned_force_shape": list(owned_local_forces.shape),
            "returned_energy": float(owned_local_energy.detach().cpu().item()),
        }
        gathered_records = maybe_gather_records(local_record)

        if not dist.is_initialized() or dist.get_rank() == 0:
            monolithic = build_monolithic_reference(
                raw_structure=raw_structure,
                sim_config=sim_config,
                args=args,
                device=device,
            )
            summary = build_summary(
                raw_structure=raw_structure,
                diagnostics=structure_diagnostics,
                rank_records=gathered_records,
                monolithic=monolithic,
            )
            print_summary(summary, args)
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
