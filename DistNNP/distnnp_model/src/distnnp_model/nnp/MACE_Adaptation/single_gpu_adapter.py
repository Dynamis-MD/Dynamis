"""Single-GPU MACE adapter for the DistNNP model contract.

This adapter is intentionally simple:

- no halo exchange
- no repartition-aware buffers
- no use of ASE

It exists to verify two things:
1. DistNNPState can be converted into the public MACE graph input path.
2. A loaded MACE model can satisfy the DistNNPModel energy/forces contract.

Also note that the force computation comes from within MACE and not DistNNP

Step 1 baseline role
--------------------
Treat this file as the frozen single-rank baseline while the first distributed
MACE prototype is developed elsewhere. It should remain the simplest path for
checking the DistNNPState -> public MACE input -> DistNNPModel contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from distnnp_model.potential import DistNNPModel
from distnnp_state import (
    DistEnvironment,
    DistNNPState,
    MolecularInfo,
    PartitionMessagePassingInfo,
    SimulationConfig,
)

try:
    from mace.calculators import (
        MACECalculator,
        mace_mp,
        mace_off,
        mace_omol,
        mace_polar,
    )
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional runtime install
    MACECalculator = mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None
    _MACE_IMPORT_ERROR = exc

try:
    from ase import Atoms

    _ASE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional runtime install
    Atoms = None
    _ASE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "MACE imports failed. Install the public 'mace-torch' package in the "
            "runtime environment before using MACEPotentialModel."
        ) from _MACE_IMPORT_ERROR


def _require_ase() -> None:
    if _ASE_IMPORT_ERROR is not None:
        raise ImportError(
            "ASE imports failed. Install 'ase' in the runtime environment before "
            "using the standalone MACE calculator comparison path."
        ) from _ASE_IMPORT_ERROR


def load_raw_mace_model(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    """Load a raw public MACE model object.

    This follows the same public loading path as the direct standalone probe by
    requesting ``return_raw_model=True`` from the public calculators API.
    """
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
    """Cast floating tensors in the batch to the model parameter dtype."""
    model_dtype = next(model.parameters()).dtype
    aligned: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            aligned[key] = value.to(dtype=model_dtype)
        else:
            aligned[key] = value
    return aligned


def load_mace_calculator(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    """Load a public MACE calculator object for the standalone comparison path."""
    _require_mace()
    _require_ase()

    common = {"model": model, "device": device}
    if model_family == "mp":
        return mace_mp(dispersion=False, **common)
    if model_family == "off":
        return mace_off(**common)
    if model_family == "omol":
        return mace_omol(**common)
    if model_family == "polar":
        return mace_polar(**common)
    raise ValueError(f"Unsupported MACE model family: {model_family}")


BUILTIN_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "adapter_xyz6": {
        "kind": "file",
        "description": "6-atom non-overlapping xyz fixture stored inside the adaptation folder.",
    },
    "distnnp_xyz6": {
        "kind": "file",
        "description": "6-atom DistNNP xyz fixture with lattice and PBC metadata.",
    },
    "h2o": {
        "atomic_numbers": [8, 1, 1],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.758602, 0.0, 0.504284],
            [-0.758602, 0.0, 0.504284],
        ],
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False, False, False],
    },
    "si_dimer": {
        "atomic_numbers": [14, 14],
        "positions": [
            [0.0, 0.0, 0.0],
            [2.35, 0.0, 0.0],
        ],
        "cell": np.zeros((3, 3)).tolist(),
        "pbc": [False, False, False],
    },
}

DEFAULT_ADAPTER_STRUCTURE_PATH = Path(__file__).resolve().with_name("sample_nonoverlap.xyz")
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
            "Run the single-GPU DistNNP-to-MACE adapter on a simple in-memory "
            "DistNNPState and print energy/forces."
        )
    )
    parser.add_argument(
        "--builtin-system",
        choices=sorted(BUILTIN_SYSTEMS.keys()),
        default="adapter_xyz6",
        help="Use a built-in structure example. Defaults to a non-overlapping xyz fixture in the adaptation folder.",
    )
    parser.add_argument(
        "--species",
        help=(
            "Comma-separated atomic numbers, e.g. '8,1,1'. "
            "Required if --builtin-system is not used."
        ),
    )
    parser.add_argument(
        "--positions-json",
        help=(
            "JSON list of positions in Angstrom, e.g. "
            "'[[0,0,0],[0.75,0,0.5],[-0.75,0,0.5]]'."
        ),
    )
    parser.add_argument(
        "--cell-json",
        default="[[0,0,0],[0,0,0],[0,0,0]]",
        help="Optional JSON 3x3 cell matrix.",
    )
    parser.add_argument(
        "--pbc-json",
        default="[false,false,false]",
        help="Optional JSON periodic boundary condition triple.",
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
        "--output-json",
        help="Optional path to save a JSON summary.",
    )
    parser.add_argument(
        "--xyz-path",
        help=(
            "Optional path to a simple XYZ/extxyz structure file. "
            "If set, this overrides --builtin-system."
        ),
    )
    return parser


def parse_xyz_file(path: Path) -> Dict[str, Any]:
    """Parse a minimal XYZ/extxyz file into the raw structure format used here.

    This stays self-contained inside the adaptation folder and is sufficient for
    the DistNNP test fixtures. It reads the first frame only.
    """
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
            raise ValueError(
                f"Element symbol '{symbol}' is not supported by this lightweight parser."
            )
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


def analyze_structure(raw_structure: Dict[str, Any]) -> Dict[str, Any]:
    """Compute simple geometry diagnostics to explain unstable model outputs."""
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


def parse_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    if args.xyz_path:
        return parse_xyz_file(Path(args.xyz_path))

    if args.builtin_system == "adapter_xyz6":
        return parse_xyz_file(DEFAULT_ADAPTER_STRUCTURE_PATH)

    if args.builtin_system == "distnnp_xyz6":
        return parse_xyz_file(DEFAULT_REAL_STRUCTURE_PATH)

    if args.builtin_system:
        return BUILTIN_SYSTEMS[args.builtin_system]

    if not args.species or not args.positions_json:
        raise ValueError(
            "Provide --xyz-path, --builtin-system, or both --species and --positions-json."
        )

    atomic_numbers = [
        int(token.strip()) for token in args.species.split(",") if token.strip()
    ]
    positions = json.loads(args.positions_json)
    cell = json.loads(args.cell_json)
    pbc = json.loads(args.pbc_json)

    if len(atomic_numbers) != len(positions):
        raise ValueError(
            "Number of atomic numbers must match number of position rows."
        )

    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
        "source_path": "cli-json",
    }


def build_single_gpu_distnnp_state(
    raw_structure: Dict[str, Any], device: torch.device, cutoff: float
) -> DistNNPState:
    """Create the smallest DistNNPState needed by the single-GPU adapter.

    This mirrors the raw structure -> Configuration mapping in the direct driver,
    but packages the same information into DistNNP-owned objects first.
    """

    def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
        """Assign only fields exposed by the currently loaded extension.

        On some environments the compiled ``distnnp_state`` extension can lag
        behind the source tree/stubs. This adapter only requires positions,
        atom types, cell, and PBC, so optional metadata should not be treated as
        mandatory for this single-GPU verification path.
        """
        if hasattr(obj, name):
            setattr(obj, name, value)

    def _portable_uint64_arange(count: int, target_device: torch.device) -> torch.Tensor:
        """Build uint64 vertex ids without relying on unsupported CUDA arange paths.

        DistNNP's IO tests expect ``atom_vertex_numbers`` to be uint64, but
        ``torch.arange(..., dtype=torch.uint64, device='cuda')`` is not
        implemented on the CUDA build used on the cluster. Creating the range as
        int64 first and casting preserves the intended DistNNP type while
        staying portable across CPU and CUDA.
        """
        return torch.arange(count, dtype=torch.int64, device=target_device).to(
            torch.uint64
        )

    num_atoms = len(raw_structure["atomic_numbers"])

    local_info = MolecularInfo()
    local_info.num_atoms = num_atoms
    local_info.atom_positions = torch.tensor(
        raw_structure["positions"], dtype=torch.float32, device=device
    )
    local_info.atom_types = torch.tensor(
        raw_structure["atomic_numbers"], dtype=torch.long, device=device
    )
    local_info.atom_velocities = torch.zeros(
        num_atoms, 3, dtype=torch.float32, device=device
    )
    local_info.atom_forces = torch.zeros(
        num_atoms, 3, dtype=torch.float32, device=device
    )
    _assign_if_present(
        local_info,
        "atom_masses",
        torch.zeros(num_atoms, dtype=torch.float32, device=device),
    )
    _assign_if_present(
        local_info,
        "atom_vertex_numbers",
        _portable_uint64_arange(num_atoms, device),
    )

    sim_config = SimulationConfig()
    sim_config.cutoff = cutoff
    sim_config.ghost_cutoff_buffer = 0.0
    sim_config.cell = torch.tensor(
        raw_structure["cell"], dtype=torch.float32, device=device
    )
    sim_config.pbc = torch.tensor(raw_structure["pbc"], dtype=torch.bool, device=device)

    dist_env = DistEnvironment()
    dist_env.global_rank = 0
    dist_env.local_rank = 0
    dist_env.world_size = 1
    dist_env.num_nodes = 1
    dist_env.gpus_per_node = 1

    partition_info = PartitionMessagePassingInfo()
    partition_info.partition_indices = torch.arange(
        num_atoms, dtype=torch.int64, device=device
    )
    partition_info.send_offsets = torch.zeros(2, dtype=torch.int64, device=device)
    partition_info.recv_offsets = torch.zeros(2, dtype=torch.int64, device=device)
    partition_info.send_local_indices = torch.empty(0, dtype=torch.int64, device=device)
    partition_info.recv_ghost_indices = torch.empty(0, dtype=torch.int64, device=device)
    partition_info.interior_mask = torch.ones(num_atoms, dtype=torch.bool, device=device)
    partition_info.global_send_recv_matrix = torch.tensor(
        [[num_atoms]], dtype=torch.int64, device=device
    )
    partition_info.partition_id = torch.zeros(num_atoms, dtype=torch.int64, device=device)
    partition_info.edge_index = torch.empty((2, 0), dtype=torch.int64, device=device)
    partition_info.edge_index_halo = torch.empty(
        (2, 0), dtype=torch.int64, device=device
    )

    state = DistNNPState()
    state.dist_env = dist_env
    state.global_molecular_info = local_info
    state.local_molecular_info = local_info
    state.local_partition_info = partition_info
    state.sim_config = sim_config
    return state


def summarize_adapter_outputs(
    raw_structure: Dict[str, Any],
    energy: torch.Tensor,
    forces: torch.Tensor,
    structure_diagnostics: Dict[str, Any],
    standalone_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source_path": raw_structure.get("source_path", f"builtin:{raw_structure.get('name', 'inline')}"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "energy": float(energy.detach().cpu().item()),
        "forces_shape": list(forces.shape),
        "forces": forces.detach().cpu().numpy().tolist(),
        "standalone_results": standalone_results,
    }


def print_summary(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("Single-GPU DistNNP -> MACE Adapter")
    print("============================================================")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print()
    print("------------------------------------------------------------")
    print("DistNNPState Input")
    print("------------------------------------------------------------")
    print(f"source_path: {payload['source_path']}")
    print(f"num_atoms: {payload['num_atoms']}")
    print(f"atomic_numbers: {payload['atomic_numbers']}")
    print(f"positions_A: {payload['positions_A']}")
    print()
    print("------------------------------------------------------------")
    print("Structure Diagnostics")
    print("------------------------------------------------------------")
    diagnostics = payload["structure_diagnostics"]
    print(f"min_pair_distance_A: {diagnostics['min_pair_distance_A']}")
    print(f"min_pair_pair: {diagnostics['min_pair_pair']}")
    print(f"has_duplicate_positions: {diagnostics['has_duplicate_positions']}")
    print(f"duplicate_pairs: {diagnostics['duplicate_pairs']}")
    print()
    print("------------------------------------------------------------")
    print("Adapter Output")
    print("------------------------------------------------------------")
    print(f"energy: {payload['energy']}")
    print(f"forces_shape: {payload['forces_shape']}")
    print(np.asarray(payload["forces"]))
    print()
    print("------------------------------------------------------------")
    print("Standalone MACE Calculator Output")
    print("------------------------------------------------------------")
    standalone_results = payload.get("standalone_results")
    if standalone_results is None:
        print("Standalone calculator path was not executed.")
    else:
        print(f"energy: {standalone_results['energy']}")
        print(f"forces_shape: {standalone_results['forces_shape']}")
        print(np.asarray(standalone_results["forces"]))


class MACEPotentialModel(DistNNPModel):
    """Minimal DistNNP adapter around a raw MACE model.

    By default this assumes ``DistNNPState.local_molecular_info.atom_types``
    already stores atomic numbers, which matches DistNNP's normal trajectory IO.
    If a workflow uses compact type IDs instead, provide ``atomic_number_map``.
    """

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
        self.z_table = AtomicNumberTable(
            [int(z) for z in model.atomic_numbers.detach().cpu().tolist()]
        )
        self.cutoff = float(model.r_max.detach().cpu().item())
        self.heads = getattr(model, "heads", ["Default"])

    def initialize_buffers(
        self, partition_info: PartitionMessagePassingInfo, device: torch.device
    ) -> None:
        # No communication or partition-shaped buffers are needed in this
        # single-GPU verification adapter.
        return None

    def _atomic_numbers_from_state(self, distributed_state: DistNNPState) -> np.ndarray:
        atom_types = distributed_state.local_molecular_info.atom_types
        atom_types_cpu = atom_types.detach().to(torch.int64).cpu()

        if not self.atomic_number_map:
            return atom_types_cpu.numpy()

        mapped = [self.atomic_number_map[int(atom_type)] for atom_type in atom_types_cpu]
        return np.asarray(mapped, dtype=np.int64)

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

    def _build_configuration(self, distributed_state: DistNNPState) -> Configuration:
        local_info = distributed_state.local_molecular_info
        sim_config = distributed_state.sim_config

        return Configuration(
            atomic_numbers=self._atomic_numbers_from_state(distributed_state),
            positions=local_info.atom_positions.detach().cpu().to(torch.float64).numpy(),
            properties={},
            property_weights={},
            cell=self._cell_to_numpy(sim_config.cell),
            pbc=self._pbc_to_tuple(sim_config.pbc),
            weight=1.0,
            config_type="DistNNP_SingleGPU",
            head="Default",
        )

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        config = self._build_configuration(distributed_state)
        atomic_data = AtomicData.from_config(
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.heads,
        )

        device = distributed_state.local_molecular_info.atom_positions.device
        batch = Batch.from_data_list([atomic_data]).to(device)
        batch_dict = align_batch_dtype(batch.to_dict(), self.model)

        outputs = self.model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_stress=False,
            compute_hessian=False,
        )

        energy = outputs["energy"]
        forces = outputs["forces"]

        if torch.is_tensor(energy) and energy.numel() == 1:
            energy = energy.reshape(())

        return energy.detach(), forces.detach()


def compute_standalone_calculator_outputs(
    raw_structure: Dict[str, Any],
    model_family: str,
    model: str,
    device: str,
) -> Dict[str, Any]:
    """Run the same structure through the public MACE calculator API."""
    _require_ase()
    calc = load_mace_calculator(
        model_family=model_family,
        model=model,
        device=device,
    )

    atomic_numbers = raw_structure["atomic_numbers"]
    positions = np.asarray(raw_structure["positions"], dtype=np.float64)
    cell = np.asarray(raw_structure["cell"], dtype=np.float64)
    pbc = tuple(bool(x) for x in raw_structure["pbc"])

    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    atoms.calc = calc

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    return {
        "energy": energy,
        "forces_shape": list(forces.shape),
        "forces": forces.tolist(),
    }


def main() -> None:
    args = build_parser().parse_args()
    raw_structure = parse_raw_structure(args)
    structure_diagnostics = analyze_structure(raw_structure)

    raw_model = load_raw_mace_model(
        model_family=args.model_family,
        model=args.model,
        device=args.device,
    )
    raw_model.to(args.device)
    raw_model.eval()

    adapter = MACEPotentialModel(raw_model)
    distnnp_state = build_single_gpu_distnnp_state(
        raw_structure=raw_structure,
        device=torch.device(args.device),
        cutoff=float(raw_model.r_max.detach().cpu().item()),
    )

    # The direct standalone driver maps raw structure -> Configuration directly.
    # This verification script inserts the DistNNPState layer in between:
    # raw structure -> DistNNPState -> MACEPotentialModel -> energy/forces.
    energy, forces = adapter(distnnp_state)

    standalone_results = compute_standalone_calculator_outputs(
        raw_structure=raw_structure,
        model_family=args.model_family,
        model=args.model,
        device=args.device,
    )

    payload = summarize_adapter_outputs(
        raw_structure,
        energy,
        forces,
        structure_diagnostics,
        standalone_results=standalone_results,
    )
    print_summary(payload, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
