"""Single-GPU MACE adapter for the DistNNP model contract.

This adapter is intentionally simple:

- no halo exchange
- no repartition-aware buffers
- no use of ASE

It exists to verify two things:
1. DistNNPState can be converted into the public MACE graph input path.
2. A loaded MACE model can satisfy the DistNNPModel energy/forces contract.

Also note that the force computation comes from within MACE and not DistNNP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional runtime install
    mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None
    _MACE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "MACE imports failed. Install the public 'mace-torch' package in the "
            "runtime environment before using MACEPotentialModel."
        ) from _MACE_IMPORT_ERROR


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


BUILTIN_SYSTEMS: Dict[str, Dict[str, Any]] = {
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
        help="Use a built-in structure example.",
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
    return parser


def parse_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    if args.builtin_system:
        return BUILTIN_SYSTEMS[args.builtin_system]

    if not args.species or not args.positions_json:
        raise ValueError(
            "Provide --builtin-system or both --species and --positions-json."
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
    }


def build_single_gpu_distnnp_state(
    raw_structure: Dict[str, Any], device: torch.device, cutoff: float
) -> DistNNPState:
    """Create the smallest DistNNPState needed by the single-GPU adapter.

    This mirrors the raw structure -> Configuration mapping in the direct driver,
    but packages the same information into DistNNP-owned objects first.
    """
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
    local_info.atom_masses = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    local_info.atom_vertex_numbers = torch.arange(
        num_atoms, dtype=torch.uint64, device=device
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
    raw_structure: Dict[str, Any], energy: torch.Tensor, forces: torch.Tensor
) -> Dict[str, Any]:
    return {
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "energy": float(energy.detach().cpu().item()),
        "forces_shape": list(forces.shape),
        "forces": forces.detach().cpu().numpy().tolist(),
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
    print(f"atomic_numbers: {payload['atomic_numbers']}")
    print(f"positions_A: {payload['positions_A']}")
    print()
    print("------------------------------------------------------------")
    print("Adapter Output")
    print("------------------------------------------------------------")
    print(f"energy: {payload['energy']}")
    print(f"forces_shape: {payload['forces_shape']}")
    print(np.asarray(payload["forces"]))


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


def main() -> None:
    args = build_parser().parse_args()
    raw_structure = parse_raw_structure(args)

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

    payload = summarize_adapter_outputs(raw_structure, energy, forces)
    print_summary(payload, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
