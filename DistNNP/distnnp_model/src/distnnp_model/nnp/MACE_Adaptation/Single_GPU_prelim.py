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

from typing import Dict, Mapping, Optional

import numpy as np
import torch

from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, PartitionMessagePassingInfo

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
    device: str = "cpu",
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
