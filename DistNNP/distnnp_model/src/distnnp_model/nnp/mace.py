"""Distributed MACE model focused on steady-state execution.

This adapter keeps the current DistNNP + MACE algorithm intact:

* DistNNP owns partitioning and halo exchange metadata.
* Positions are exchanged every step to build a local ``[owned | halo]`` view.
* DistNNP's ``radius_graph_pbc_cuda`` builds the explicit graph.
* Only edges whose destination is locally owned are passed into MACE.
* Learned node features are exchanged after every non-final product block.
* Forces are recovered from the owned local energy via autograd.

This implementation intentionally excludes diagnostics, debug tracing,
observations, and execution modes from the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, PartitionMessagePassingInfo, radius_graph_pbc_cuda

COMPUTE_DTYPE = torch.float32

try:
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.modules import ScaleShiftMACE
    from mace.modules.utils import prepare_graph

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    mace_mp = mace_off = mace_omol = mace_polar = None
    ScaleShiftMACE = None  # type: ignore[assignment]
    prepare_graph = None
    _MACE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError("MACEModel requires the MACE package.") from _MACE_IMPORT_ERROR


def load_raw_mace_model(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    """Load a public MACE foundation model without its ASE calculator wrapper."""
    _require_mace()
    loaders = {"mp": mace_mp, "off": mace_off, "omol": mace_omol, "polar": mace_polar}
    if model_family not in loaders:
        raise ValueError(f"Unsupported MACE model family: {model_family}")
    loaded_model = loaders[model_family](
        model=model,
        device=device,
        default_dtype="float32",
        return_raw_model=True,
    )
    return loaded_model.to(dtype=COMPUTE_DTYPE)


def validate_partition_info(
    partition_info: PartitionMessagePassingInfo,
    world_size: int,
) -> Dict[str, Any]:
    """Validate the CSR metadata consumed by DistNNP exchange."""
    send_offsets = partition_info.send_offsets.to(torch.long)
    recv_offsets = partition_info.recv_offsets.to(torch.long)
    send_local_indices = partition_info.send_local_indices.to(torch.long)
    recv_ghost_indices = partition_info.recv_ghost_indices.to(torch.long)
    n_local = int(partition_info.partition_indices.numel())
    if send_offsets.numel() != world_size + 1 or recv_offsets.numel() != world_size + 1:
        raise ValueError("Halo offsets must contain world_size + 1 entries.")
    if send_offsets[0].item() != 0 or recv_offsets[0].item() != 0:
        raise ValueError("Halo offsets must start at zero.")
    if not bool(torch.all(send_offsets[1:] >= send_offsets[:-1])):
        raise ValueError("send_offsets must be monotonic.")
    if not bool(torch.all(recv_offsets[1:] >= recv_offsets[:-1])):
        raise ValueError("recv_offsets must be monotonic.")
    total_send = int(send_offsets[-1].item())
    total_recv = int(recv_offsets[-1].item())
    if send_local_indices.numel() != total_send:
        raise ValueError("send_local_indices does not match send_offsets.")
    if recv_ghost_indices.numel() != total_recv:
        raise ValueError("recv_ghost_indices does not match recv_offsets.")
    if total_send > 0 and (
        int(send_local_indices.min().item()) < 0 or int(send_local_indices.max().item()) >= n_local
    ):
        raise ValueError("send_local_indices must stay within the local atom range.")
    expected_ghost_indices = torch.arange(
        n_local,
        n_local + total_recv,
        dtype=torch.long,
        device=recv_ghost_indices.device,
    )
    if total_recv > 0 and not torch.equal(recv_ghost_indices, expected_ghost_indices):
        raise ValueError(
            "recv_ghost_indices must enumerate contiguous ghost slots in peer-grouped CSR order."
        )
    return {
        "n_local": n_local,
        "total_send": total_send,
        "total_recv": total_recv,
        "send_counts_by_peer": (send_offsets[1:] - send_offsets[:-1]).cpu().tolist(),
        "recv_counts_by_peer": (recv_offsets[1:] - recv_offsets[:-1]).cpu().tolist(),
    }


def resolve_model_head_name(model: Any, requested_head_name: Optional[str] = None) -> str:
    heads = list(getattr(model, "heads", []))
    if requested_head_name:
        if heads and requested_head_name not in heads:
            raise ValueError(
                f"Requested head_name={requested_head_name!r} is not available in model heads {heads}."
            )
        return requested_head_name
    if not heads:
        return "Default"
    if len(heads) == 1:
        return str(heads[0])
    if "default" in heads:
        return "default"
    if "Default" in heads:
        return "Default"
    raise ValueError(
        "A model head could not be inferred from the loaded model; "
        f"available heads: {heads}. Pass head_name explicitly."
    )


def resolve_model_head_index(model: Any, requested_head_name: Optional[str] = None) -> int:
    heads = list(getattr(model, "heads", []))
    if not heads:
        return 0
    return heads.index(resolve_model_head_name(model, requested_head_name))


@dataclass
class MACEStructuralView:
    local_positions: torch.Tensor
    exchanged_positions: torch.Tensor
    num_local: int
    num_total: int


@dataclass
class MACEOwnedGraph:
    edge_index: torch.Tensor
    unit_shifts: torch.Tensor
    shifts: torch.Tensor
    batch_dict: Dict[str, torch.Tensor]


class _RadiusGraphInput:
    def __init__(self, positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> None:
        self.pos = positions
        self.cell = cell.reshape(1, 3, 3)
        self.pbc = pbc.reshape(1, 3)
        self.natoms = torch.tensor([positions.shape[0]], dtype=torch.long, device=positions.device)


##===================================================
#End of utility functions
##===================================================

class MACEScaleShiftRuntime(torch.nn.Module):
    """Explicit runtime wrapper for the distributed MACE forward path."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self._state: Optional[DistNNPState] = None
        self._exchange_enabled = False
        self._feature_partition_info: Optional[PartitionMessagePassingInfo] = None
        self._feature_device: Optional[torch.device] = None
        self._feature_width: Optional[int] = None
        self._feature_exchanger = GPUDirectExchanger()

    def configure_feature_exchange_runtime(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None:
        self._feature_partition_info = partition_info
        self._feature_device = device
        self._feature_width = None
        self._feature_exchanger = GPUDirectExchanger()

    def begin_forward(self, state: DistNNPState, exchange_enabled: bool) -> None:
        self._state = state
        self._exchange_enabled = bool(exchange_enabled)

    def end_forward(self) -> None:
        self._state = None
        self._exchange_enabled = False

    def _ensure_feature_exchanger_ready(
        self,
        node_feats: torch.Tensor,
        state: DistNNPState,
    ) -> None:
        if node_feats.dim() != 2:
            raise ValueError("The MACE runtime expects 2-D node features.")
        feature_width = int(node_feats.shape[-1])
        if self._feature_width is None:
            self._feature_width = feature_width
        elif self._feature_width != feature_width:
            raise ValueError("Feature exchange width changed across interaction layers.")
        if self._feature_exchanger.is_ipc_setup:
            return
        partition_info = self._feature_partition_info or state.local_partition_info
        device = self._feature_device or node_feats.device
        self._feature_exchanger.setup_exchanger(feature_width, partition_info, device)

    def _should_exchange_after_product(self, layer_index: int) -> bool:
        return layer_index < max(0, len(getattr(self.model, "interactions", ())) - 1)

    def _exchange_after_product(
        self,
        node_feats: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if not self._exchange_enabled or not self._should_exchange_after_product(layer_index):
            return node_feats
        state = self._state
        if state is None:
            raise RuntimeError("Feature exchange reached without an active DistNNPState.")
        num_local = int(state.local_molecular_info.num_atoms)
        self._ensure_feature_exchanger_ready(node_feats, state)
        #The actual feature exchange point between layers
        exchanged = FeaturePushToExchange.apply(
            node_feats[:num_local].contiguous(),
            state,
            self._feature_exchanger,
        )
        return exchanged.to(dtype=node_feats.dtype)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _require_mace()
        ctx = prepare_graph(
            data,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        node_heads = ctx.node_heads.to(torch.int64)
        atom_range = ctx.num_atoms_arange.to(torch.int64)
        node_e0 = self.model.atomic_energies_fn(data["node_attrs"])[atom_range, node_heads]
        node_feats = self.model.node_embedding(data["node_attrs"])
        edge_attrs = self.model.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.model.radial_embedding(
            ctx.lengths,
            data["node_attrs"],
            data["edge_index"],
            self.model.atomic_numbers,
        )
        pair_node_energy = (
            self.model.pair_repulsion_fn(
                ctx.lengths,
                data["node_attrs"],
                data["edge_index"],
                self.model.atomic_numbers,
            )
            if hasattr(self.model, "pair_repulsion")
            else torch.zeros_like(node_e0)
        )
        node_es_list = [pair_node_energy]
        node_feats_list = []
        num_readouts = len(self.model.readouts)
        num_layers = len(self.model.interactions)
        if num_readouts not in (1, num_layers):
            raise TypeError(
                "The MACE runtime expects either one final readout or one readout per "
                f"interaction layer; got num_readouts={num_readouts}, num_layers={num_layers}."
            )
        for layer_index, (interaction, product) in enumerate(
            zip(self.model.interactions, self.model.products)
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(layer_index == 0),
                lammps_class=ctx.interaction_kwargs.lammps_class,
                lammps_natoms=ctx.interaction_kwargs.lammps_natoms,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            node_feats = self._exchange_after_product(node_feats, layer_index)
            node_feats_list.append(node_feats)
        for readout_index, readout in enumerate(self.model.readouts):
            feature_index = -1 if len(self.model.readouts) == 1 else readout_index
            node_es_list.append(
                readout(node_feats_list[feature_index], node_heads)[atom_range, node_heads]
            )
        node_inter_es = self.model.scale_shift(
            torch.sum(torch.stack(node_es_list), dim=0),
            node_heads,
        )
        node_energy = node_e0 + node_inter_es
        state = self._state
        if state is None:
            raise RuntimeError("The MACE runtime requires an active DistNNPState.")
        num_local = int(state.local_molecular_info.num_atoms)
        owned_node_energy = node_energy.reshape(-1)[:num_local]
        return {
            "owned_node_energy": owned_node_energy,
            "owned_local_energy": owned_node_energy.sum(),
        }
        #removed the force recovery logic performed by MACE beyond this point
        #can experiment with this enabled to verify if speedup caused by this removal?


class MACEModel(DistNNPModel):
    """MP/MPA-oriented explicit-graph MACE model."""

    def __init__(
        self,
        model: torch.nn.Module,
        atomic_number_map: Optional[Mapping[int, int]] = None,
        head_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        _require_mace()
        self._validate_model_contract(model)
        self.model = model.to(dtype=COMPUTE_DTYPE).eval()
        self.head_name = resolve_model_head_name(self.model, head_name)
        self.atomic_number_map = dict(atomic_number_map or {})
        self.cutoff = float(self.model.r_max.detach().cpu().item())
        self.runtime = MACEScaleShiftRuntime(self.model)
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        self.exchange_enabled = False
        self.partition_validation: Dict[str, Any] = {}
        self._partition_info_id: Optional[int] = None
        self._partition_device: Optional[torch.device] = None
        self._static_cache_ready = False
        self._static_cache: Dict[str, torch.Tensor] = {}
        self._atomic_number_lookup_by_device: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_model_contract(model: Any) -> None:
        required_attrs: Sequence[str] = (
            "r_max",
            "atomic_numbers",
            "node_embedding",
            "spherical_harmonics",
            "radial_embedding",
            "interactions",
            "products",
            "readouts",
            "atomic_energies_fn",
            "scale_shift",
        )
        missing = [name for name in required_attrs if not hasattr(model, name)]
        if missing:
            raise TypeError(
                "MACEModel targets MP/MPA-style ScaleShiftMACE-compatible "
                f"models. Missing required attributes: {', '.join(missing)}."
            )
        if ScaleShiftMACE is not None and not isinstance(model, ScaleShiftMACE):
            raise TypeError(
                "MACEModel currently supports ScaleShiftMACE-compatible checkpoints."
            )
        num_layers = len(getattr(model, "interactions"))
        if num_layers != len(getattr(model, "products")):
            raise TypeError("MACE interactions and products must have matching lengths.")
        num_readouts = len(getattr(model, "readouts"))
        if num_readouts not in (1, num_layers):
            raise TypeError(
                "MP/MPA checkpoints must expose either one final readout or one readout per "
                f"interaction layer; got num_readouts={num_readouts}, num_layers={num_layers}."
            )

    def initialize_buffers(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
        world_size: Optional[int] = None,
    ) -> None:
        if world_size is None:
            world_size = int(partition_info.global_send_recv_matrix.shape[0])
        self.partition_validation = validate_partition_info(partition_info, world_size)
        total_send = self.partition_validation["total_send"]
        total_recv = self.partition_validation["total_recv"]
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        self.runtime.configure_feature_exchange_runtime(partition_info, device)
        self._partition_info_id = id(partition_info)
        self._partition_device = device
        self._static_cache_ready = False
        self._static_cache = {}
        self._atomic_number_lookup_by_device = {}
        if total_send == 0 and total_recv == 0:
            self.exchange_enabled = False
            return
        if total_send == 0 or total_recv == 0:
            raise RuntimeError("Asymmetric halo traffic cannot form a local DistNNP view.")
        self.position_exchanger.setup_exchanger(3, partition_info, device)
        self.atom_type_exchanger.setup_exchanger(1, partition_info, device)
        self.exchange_enabled = True

    def _current_partition_is_initialized(self, state: DistNNPState) -> bool:
        return self._partition_info_id is not None and self._partition_info_id == id(
            state.local_partition_info
        )

    def _build_atomic_number_lookup(
        self,
        device: torch.device,
        max_atom_type: int,
    ) -> torch.Tensor:
        key = str(device)
        lookup = self._atomic_number_lookup_by_device.get(key)
        if lookup is None or int(lookup.shape[0]) <= max_atom_type:
            max_key = max(int(source) for source in self.atomic_number_map)
            size = max(max_atom_type, max_key) + 1
            lookup = torch.full((size,), -1, dtype=torch.long, device=device)
            for source, destination in self.atomic_number_map.items():
                lookup[int(source)] = int(destination)
            self._atomic_number_lookup_by_device[key] = lookup
        return lookup

    def _map_atomic_numbers(self, atom_types: torch.Tensor) -> torch.Tensor:
        atom_types = atom_types.to(torch.long)
        if not self.atomic_number_map:
            return atom_types
        max_atom_type = int(atom_types.max().item()) if atom_types.numel() > 0 else -1
        lookup = self._build_atomic_number_lookup(atom_types.device, max_atom_type)
        mapped = lookup.index_select(0, atom_types.reshape(-1)).reshape(atom_types.shape)
        missing_mask = mapped.lt(0)
        if bool(missing_mask.any()):
            missing = atom_types[missing_mask].unique().cpu().tolist()
            raise ValueError(
                "MACE species compatibility check failed: unsupported atomic numbers "
                f"after remapping {missing}"
            )
        return mapped

    def _build_static_partition_cache(self, state: DistNNPState) -> None:
        if self._static_cache_ready:
            return
        if not self._current_partition_is_initialized(state):
            raise RuntimeError(
                "Partition buffers are stale for the current DistNNPState. Call initialize_buffers() "
                "after repartition before using MACEModel."
            )
        local_types = state.local_molecular_info.atom_types.to(torch.long)
        local_global = state.local_partition_info.partition_indices.long()
        expected_total = int(local_global.numel() + self.partition_validation["total_recv"])
        if self.exchange_enabled:
            with torch.no_grad():
                exchanged_types = FeaturePushToExchange.apply(
                    local_types.float().unsqueeze(-1),
                    state,
                    self.atom_type_exchanger,
                ).squeeze(-1).long()
        else:
            exchanged_types = local_types
        mapped_atomic_numbers = self._map_atomic_numbers(exchanged_types)
        model_atomic_numbers = self.model.atomic_numbers.to(
            device=mapped_atomic_numbers.device,
            dtype=torch.long,
        )
        node_attrs = mapped_atomic_numbers[:, None].eq(model_atomic_numbers[None, :]).to(
            dtype=COMPUTE_DTYPE
        )
        row_match_counts = node_attrs.sum(dim=1)
        if not bool(torch.all(row_match_counts == 1)):
            raise ValueError("MACE species compatibility check failed for the exchanged atom types.")
        device = mapped_atomic_numbers.device
        head_index = resolve_model_head_index(self.model, self.head_name)
        self._static_cache = {
            "node_attrs": node_attrs,
            "batch": torch.zeros(expected_total, dtype=torch.long, device=device),
            "ptr": torch.tensor([0, expected_total], dtype=torch.long, device=device),
            "head": torch.tensor([head_index], dtype=torch.long, device=device),
            "expected_total": torch.tensor(expected_total, dtype=torch.long, device=device),
            "num_local": torch.tensor(int(local_global.numel()), dtype=torch.long, device=device),
        }
        self._static_cache_ready = True

    def _build_structural_view(self, state: DistNNPState) -> MACEStructuralView:
        self._build_static_partition_cache(state)
        local_positions = (
            state.local_molecular_info.atom_positions.detach()
            .to(dtype=COMPUTE_DTYPE)
            .clone()
            .requires_grad_(True)
        )
        if self.exchange_enabled:
            exchanged_positions = FeaturePushToExchange.apply(
                local_positions,
                state,
                self.position_exchanger,
            )
        else:
            exchanged_positions = local_positions
        expected_total = int(self._static_cache["expected_total"].item())
        if exchanged_positions.shape[0] != expected_total:
            raise ValueError("Structural rows disagree with DistNNP recv metadata.")
        return MACEStructuralView(
            local_positions=local_positions,
            exchanged_positions=exchanged_positions,
            num_local=int(self._static_cache["num_local"].item()),
            num_total=expected_total,
        )

    def _radius_graph(
        self,
        positions: torch.Tensor,
        state: DistNNPState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = state.sim_config
        with torch.no_grad():
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                _RadiusGraphInput(positions, sim.cell, sim.pbc),
                radius=self.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )
        return edge_index.long(), -cell_offsets

    def _build_owned_graph(self, view: MACEStructuralView, state: DistNNPState) -> MACEOwnedGraph:
        edge_index, unit_shifts = self._radius_graph(view.exchanged_positions, state)
        owned_destination = edge_index[1] < view.num_local
        edge_index = edge_index[:, owned_destination].contiguous()
        unit_shifts = unit_shifts[owned_destination].to(dtype=COMPUTE_DTYPE)
        cell = state.sim_config.cell.reshape(3, 3).to(dtype=COMPUTE_DTYPE)
        shifts = torch.matmul(unit_shifts, cell)
        batch_dict = {
            "positions": view.exchanged_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "cell": cell,
            "node_attrs": self._static_cache["node_attrs"],
            "batch": self._static_cache["batch"],
            "ptr": self._static_cache["ptr"],
            "head": self._static_cache["head"],
        }
        batch_dict = {
            key: value.to(COMPUTE_DTYPE) if torch.is_floating_point(value) else value
            for key, value in batch_dict.items()
        }
        return MACEOwnedGraph(edge_index=edge_index, unit_shifts=unit_shifts, shifts=shifts, batch_dict=batch_dict)

    @staticmethod
    def _recover_forces(
        owned_energy: torch.Tensor,
        view: MACEStructuralView,
    ) -> torch.Tensor:
        (local_grad,) = torch.autograd.grad(owned_energy, [view.local_positions], allow_unused=False)
        return -local_grad

    def __call__(self, state: DistNNPState, *args: Any, **kwargs: Any):
        view = self._build_structural_view(state)
        graph = self._build_owned_graph(view, state)
        self.runtime.begin_forward(state, self.exchange_enabled)
        try:
            outputs = self.runtime(graph.batch_dict)
            owned_energy = outputs["owned_local_energy"]
            owned_forces = self._recover_forces(owned_energy, view)
        finally:
            self.runtime.end_forward()
        return owned_energy.detach(), owned_forces.detach()


__all__ = [
    "COMPUTE_DTYPE",
    "MACEModel",
    "MACEScaleShiftRuntime",
    "load_raw_mace_model",
    "resolve_model_head_index",
    "resolve_model_head_name",
    "validate_partition_info",
]
