"""Distributed MACE adapter with DistNNP-owned graph construction.

DistNNP supplies ownership and halo membership. ``radius_graph_pbc_cuda`` then
builds one graph over the exchanged ``[local | halo]`` coordinates, and only
edges whose destination is locally owned enter MACE. The distributed path does
not use ``AtomicData.from_config`` for neighbor generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, PartitionMessagePassingInfo, radius_graph_pbc_cuda

# Keep model, structural exchange, graph geometry, and recovered forces aligned
# with DistNNP's standard communication dtype.
COMPUTE_DTYPE = torch.float32

try:
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.modules import ScaleShiftMACE
    from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
    from mace.tools.scatter import scatter_sum

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional cluster dependency
    mace_mp = mace_off = mace_omol = mace_polar = None
    ScaleShiftMACE = None  # type: ignore[assignment]
    get_atomic_virials_stresses = get_outputs = prepare_graph = scatter_sum = None
    _MACE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError("The DistMACE adapter requires the MACE package.") from _MACE_IMPORT_ERROR


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


@dataclass
class StructuralHaloView:
    local_positions: torch.Tensor
    exchanged_positions: torch.Tensor
    exchanged_atomic_numbers: torch.Tensor
    local_global_indices: torch.Tensor
    halo_global_indices: torch.Tensor
    expanded_global_indices: torch.Tensor
    num_local: int
    num_total: int

    @property
    def num_halo(self) -> int:
        return self.num_total - self.num_local


@dataclass
class ExplicitOwnedGraph:
    edge_index: torch.Tensor
    unit_shifts: torch.Tensor
    shifts: torch.Tensor
    batch_dict: Dict[str, torch.Tensor]

#long term issue to be fixed in radius graph pbc cuda fn?
def canonicalize_nonperiodic_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    padding: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place nonperiodic coordinates inside an orthorhombic padded cell.

    DistNNP's current CUDA radius graph wraps coordinates on every axis. A
    rigid translation into the positive cell prevents that implementation
    detail from changing nonperiodic distances before halo construction.
    """
    positions = positions.to(dtype=COMPUTE_DTYPE)
    cell = cell.reshape(3, 3).to(device=positions.device, dtype=COMPUTE_DTYPE)
    pbc = pbc.reshape(3).to(device=positions.device, dtype=torch.bool)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_atoms, 3].")
    if padding <= 0:
        raise ValueError("nonperiodic padding must be positive.")
    if bool(torch.all(pbc)):
        return positions, cell, torch.zeros(3, dtype=COMPUTE_DTYPE, device=positions.device)

    off_diagonal = cell - torch.diag(torch.diagonal(cell))
    if not bool(torch.allclose(off_diagonal, torch.zeros_like(off_diagonal))):
        raise ValueError(
            "Nonperiodic coordinate canonicalization currently requires an "
            "orthorhombic cell."
        )

    adjusted_positions = positions.clone()
    adjusted_cell = cell.clone()
    translation = torch.zeros(3, dtype=COMPUTE_DTYPE, device=positions.device)
    for axis in range(3):
        if bool(pbc[axis]):
            continue
        axis_min = positions[:, axis].min()
        axis_max = positions[:, axis].max()
        translation[axis] = float(padding) - axis_min
        adjusted_positions[:, axis] += translation[axis]
        required_length = (axis_max - axis_min) + 2.0 * float(padding)
        adjusted_cell[axis, axis] = torch.maximum(
            adjusted_cell[axis, axis], required_length
        )
    return adjusted_positions, adjusted_cell, translation

def validate_partition_info(
    partition_info: PartitionMessagePassingInfo,
    world_size: int,
) -> Dict[str, Any]:
    """Validate the CSR metadata consumed by every DistNNP exchange.

    The same peer-grouped CSR layout is reused for structural exchange and for
    every learned-feature exchange, so we validate it once up front.
    """
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


def _edge_records(
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    global_indices: torch.Tensor,
) -> set[Tuple[int, int, int, int, int]]:
    if edge_index.numel() == 0:
        return set()
    src = global_indices.index_select(0, edge_index[0].long()).cpu().tolist()
    dst = global_indices.index_select(0, edge_index[1].long()).cpu().tolist()
    offsets = unit_shifts.round().long().cpu().tolist()
    return {
        (int(i), int(j), int(offset[0]), int(offset[1]), int(offset[2]))
        for i, j, offset in zip(src, dst, offsets)
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


def summarize_model_contract(
    model: Any,
    expected_exchange_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    summary = {
        "model_class": model.__class__.__name__,
        "num_interactions": len(getattr(model, "interactions", ())),
        "num_products": len(getattr(model, "products", ())),
        "num_readouts": len(getattr(model, "readouts", ())),
        "heads": list(getattr(model, "heads", [])),
        "atomic_number_count": int(getattr(model, "atomic_numbers").numel()),
        "cutoff": float(model.r_max.detach().cpu()),
        "interaction_types": [
            interaction.__class__.__name__
            for interaction in getattr(model, "interactions", ())
        ],
        "product_types": [
            product.__class__.__name__ for product in getattr(model, "products", ())
        ],
        "readout_types": [
            readout.__class__.__name__ for readout in getattr(model, "readouts", ())
        ],
    }
    if expected_exchange_layers is not None:
        summary["expected_exchange_layers"] = list(expected_exchange_layers)
    return summary


def build_graph_authority_record(
    view: StructuralHaloView,
    local_graph: ExplicitOwnedGraph,
    state: DistNNPState,
    radius_graph_fn: Any,
    edge_records_fn: Any,
) -> Dict[str, Any]:
    global_positions = state.global_molecular_info.atom_positions
    global_edge_index, global_unit_shifts = radius_graph_fn(global_positions, state)
    owner = state.local_partition_info.partition_id.long()
    rank = int(state.dist_env.global_rank)
    mask = owner.index_select(0, global_edge_index[1]).eq(rank)
    global_ids = torch.arange(global_positions.shape[0], device=global_positions.device)
    expected = edge_records_fn(global_edge_index[:, mask], global_unit_shifts[mask], global_ids)
    actual = edge_records_fn(
        local_graph.edge_index, local_graph.unit_shifts, view.expanded_global_indices
    )
    owned = set(view.local_global_indices.cpu().tolist())
    required_remote = sorted(
        {src for src, dst, *_ in expected if dst in owned and src not in owned}
    )
    actual_halo = set(view.halo_global_indices.cpu().tolist())
    missing_halo = sorted(set(required_remote) - actual_halo)
    missing_edges = sorted(expected - actual)
    extra_edges = sorted(actual - expected)
    halo_issue = bool(missing_halo)
    reconstruction_issue = not halo_issue and bool(missing_edges or extra_edges)
    return {
        "stage": "pre_forward_graph_authority",
        "radius_graph_authority": "distnnp_state.radius_graph_pbc_cuda",
        "required_remote_global_indices": required_remote,
        "actual_halo_global_indices": sorted(actual_halo),
        "missing_required_halo_global_indices": missing_halo,
        "halo_builder_issue": halo_issue,
        "local_graph_matches_global_owned_graph": not (missing_edges or extra_edges),
        "local_graph_reconstruction_issue": reconstruction_issue,
        "expected_owned_edge_count": len(expected),
        "actual_owned_edge_count": len(actual),
        "missing_owned_edge_preview": [list(item) for item in missing_edges[:12]],
        "extra_owned_edge_preview": [list(item) for item in extra_edges[:12]],
        "structural_classification": (
            "halo_builder_issue"
            if halo_issue
            else "local_graph_reconstruction_issue"
            if reconstruction_issue
            else "structural_graph_ok"
        ),
    }


def make_adapter_observation(
    model: Any,
    state: DistNNPState,
    graph_verification: Dict[str, Any],
    expected_exchange_layers: List[int],
) -> Dict[str, Any]:
    return {
        "rank": int(state.dist_env.global_rank),
        "feature_exchange_used_ipc": False,
        "layer_records": [],
        "graph_verification": graph_verification,
        "model_contract": summarize_model_contract(
            model, expected_exchange_layers=expected_exchange_layers
        ),
    }


def finalize_adapter_observation(
    observation: Dict[str, Any],
    partition_validation: Dict[str, Any],
    view: StructuralHaloView,
    graph: ExplicitOwnedGraph,
    outputs: Dict[str, Any],
    owned_energy: torch.Tensor,
    owned_forces: torch.Tensor,
    force_checks: Dict[str, Any],
    model_parameter_dtype: str,
) -> Dict[str, Any]:
    observation.update(
        {
            "n_local": view.num_local,
            "n_halo": view.num_halo,
            "n_total_after_exchange": view.num_total,
            "local_global_indices": view.local_global_indices.cpu().tolist(),
            "halo_global_indices": view.halo_global_indices.cpu().tolist(),
            "send_counts_by_peer": partition_validation["send_counts_by_peer"],
            "recv_counts_by_peer": partition_validation["recv_counts_by_peer"],
            "owned_destination_edge_count": int(graph.edge_index.shape[1]),
            "compute_dtype": str(view.exchanged_positions.dtype),
            "model_parameter_dtype": model_parameter_dtype,
            "position_dtype": str(view.exchanged_positions.dtype),
            "force_dtype": str(owned_forces.dtype),
            "owned_node_energy_values": outputs["owned_node_energy"].detach().cpu().tolist(),
            "owned_local_energy": float(owned_energy.detach().cpu()),
            "owned_local_force_values": owned_forces.detach().cpu().tolist(),
            **force_checks,
        }
    )
    return observation


class _RadiusGraphInput:
    def __init__(self, positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> None:
        self.pos = positions
        self.cell = cell.reshape(1, 3, 3)
        self.pbc = pbc.reshape(1, 3)
        self.natoms = torch.tensor([positions.shape[0]], dtype=torch.long, device=positions.device)

##===============================================================
##--- End of utilites ------
##===============================================================

class DistScaleShiftMACERuntimeMixin:
    """Runtime mixin for MP/MPA-like ScaleShiftMACE models.

    The adapter installs this mixin onto the loaded model class so we can keep
    the normal MACE loop while inserting learned-feature exchange between
    interaction stages.
    """

    __distmace_explicit_graph__ = True

    def _configure_dist_runtime(self) -> None:
        self._distmace_context: Dict[str, Any] = {
            "state": None,
            "exchange_enabled": False,
            "observation": None,
            "feature_partition_info": None,
            "feature_device": None,
            "feature_width": None,
            "execution_mode": "validate",
            "debug_tracing": False,
            "debug_prefix": "distmace-debug",
        }
        self._distmace_feature_exchanger = GPUDirectExchanger()

    def _configure_feature_exchange_runtime(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None:
        self._distmace_feature_exchanger = GPUDirectExchanger()
        self._distmace_context.update(
            feature_partition_info=partition_info,
            feature_device=device,
            feature_width=None,
        )

    def _set_debug_tracing(self, enabled: bool, prefix: str = "distmace-debug") -> None:
        self._distmace_context.update(debug_tracing=bool(enabled), debug_prefix=str(prefix))

    def _set_execution_mode(self, mode: str) -> None:
        self._distmace_context["execution_mode"] = str(mode)

    def _execution_mode(self) -> str:
        return str(self._distmace_context.get("execution_mode", "validate"))

    def _debug_enabled(self) -> bool:
        return bool(self._distmace_context.get("debug_tracing"))

    def _debug_rank(self) -> int:
        if dist.is_initialized():
            return int(dist.get_rank())
        return -1

    def _emit_debug_log(self, stage: str, **payload: Any) -> None:
        if not self._debug_enabled():
            return
        prefix = self._distmace_context.get("debug_prefix", "distmace-debug")
        rank = self._debug_rank()
        message = [f"[{prefix}][rank={rank}][stage={stage}]"]
        for key, value in payload.items():
            message.append(f"{key}={value}")
        print(" ".join(message), flush=True)

    @staticmethod
    def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
        if tensor is None:
            return None
        return list(tensor.shape)

    @staticmethod
    def _describe_optional_tensor(tensor: Optional[torch.Tensor]) -> str:
        if tensor is None:
            return "None"
        return str(list(tensor.shape))

    @staticmethod
    def _normalize_interaction_output(
        interaction_output: Any,
        interaction: torch.nn.Module,
        layer_index: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(interaction_output, tuple) or len(interaction_output) != 2:
            raise TypeError(
                "MP/MPA interaction blocks must return a 2-tuple of "
                f"(node_feats, sc); layer_index={layer_index}, "
                f"interaction_type={interaction.__class__.__name__}, "
                f"got={type(interaction_output).__name__}."
            )
        node_feats, sc = interaction_output
        if not isinstance(node_feats, torch.Tensor):
            raise TypeError(
                "MP/MPA interaction blocks must return tensor node features as the first tuple "
                f"item; layer_index={layer_index}, interaction_type={interaction.__class__.__name__}."
            )
        if sc is not None and not isinstance(sc, torch.Tensor):
            raise TypeError(
                "MP/MPA interaction blocks must return either a tensor or None for the skip "
                f"connection; layer_index={layer_index}, interaction_type={interaction.__class__.__name__}."
            )
        return node_feats, sc

    def _ensure_feature_exchanger_ready(
        self,
        node_feats: torch.Tensor,
        state: DistNNPState,
    ) -> None:
        if node_feats.dim() != 2:
            raise ValueError(
                "The MP/MPA top-down adapter expects 2-D node features after each product block."
            )
        expected_width = self._distmace_context.get("feature_width")
        feature_width = int(node_feats.shape[-1])
        if expected_width is None:
            self._distmace_context["feature_width"] = feature_width
        elif int(expected_width) != feature_width:
            raise ValueError(
                "Feature exchange width changed across interaction layers; the current MP/MPA "
                "adapter requires a stable width."
            )
        self._emit_debug_log(
            "feature_exchange_ready_check",
            feature_width=feature_width,
            expected_width=self._distmace_context.get("feature_width"),
            num_rows=int(node_feats.shape[0]),
        )

        if self._distmace_feature_exchanger.is_ipc_setup:
            self._emit_debug_log("feature_exchange_reuse", feature_width=feature_width)
            return
        partition_info = self._distmace_context.get("feature_partition_info")
        device = self._distmace_context.get("feature_device")
        if partition_info is None or device is None:
            partition_info = state.local_partition_info
            device = node_feats.device
        self._emit_debug_log(
            "feature_exchange_setup_begin",
            feature_width=feature_width,
            device=str(device),
        )
        self._distmace_feature_exchanger.setup_exchanger(
            feature_width,
            partition_info,
            device,
        )
        self._emit_debug_log("feature_exchange_setup_end", feature_width=feature_width)

    def _should_exchange_after_product(self, layer_index: int) -> bool:
        interactions = getattr(self, "interactions", ())
        return layer_index < max(0, len(interactions) - 1)

    def _begin_dist_forward(
        self, state: DistNNPState, exchange_enabled: bool, observation: Optional[Dict[str, Any]]
    ) -> None:
        self._distmace_context.update(
            state=state, exchange_enabled=exchange_enabled, observation=observation
        )
        self._emit_debug_log(
            "forward_begin",
            exchange_enabled=exchange_enabled,
            num_interactions=len(getattr(self, "interactions", ())),
            num_readouts=len(getattr(self, "readouts", ())),
        )

    def _end_dist_forward(self) -> None:
        self._emit_debug_log("forward_end")
        self._distmace_context.update(state=None, exchange_enabled=False, observation=None)

    def _exchange_after_product(
        self, node_feats: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        observation = self._distmace_context["observation"]
        if not self._distmace_context["exchange_enabled"]:
            if observation is not None:
                observation.setdefault("layer_records", []).append(
                    {
                        "layer_index": layer_index,
                        "executed": False,
                        "reason": "no_structural_halo",
                    }
                )
            self._emit_debug_log(
                "exchange_skipped",
                layer_index=layer_index,
                reason="no_structural_halo",
            )
            return node_feats
        if not self._should_exchange_after_product(layer_index):
            if observation is not None:
                observation.setdefault("layer_records", []).append(
                    {
                        "layer_index": layer_index,
                        "executed": False,
                        "reason": "last_interaction_has_no_downstream_consumer",
                    }
                )
            self._emit_debug_log(
                "exchange_skipped",
                layer_index=layer_index,
                reason="last_interaction_has_no_downstream_consumer",
            )
            return node_feats
        state = self._distmace_context["state"]
        if state is None:
            raise RuntimeError("Feature exchange reached without an active DistNNPState.")
        num_local = int(state.local_molecular_info.num_atoms)
        self._ensure_feature_exchanger_ready(node_feats, state)
        self._emit_debug_log(
            "exchange_begin",
            layer_index=layer_index,
            owned_shape=self._tensor_shape(node_feats[:num_local]),
            full_shape=self._tensor_shape(node_feats),
        )
        exchanged = FeaturePushToExchange.apply(
            node_feats[:num_local].contiguous(), state, self._distmace_feature_exchanger
        ).to(dtype=node_feats.dtype)
        if exchanged.shape[0] != node_feats.shape[0]:
            raise ValueError("Feature exchange changed the structural row count.")
        self._emit_debug_log(
            "exchange_end",
            layer_index=layer_index,
            exchanged_shape=self._tensor_shape(exchanged),
        )
        if observation is not None:
            observation["feature_exchange_used_ipc"] = True
            observation.setdefault("layer_records", []).append(
                {
                    "layer_index": layer_index,
                    "executed": True,
                    "owned_shape": list(node_feats[:num_local].shape),
                    "exchanged_shape": list(exchanged.shape),
                }
            )
        return exchanged

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
        _require_mace()
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        node_heads = ctx.node_heads.to(torch.int64)
        atom_range = ctx.num_atoms_arange.to(torch.int64)
        benchmark_mode = self._execution_mode() == "benchmark"
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[atom_range, node_heads]
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        pair_node_energy = (
            self.pair_repulsion_fn(
                ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if hasattr(self, "pair_repulsion")
            else torch.zeros_like(node_e0)
        )
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        num_readouts = len(self.readouts)
        num_layers = len(self.interactions)
        if num_readouts not in (1, num_layers):
            raise TypeError(
                "MP/MPA top-down adapter expects either one final readout or one readout per "
                f"interaction layer; got num_readouts={num_readouts}, num_layers={num_layers}."
            )

        for layer_index, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            interaction_output = interaction(
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
            node_feats, sc = self._normalize_interaction_output(
                interaction_output, interaction, layer_index
            )
            self._emit_debug_log(
                "after_interaction",
                layer_index=layer_index,
                node_feats_shape=self._tensor_shape(node_feats),
                skip_connection_shape=self._describe_optional_tensor(sc),
                skip_connection_present=sc is not None,
                interaction_type=interaction.__class__.__name__,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            self._emit_debug_log(
                "after_product",
                layer_index=layer_index,
                node_feats_shape=self._tensor_shape(node_feats),
                product_type=product.__class__.__name__,
            )
            # Communication point
            # Exchange after each product block whose outputs feed a later interaction.
            node_feats = self._exchange_after_product(node_feats, layer_index)
            node_feats_list.append(node_feats)
        for readout_index, readout in enumerate(self.readouts):
            feature_index = -1 if len(self.readouts) == 1 else readout_index
            node_es_list.append(
                readout(node_feats_list[feature_index], node_heads)[atom_range, node_heads]
            )
        node_inter_es = self.scale_shift(torch.sum(torch.stack(node_es_list), dim=0), node_heads)
        node_energy = node_e0 + node_inter_es
        state = self._distmace_context["state"]
        owned_node_energy = owned_local_energy = None
        if state is not None:
            num_local = int(state.local_molecular_info.num_atoms)
            owned_node_energy = node_energy.reshape(-1)[:num_local]
            owned_local_energy = owned_node_energy.sum()

        #earlier return since forces from MACE are redundant during benchmark
        if benchmark_mode and owned_local_energy is not None and state is not None:
            return {
                "owned_node_energy": owned_node_energy,
                "owned_local_energy": owned_local_energy,
            }

        interaction_energy = scatter_sum(
            node_inter_es, data["batch"], dim=-1, dim_size=ctx.num_graphs
        )
        e0 = scatter_sum(node_e0, data["batch"], dim=0, dim_size=ctx.num_graphs).to(ctx.vectors.dtype)
        total_energy = e0 + interaction_energy

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=interaction_energy,
            positions=ctx.positions,
            displacement=ctx.displacement,
            vectors=ctx.vectors,
            cell=ctx.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )
        atomic_virials = atomic_stresses = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=ctx.vectors,
                num_atoms=ctx.positions.shape[0],
                batch=data["batch"],
                cell=ctx.cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": interaction_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": ctx.displacement,
            "node_feats": torch.cat(node_feats_list, dim=-1),
            "owned_node_energy": owned_node_energy,
            "owned_local_energy": owned_local_energy,
        }

#-------------------------------------------------
# The DistNNP model contract facing adapter class
#---------------------------------------------

class DistMACETopDownAdapter(DistNNPModel):
    """MP/MPA-focused MACE adapter using DistNNP partitions, halos, graphs, and exchange."""

    def __init__(
        self,
        model,
        atomic_number_map: Optional[Mapping[int, int]] = None,
        diagnostics_enabled: bool = True,
        head_name: Optional[str] = None,
        debug_tracing: bool = False,
        debug_prefix: str = "distmace-debug",
    ) -> None:
        super().__init__()
        _require_mace()
        self._validate_model_contract(model)
        self.model = model.to(dtype=COMPUTE_DTYPE).eval()
        self.atomic_number_map = dict(atomic_number_map or {})
        self.cutoff = float(model.r_max.detach().cpu().item())
        
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        
        self.exchange_enabled = False
        self.diagnostics_enabled = diagnostics_enabled
        self.partition_validation: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}
        self.head_name = head_name
        self.debug_tracing = bool(debug_tracing)
        self.debug_prefix = debug_prefix
        self.execution_mode = "validate"
        self._static_graph_cache: Dict[str, Any] = {}
        self._install_runtime_subclass()

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
                "DistMACETopDownAdapter currently targets MP/MPA-style ScaleShiftMACE-compatible "
                f"models. Missing required attributes: {', '.join(missing)}."
            )
        if ScaleShiftMACE is not None and not isinstance(model, ScaleShiftMACE):
            raise TypeError(
                "DistMACETopDownAdapter currently supports MP/MPA checkpoints through the "
                "ScaleShiftMACE-compatible contract exposed by the foundation-model loaders."
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

    def set_diagnostics(self, enabled: bool) -> None:
        """Enable correctness records or the synchronization-free timed path."""
        self.diagnostics_enabled = bool(enabled)

    def set_debug_tracing(self, enabled: bool, prefix: Optional[str] = None) -> None:
        self.debug_tracing = bool(enabled)
        if prefix is not None:
            self.debug_prefix = str(prefix)
        self.model._set_debug_tracing(self.debug_tracing, self.debug_prefix)

    def set_execution_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"validate", "benchmark", "debug"}:
            raise ValueError(f"Unsupported execution mode: {mode!r}")
        self.execution_mode = normalized
        self.model._set_execution_mode(normalized)

    def _install_runtime_subclass(self) -> None:
        if not getattr(self.model.__class__, "__distmace_explicit_graph__", False):
            runtime_class = type(
                f"DistMACE{self.model.__class__.__name__}",
                (DistScaleShiftMACERuntimeMixin, self.model.__class__),
                {"__distmace_explicit_graph__": True},
            )
            self.model.__class__ = runtime_class
        self.model._configure_dist_runtime()
        self.model._set_execution_mode(self.execution_mode)
        self.model._set_debug_tracing(self.debug_tracing, self.debug_prefix)

    def _reset_static_graph_cache(self) -> None:
        self._static_graph_cache = {}

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
        self._reset_static_graph_cache()
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        #for num of layers in the model, defined in prev class
        self.model._configure_feature_exchange_runtime(partition_info, device)
        if total_send == 0 and total_recv == 0:
            self.exchange_enabled = False
            return
        if total_send == 0 or total_recv == 0:
            raise RuntimeError("Asymmetric halo traffic cannot form a local DistNNP view.")
        self.position_exchanger.setup_exchanger(3, partition_info, device)
        self.atom_type_exchanger.setup_exchanger(1, partition_info, device)
        self.exchange_enabled = True

    def _map_atomic_numbers(self, atom_types: torch.Tensor) -> torch.Tensor:
        atom_types = atom_types.to(torch.long)
        if not self.atomic_number_map:
            return atom_types
        max_atom_type = int(atom_types.max().item()) if atom_types.numel() > 0 else -1
        cache = self._static_graph_cache
        lookup = cache.get("atomic_number_lookup")
        lookup_device = cache.get("atomic_number_lookup_device")
        if (
            lookup is None
            or lookup_device != atom_types.device
            or int(lookup.shape[0]) <= max_atom_type
        ):
            max_key = max(int(key) for key in self.atomic_number_map)
            size = max(max_atom_type, max_key) + 1
            lookup = torch.full(
                (size,),
                -1,
                dtype=torch.long,
                device=atom_types.device,
            )
            for src, dst in self.atomic_number_map.items():
                lookup[int(src)] = int(dst)
            cache["atomic_number_lookup"] = lookup
            cache["atomic_number_lookup_device"] = atom_types.device
        mapped = lookup.index_select(0, atom_types.reshape(-1)).reshape(atom_types.shape)
        missing_mask = mapped.lt(0)
        if bool(missing_mask.any()):
            missing = atom_types[missing_mask].unique().cpu().tolist()
            raise ValueError(
                "MACE species compatibility check failed: unsupported atomic numbers "
                f"after remapping {missing}"
            )
        return mapped

    def _prepare_static_graph_cache(
        self,
        state: DistNNPState,
        local_types: torch.Tensor,
        local_global: torch.Tensor,
        expected_total: int,
    ) -> Dict[str, Any]:
        cache = self._static_graph_cache
        cache_key = (
            local_types.device,
            int(local_types.numel()),
            expected_total,
            tuple(int(x) for x in local_global.detach().cpu().tolist()),
            self.head_name,
            bool(self.exchange_enabled),
        )
        if cache.get("cache_key") == cache_key:
            return cache

        if self.exchange_enabled:
            with torch.no_grad():
                exchanged_types = FeaturePushToExchange.apply(
                    local_types.float().unsqueeze(-1), state, self.atom_type_exchanger
                ).squeeze(-1).long()
        else:
            exchanged_types = local_types

        mapped_atomic_numbers = self._map_atomic_numbers(exchanged_types)
        model_atomic_numbers = self.model.atomic_numbers.to(
            device=mapped_atomic_numbers.device, dtype=torch.long
        )
        node_attrs = mapped_atomic_numbers[:, None].eq(model_atomic_numbers[None, :]).to(
            dtype=COMPUTE_DTYPE
        )
        row_match_counts = node_attrs.sum(dim=1)
        if not bool(torch.all(row_match_counts == 1)):
            missing = mapped_atomic_numbers[row_match_counts == 0].unique()
            ambiguous = mapped_atomic_numbers[row_match_counts > 1].unique()
            messages = []
            if missing.numel() > 0:
                messages.append(f"unsupported atomic numbers {missing.cpu().tolist()}")
            if ambiguous.numel() > 0:
                messages.append(f"ambiguous atomic-number mappings {ambiguous.cpu().tolist()}")
            raise ValueError("MACE species compatibility check failed: " + "; ".join(messages))

        device = mapped_atomic_numbers.device
        head_index = resolve_model_head_index(self.model, self.head_name)
        cache = {
            "cache_key": cache_key,
            "exchanged_atomic_numbers": mapped_atomic_numbers,
            "node_attrs": node_attrs,
            "model_atomic_numbers": model_atomic_numbers,
            "batch": torch.zeros(expected_total, dtype=torch.long, device=device),
            "ptr": torch.tensor([0, expected_total], dtype=torch.long, device=device),
            "head": torch.tensor([head_index], dtype=torch.long, device=device),
            "device": device,
        }
        self._static_graph_cache = cache
        return cache

    @staticmethod
    def _halo_global_indices(state: DistNNPState) -> torch.Tensor:
        """Reconstruct halo global IDs in DistNNP's peer-then-global-id recv order."""
        info = state.local_partition_info
        edge_index = info.edge_index_halo
        device = state.local_molecular_info.atom_positions.device
        if edge_index is None or edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        src, dst = edge_index.to(torch.long)
        owner = info.partition_id.to(torch.long)
        rank = int(state.dist_env.global_rank)
        expected_total = int(info.recv_offsets[-1].item())
        remote = src[owner.index_select(0, dst).eq(rank) & owner.index_select(0, src).ne(rank)]
        groups = []
        for peer in range(int(state.dist_env.world_size)):
            if peer == rank:
                continue
            peer_atoms = remote[owner.index_select(0, remote).eq(peer)]
            if peer_atoms.numel() > 0:
                groups.append(torch.unique(peer_atoms.sort()[0]))
        ordered = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long, device=device)
        if ordered.numel() != expected_total:
            raise ValueError(
                "Derived halo global IDs do not match recv_offsets. The current DistNNPState does not "
                "expose halo_recv_global_ids directly, so the adapter requires edge_index_halo to imply "
                "the same peer-then-global-id ordering."
            )
        return ordered

    def _build_structural_view(self, state: DistNNPState) -> StructuralHaloView:
        info = state.local_partition_info
        if not self.partition_validation:
            self.partition_validation = validate_partition_info(
                info, int(state.dist_env.world_size)
            )
        local_positions = (
            state.local_molecular_info.atom_positions.detach()
            .to(dtype=COMPUTE_DTYPE)
            .clone()
            .requires_grad_(True)
        )
        local_types = state.local_molecular_info.atom_types.to(torch.long)
        if self.exchange_enabled:
            exchanged_positions = FeaturePushToExchange.apply(
                local_positions, state, self.position_exchanger
            )
        else:
            exchanged_positions = local_positions
        if self.diagnostics_enabled:
            exchanged_positions.retain_grad()
        local_global = info.partition_indices.long()
        expected_total = int(local_global.numel() + self.partition_validation["total_recv"])
        if exchanged_positions.shape[0] != expected_total:
            raise ValueError("Structural rows disagree with DistNNP recv metadata.")
        cache = self._prepare_static_graph_cache(state, local_types, local_global, expected_total)
        if self.diagnostics_enabled:
            halo_global = self._halo_global_indices(state)
            expanded_global = torch.cat((local_global, halo_global))
            if expanded_global.shape[0] != expected_total:
                raise ValueError("Halo indices disagree with DistNNP recv metadata.")
        else:
            halo_global = torch.empty(0, dtype=torch.long, device=local_global.device)
            expanded_global = torch.empty(0, dtype=torch.long, device=local_global.device)
        return StructuralHaloView(
            local_positions,
            exchanged_positions,
            cache["exchanged_atomic_numbers"],
            local_global,
            halo_global,
            expanded_global,
            int(local_global.numel()),
            expected_total,
        )

    def _radius_graph(
        self, positions: torch.Tensor, state: DistNNPState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = state.sim_config
        with torch.no_grad():
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                _RadiusGraphInput(positions, sim.cell, sim.pbc),
                radius=self.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )
        # DistNNP shifts the source image; MACE shifts source-to-destination vectors.
        return edge_index.long(), -cell_offsets # verify if this line is syntactically correct

    def _build_owned_graph(
        self, view: StructuralHaloView, state: DistNNPState
    ) -> ExplicitOwnedGraph:
        edge_index, unit_shifts = self._radius_graph(view.exchanged_positions, state)
        owned_destination = edge_index[1] < view.num_local
        edge_index = edge_index[:, owned_destination].contiguous()
        unit_shifts = unit_shifts[owned_destination].to(dtype=COMPUTE_DTYPE)
        cell = state.sim_config.cell.reshape(3, 3).to(dtype=COMPUTE_DTYPE)
        shifts = torch.matmul(unit_shifts, cell)
        cache = self._static_graph_cache
        batch_dict = {
            "positions": view.exchanged_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "cell": cell,
            "node_attrs": cache["node_attrs"],
            "batch": cache["batch"],
            "ptr": cache["ptr"],
            "head": cache["head"],
        }
        batch_dict = {
            key: value.to(COMPUTE_DTYPE) if torch.is_floating_point(value) else value
            for key, value in batch_dict.items()
        }
        return ExplicitOwnedGraph(edge_index, unit_shifts, shifts, batch_dict)

    @staticmethod
    def _recover_forces(
        owned_energy: torch.Tensor,
        view: StructuralHaloView,
        diagnostics_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if diagnostics_enabled:
            local_grad, exchanged_grad = torch.autograd.grad(
                owned_energy, [view.local_positions, view.exchanged_positions], allow_unused=False
            )
        else:
            (local_grad,) = torch.autograd.grad(owned_energy, [view.local_positions], allow_unused=False)
            exchanged_grad = None
        forces = -local_grad
        checks = {}
        if diagnostics_enabled:
            checks = {
                "local_grad_is_finite": bool(torch.isfinite(local_grad).all()),
                "exchanged_grad_is_finite": bool(torch.isfinite(exchanged_grad).all()),
                "recovered_force_shape": list(forces.shape),
            }
        return forces, checks

    def _expected_exchange_layers(self) -> List[int]:
        return [
            layer_index
            for layer_index in range(len(getattr(self.model, "interactions", ())))
            if self.model._should_exchange_after_product(layer_index)
        ]

    def _build_observation(
        self,
        view: StructuralHaloView,
        graph: ExplicitOwnedGraph,
        state: DistNNPState,
    ) -> Optional[Dict[str, Any]]:
        if not self.diagnostics_enabled:
            return None
        verification = build_graph_authority_record(
            view=view,
            local_graph=graph,
            state=state,
            radius_graph_fn=self._radius_graph,
            edge_records_fn=_edge_records,
        )
        return make_adapter_observation(
            self.model,
            state,
            graph_verification=verification,
            expected_exchange_layers=self._expected_exchange_layers(),
        )

    def _execute_model(
        self,
        view: StructuralHaloView,
        graph: ExplicitOwnedGraph,
        state: DistNNPState,
        observation: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self.model._begin_dist_forward(state, self.exchange_enabled, observation)
        try:
            outputs = self.model(
                graph.batch_dict,
                training=False,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
            owned_energy = outputs.get("owned_local_energy")
            if owned_energy is None:
                raise KeyError("Distributed MACE forward did not return owned_local_energy.")
            if self.model._debug_enabled():
                self.model._emit_debug_log(
                    "owned_energy_ready",
                    owned_energy=float(owned_energy.detach().cpu()),
                )
            owned_forces, force_checks = self._recover_forces(
                owned_energy, view, self.diagnostics_enabled
            )
            if self.model._debug_enabled():
                self.model._emit_debug_log(
                    "force_recovery_end",
                    returned_force_shape=list(owned_forces.shape),
                )
            return outputs, owned_energy, owned_forces, force_checks
        except Exception as exc:
            self.model._emit_debug_log(
                "adapter_exception",
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            if observation is not None:
                observation["adapter_exception"] = {
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            raise
        finally:
            self.model._end_dist_forward()

    def __call__(self, state: DistNNPState, *args, **kwargs):
        view = self._build_structural_view(state)
        graph = self._build_owned_graph(view, state)
        observation = self._build_observation(view, graph, state)
        self.last_observation = observation or {}
        outputs, owned_energy, owned_forces, force_checks = self._execute_model(
            view, graph, state, observation
        )
        if self.diagnostics_enabled:
            finalize_adapter_observation(
                self.last_observation,
                self.partition_validation,
                view,
                graph,
                outputs,
                owned_energy,
                owned_forces,
                force_checks,
                model_parameter_dtype=str(next(self.model.parameters()).dtype),
            )
        return owned_energy.detach(), owned_forces.detach()


__all__ = [
    "COMPUTE_DTYPE",
    "DistMACETopDownAdapter",
    "canonicalize_nonperiodic_geometry",
    "load_raw_mace_model",
    "resolve_model_head_index",
    "resolve_model_head_name",
    "summarize_model_contract",
    "validate_partition_info",
]
