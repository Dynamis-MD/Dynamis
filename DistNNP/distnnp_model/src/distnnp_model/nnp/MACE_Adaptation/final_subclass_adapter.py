"""Full-model subclass distributed MACE adaptation for DistNNP.

This module is the end-state oriented successor to the earlier phase files.
It keeps the structural halo reconstruction path from Phase 1, but moves
learned-feature communication into a model-level subclass of MACE itself so
the distributed schedule is owned by the MACE forward loop rather than by a
single interaction-block hook.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.nnp.MACE_Adaptation.phase1_raw_exchange_adapter import (
    MACEPotentialModelWithRawExchange,
    align_batch_dtype,
    analyze_structure,
    build_global_molecular_info,
    build_parser as build_phase1_parser,
    build_phase1_distnnp_state,
    build_sim_config,
    finalize_distributed,
    load_raw_mace_model,
    maybe_gather_records,
    parse_raw_structure,
    validate_phase1_state,
)
from distnnp_state import DistNNPState, SFCCurveType, SFCPartitioner, TorchDistEnv

try:
    from mace.modules import (
        ScaleShiftMACE,
    )
    from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
    from mace.tools.scatter import scatter_sum

    _MACE_MODEL_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    ScaleShiftMACE = None  # type: ignore[assignment]
    get_atomic_virials_stresses = None  # type: ignore[assignment]
    get_outputs = None  # type: ignore[assignment]
    prepare_graph = None  # type: ignore[assignment]
    scatter_sum = None  # type: ignore[assignment]
    _MACE_MODEL_IMPORT_ERROR = exc


def _require_mace_model_ops() -> None:
    if _MACE_MODEL_IMPORT_ERROR is not None:
        raise ImportError(
            "The full-model subclass adapter requires public MACE model symbols "
            "from 'mace.modules.models'."
        ) from _MACE_MODEL_IMPORT_ERROR


@dataclass(frozen=True)
class FeatureExchangeSchedule:
    """Policy for where learned-feature communication occurs in MACE."""

    name: str = "all_noninitial_layers_post_product"
    exchange_after_product_layers: Optional[Sequence[int]] = None
    exchange_before_interaction_layers: Sequence[int] = ()

    def resolve_layers(self, num_interactions: int) -> tuple[int, ...]:
        if num_interactions <= 0:
            return ()
        if self.exchange_after_product_layers is None:
            return tuple(range(1, num_interactions))
        resolved = tuple(sorted({int(layer) for layer in self.exchange_after_product_layers}))
        for layer in resolved:
            if layer < 0 or layer >= num_interactions:
                raise ValueError(
                    f"exchange_after_product_layers contains out-of-range layer {layer} "
                    f"for num_interactions={num_interactions}."
                )
        return resolved

    def resolve_pre_interaction_layers(self, num_interactions: int) -> tuple[int, ...]:
        if num_interactions <= 0:
            return ()
        resolved = tuple(
            sorted({int(layer) for layer in self.exchange_before_interaction_layers})
        )
        for layer in resolved:
            if layer < 0 or layer >= num_interactions:
                raise ValueError(
                    f"exchange_before_interaction_layers contains out-of-range layer "
                    f"{layer} for num_interactions={num_interactions}."
                )
        return resolved

    def should_exchange_after_product(self, layer_index: int, num_interactions: int) -> bool:
        return layer_index in self.resolve_layers(num_interactions)

    def should_exchange_before_interaction(
        self, layer_index: int, num_interactions: int
    ) -> bool:
        return layer_index in self.resolve_pre_interaction_layers(num_interactions)


@dataclass
class DistMACEForwardContext:
    distributed_state: Optional[DistNNPState] = None
    exchange_mode: str = "uninitialized"
    observation: Optional[Dict[str, Any]] = None


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


class StructuralHaloBuilder:
    """Build the owned-local plus halo structural tensors for the adapter."""

    def build(
        self,
        adapter: "DistMACEFullSubclassAdapter",
        distributed_state: DistNNPState,
    ) -> StructuralHaloView:
        validation = validate_phase1_state(distributed_state)
        adapter.partition_validation = validation
        adapter.total_send = validation["total_send"]
        adapter.total_recv = validation["total_recv"]

        local_positions = (
            distributed_state.local_molecular_info.atom_positions.clone()
            .detach()
            .requires_grad_(True)
        )
        local_atom_types = distributed_state.local_molecular_info.atom_types

        exchanged_positions, exchanged_atom_types = adapter._expand_local_plus_halo_view(
            local_positions,
            local_atom_types,
            distributed_state,
        )
        exchanged_positions.retain_grad()
        exchanged_atomic_numbers = adapter._map_atomic_numbers(exchanged_atom_types)
        local_global_indices = (
            distributed_state.local_partition_info.partition_indices.to(torch.long).detach()
        )
        halo_global_indices = adapter._derive_halo_global_indices(distributed_state).to(
            torch.long
        )
        expanded_global_indices = torch.cat(
            [local_global_indices, halo_global_indices],
            dim=0,
        )

        num_local = int(local_positions.shape[0])
        num_total = int(exchanged_positions.shape[0])
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


class MACEGraphBuilder:
    """Build a differentiable public-MACE batch dictionary."""

    def build(
        self,
        adapter: "DistMACEFullSubclassAdapter",
        structural_view: StructuralHaloView,
        distributed_state: DistNNPState,
    ) -> tuple[Dict[str, torch.Tensor], int, torch.Tensor]:
        config = adapter._build_configuration(
            structural_view.exchanged_atomic_numbers,
            structural_view.exchanged_positions,
            distributed_state,
        )
        atomic_data = adapter.AtomicData.from_config(  # type: ignore[attr-defined]
            config=config,
            z_table=adapter.z_table,
            cutoff=adapter.cutoff,
            heads=adapter.heads,
        )
        edge_count = int(atomic_data.edge_index.shape[1])

        batch = adapter.Batch.from_data_list([atomic_data]).to(  # type: ignore[attr-defined]
            structural_view.exchanged_positions.device
        )
        batch_dict = batch.to_dict()
        batch_dict["positions"] = structural_view.exchanged_positions
        batch_dict = align_batch_dtype(batch_dict, adapter.model)
        return batch_dict, edge_count, atomic_data.edge_index.detach().clone()


class OwnedForceRecoverer:
    """Recover owned local forces from the subclass-owned scalar energy."""

    rule = "owned_local_forces = -grad(outputs['owned_local_energy'], local_positions)"

    def recover(
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
        halo_gradient_active = (
            bool(halo_grad_norm > 0.0) if structural_view.num_halo > 0 else False
        )
        return owned_local_forces, {
            "recovered_force_shape": list(owned_local_forces.shape),
            "local_grad_is_finite": local_grad_is_finite,
            "exchanged_grad_is_finite": exchanged_grad_is_finite,
            "local_grad_norm": local_grad_norm,
            "halo_grad_norm": halo_grad_norm,
            "local_force_norm": local_force_norm,
            "halo_gradient_active": halo_gradient_active,
            "backward_depends_on_halo": halo_gradient_active,
        }


class DistScaleShiftMACERuntimeMixin:
    """Model-level distributed schedule mixed into a loaded ScaleShiftMACE instance."""

    __distmace_full_subclass__ = True

    def _configure_dist_runtime(
        self,
        schedule: FeatureExchangeSchedule,
    ) -> None:
        num_interactions = int(self.num_interactions.detach().cpu().item())
        self._distmace_schedule = schedule
        self._distmace_schedule.resolve_layers(num_interactions)
        self._distmace_schedule.resolve_pre_interaction_layers(num_interactions)
        self._distmace_context = DistMACEForwardContext()
        self._distmace_feature_exchangers = [
            GPUDirectExchanger() for _ in range(num_interactions)
        ]

    def _begin_dist_forward(
        self,
        distributed_state: DistNNPState,
        exchange_mode: str,
        observation: Dict[str, Any],
    ) -> None:
        self._distmace_context.distributed_state = distributed_state
        self._distmace_context.exchange_mode = exchange_mode
        self._distmace_context.observation = observation

    def _end_dist_forward(self) -> None:
        self._distmace_context.distributed_state = None
        self._distmace_context.exchange_mode = "uninitialized"
        self._distmace_context.observation = None

    def _append_layer_record(self, record: Dict[str, Any]) -> None:
        observation = self._distmace_context.observation
        if observation is None:
            return
        observation.setdefault("layer_records", []).append(record)

    def _exchange_features(
        self,
        node_feats: torch.Tensor,
        layer_index: int,
        boundary: str,
    ) -> torch.Tensor:
        context = self._distmace_context
        if context.exchange_mode != "ipc":
            self._append_layer_record(
                {
                    "layer_index": layer_index,
                    "boundary": boundary,
                    "executed": False,
                    "reason": f"exchange_mode={context.exchange_mode}",
                    "node_feature_shape": list(node_feats.shape),
                }
            )
            return node_feats

        if context.distributed_state is None:
            raise RuntimeError(
                "Distributed MACE forward reached a scheduled exchange without an "
                "active DistNNPState."
            )

        distributed_state = context.distributed_state
        num_local = int(
            distributed_state.local_partition_info.partition_indices.shape[0]
        )
        expected_total_rows = int(
            num_local + distributed_state.local_partition_info.recv_offsets[-1].item()
        )
        input_dtype = node_feats.dtype
        local_node_feats = node_feats[:num_local].contiguous()
        exchanger = self._distmace_feature_exchangers[layer_index]
        exchanged_node_feats = FeaturePushToExchange.apply(
            local_node_feats,
            distributed_state,
            exchanger,
        )
        if exchanged_node_feats.dtype != input_dtype:
            exchanged_node_feats = exchanged_node_feats.to(dtype=input_dtype)

        self._append_layer_record(
            {
                "layer_index": layer_index,
                "boundary": boundary,
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
        observation = context.observation
        if observation is not None:
            observation["feature_exchange_used_ipc"] = True
            observation["executed_exchange_boundaries"] = observation.get(
                "executed_exchange_boundaries",
                [],
            ) + [f"{boundary}:{layer_index}"]
            observation["executed_exchange_layers"] = observation.get(
                "executed_exchange_layers",
                [],
            ) + [layer_index]
        return exchanged_node_feats

    def _maybe_exchange_before_interaction(
        self,
        node_feats: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        num_interactions = int(self.num_interactions.detach().cpu().item())
        schedule = self._distmace_schedule
        should_exchange = schedule.should_exchange_before_interaction(
            layer_index,
            num_interactions,
        )
        if not should_exchange:
            self._append_layer_record(
                {
                    "layer_index": layer_index,
                    "boundary": "pre_interaction",
                    "executed": False,
                    "reason": "schedule_skipped_layer",
                    "node_feature_shape": list(node_feats.shape),
                }
            )
            return node_feats

        return self._exchange_features(
            node_feats=node_feats,
            layer_index=layer_index,
            boundary="pre_interaction",
        )

    #Key interaction function
    def _maybe_exchange_after_product(
        self,
        node_feats: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        #number of interactions here corresponds to the body order
        num_interactions = int(self.num_interactions.detach().cpu().item())
        schedule = self._distmace_schedule
        should_exchange = schedule.should_exchange_after_product(
            layer_index,
            num_interactions,
        )
        if not should_exchange:
            self._append_layer_record(
                {
                    "layer_index": layer_index,
                    "boundary": "post_product_pre_next_interaction",
                    "executed": False,
                    "reason": "schedule_skipped_layer",
                    "node_feature_shape": list(node_feats.shape),
                }
            )
            return node_feats

        return self._exchange_features(
            node_feats=node_feats,
            layer_index=layer_index,
            boundary="post_product_pre_next_interaction",
        )

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
        observation = self._distmace_context.observation
        if observation is not None:
            observation["model_parameter_dtype"] = str(next(self.parameters()).dtype)
            observation["batch_positions_dtype"] = str(data["positions"].dtype)
            observation["graph_vectors_dtype"] = str(vectors.dtype)
            observation["graph_lengths_dtype"] = str(lengths.dtype)
            observation["node_embedding_dtype"] = str(node_feats.dtype)
            observation["edge_attr_dtype"] = str(edge_attrs.dtype)
            observation["edge_feat_dtype"] = str(edge_feats.dtype)
            observation["num_interactions"] = int(
                self.num_interactions.detach().cpu().item()
            )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = torch.atleast_1d(
                    self.embedding_readout(node_feats, node_heads)[
                        num_atoms_arange, node_heads
                    ].squeeze(-1)
                )
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        # Heart of the interaction
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_feats = self._maybe_exchange_before_interaction(node_feats, i)
            node_attrs_slice = data["node_attrs"]
            #The 0th layer has some special reqs
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
            node_feats = self._maybe_exchange_after_product(node_feats, i)
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
        distributed_state = self._distmace_context.distributed_state

        #significance of this block
        if distributed_state is not None:
            num_local = int(
                distributed_state.local_partition_info.partition_indices.shape[0]
            )
            owned_node_energy = node_energy.reshape(-1)[:num_local]
            owned_local_energy = owned_node_energy.sum()
            if observation is not None:
                observation["ownership_boundary"] = "model_internal_node_energy"
                observation["ownership_rule"] = (
                    "owned_local_energy = node_energy[:num_local].sum() inside "
                    "DistScaleShiftMACE.forward"
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


#Wrapper around prev class
class DistMACEModelLoopBoundary:
    """Install a model-level MACE subclass that owns the distributed schedule."""

    model_contract = "full_model_subclass_mace_execution_plan"
    model_status = "installed"

    def __init__(self, schedule: FeatureExchangeSchedule) -> None:
        self.schedule = schedule
        self._installed_model_class_name: Optional[str] = None
        self._resolved_layers: tuple[int, ...] = ()
        self._resolved_pre_interaction_layers: tuple[int, ...] = ()
        self._num_interactions: Optional[int] = None

    def install(self, model) -> None:
        _require_mace_model_ops()
        if ScaleShiftMACE is None or not isinstance(model, ScaleShiftMACE):
            raise TypeError(
                "The full-model subclass adapter currently targets public "
                "ScaleShiftMACE instances returned by MACE foundation-model loaders."
            )

        num_interactions = int(model.num_interactions.detach().cpu().item())
        self._num_interactions = num_interactions
        self._resolved_layers = self.schedule.resolve_layers(num_interactions)
        self._resolved_pre_interaction_layers = (
            self.schedule.resolve_pre_interaction_layers(num_interactions)
        )
        base_cls = model.__class__

        if not getattr(base_cls, "__distmace_full_subclass__", False):
            subclass_name = f"DistMACE{base_cls.__name__}"
            subclass_cls = type(
                subclass_name,
                (DistScaleShiftMACERuntimeMixin, base_cls),
                {
                    "__distmace_full_subclass__": True,
                    "__distmace_subclass_role__": "full_model_loop",
                },
            )
            model.__class__ = subclass_cls

        model._configure_dist_runtime(self.schedule)
        self._installed_model_class_name = model.__class__.__name__

    def begin_forward(
        self,
        model,
        distributed_state: DistNNPState,
        exchange_mode: str,
        observation: Dict[str, Any],
    ) -> None:
        model._begin_dist_forward(distributed_state, exchange_mode, observation)

    def end_forward(self, model) -> None:
        model._end_dist_forward()

    def describe(self) -> Dict[str, Any]:
        return {
            "model_subclass_contract": self.model_contract,
            "model_subclass_status": self.model_status,
            "installed_model_class_name": self._installed_model_class_name,
            "num_interactions": self._num_interactions,
            "exchange_schedule_name": self.schedule.name,
            "exchange_layers_before_interaction": list(
                self._resolved_pre_interaction_layers
            ),
            "exchange_layers_after_product": list(self._resolved_layers),
            "subclass_boundary_installed": self._installed_model_class_name is not None,
        }


class ValidationDiagnostics:
    """Centralized reporting surface for the full-model subclass adapter."""

    adapter_contract_name = "dist_mace_full_model_subclass_adapter"

    def build_record(
        self,
        adapter: "DistMACEFullSubclassAdapter",
        distributed_state: DistNNPState,
        structural_view: StructuralHaloView,
        edge_count: int,
        edge_index: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        owned_local_energy: torch.Tensor,
        force_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        validation = adapter.partition_validation or {}
        expanded_rank_graph_energy = outputs["energy"]
        if (
            torch.is_tensor(expanded_rank_graph_energy)
            and expanded_rank_graph_energy.numel() == 1
        ):
            expanded_rank_graph_energy = expanded_rank_graph_energy.reshape(())
        owned_node_energy = outputs["owned_node_energy"].reshape(-1)
        global_edge_pairs = edge_index_to_global_pairs(
            edge_index,
            structural_view.expanded_global_indices.to(edge_index.device),
        )
        owned_destination_global_edge_pairs = filter_owned_destination_pairs(
            global_edge_pairs,
            structural_view.local_global_indices.detach().cpu().tolist(),
        )

        return {
            "rank": int(distributed_state.dist_env.global_rank),
            "adapter_contract": self.adapter_contract_name,
            "adapter_target": adapter.adapter_target,
            "adapter_status": adapter.adapter_status,
            "n_local": structural_view.num_local,
            "n_total_after_exchange": structural_view.num_total,
            "n_halo": structural_view.num_halo,
            "expected_total_recv": adapter.total_recv,
            "exchange_mode": adapter.exchange_mode,
            "exchange_reason": adapter.exchange_reason,
            "total_send": adapter.total_send,
            "total_recv": adapter.total_recv,
            "send_counts_by_peer": validation.get("send_counts_by_peer", []),
            "recv_counts_by_peer": validation.get("recv_counts_by_peer", []),
            "edge_count_after_atomic_data": edge_count,
            "feature_exchange_used_ipc": adapter.last_observation.get(
                "feature_exchange_used_ipc",
                False,
            ),
            "executed_exchange_boundaries": adapter.last_observation.get(
                "executed_exchange_boundaries",
                [],
            ),
            "executed_exchange_layers": adapter.last_observation.get(
                "executed_exchange_layers",
                [],
            ),
            "layer_records": adapter.last_observation.get("layer_records", []),
            "global_owned_indices": structural_view.local_global_indices.detach()
            .cpu()
            .tolist(),
            "halo_global_indices": structural_view.halo_global_indices.detach()
            .cpu()
            .tolist(),
            "expanded_global_indices": structural_view.expanded_global_indices.detach()
            .cpu()
            .tolist(),
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
            "ownership_boundary": adapter.last_observation.get("ownership_boundary"),
            "ownership_rule": adapter.last_observation.get("ownership_rule"),
            "force_rule": adapter.owned_force_recoverer.rule,
            "node_energy_shape": adapter.last_observation.get("node_energy_shape"),
            "owned_node_energy_shape": adapter.last_observation.get(
                "owned_node_energy_shape"
            ),
            "owned_node_energy_values": owned_node_energy.detach().cpu().tolist(),
            "global_edge_pairs": global_edge_pairs,
            "owned_destination_global_edge_pairs": owned_destination_global_edge_pairs,
            "owned_destination_edge_count": len(owned_destination_global_edge_pairs),
            "expanded_rank_graph_energy": float(
                expanded_rank_graph_energy.detach().cpu().item()
            ),
            "owned_local_energy": float(owned_local_energy.detach().cpu().item()),
            "expanded_minus_owned_energy": float(
                (expanded_rank_graph_energy - owned_local_energy).detach().cpu().item()
            ),
            "model_parameter_dtype": adapter.last_observation.get("model_parameter_dtype"),
            "batch_positions_dtype": adapter.last_observation.get("batch_positions_dtype"),
            "graph_vectors_dtype": adapter.last_observation.get("graph_vectors_dtype"),
            "graph_lengths_dtype": adapter.last_observation.get("graph_lengths_dtype"),
            "node_embedding_dtype": adapter.last_observation.get("node_embedding_dtype"),
            "edge_attr_dtype": adapter.last_observation.get("edge_attr_dtype"),
            "edge_feat_dtype": adapter.last_observation.get("edge_feat_dtype"),
            "node_energy_dtype": adapter.last_observation.get("node_energy_dtype"),
            "total_energy_dtype": adapter.last_observation.get("total_energy_dtype"),
            "owned_local_energy_dtype": adapter.last_observation.get(
                "owned_local_energy_dtype"
            ),
            **adapter.model_loop_boundary.describe(),
            **force_diagnostics,
        }


class DistMACEFullSubclassAdapter(MACEPotentialModelWithRawExchange):
    """DistNNP-facing adapter whose MACE model owns the communication schedule."""

    adapter_target = (
        "structural_halo + full_model_subclass_exchange_schedule + "
        "internal_owned_energy + autograd_owned_local_forces"
    )
    adapter_status = "full_model_subclass_prototype_implemented"

    def __init__(
        self,
        model,
        schedule: FeatureExchangeSchedule | None = None,
    ) -> None:
        super().__init__(model)
        self.feature_exchange_schedule = schedule or FeatureExchangeSchedule()
        self.model_loop_boundary = DistMACEModelLoopBoundary(
            self.feature_exchange_schedule
        )
        self.model_loop_boundary.install(self.model)
        self.structural_halo_builder = StructuralHaloBuilder()
        self.mace_graph_builder = MACEGraphBuilder()
        self.owned_force_recoverer = OwnedForceRecoverer()
        self.validation_diagnostics = ValidationDiagnostics()

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        self.last_observation = {
            "feature_exchange_used_ipc": False,
            "executed_exchange_layers": [],
            "executed_exchange_boundaries": [],
            "layer_records": [],
        }
        self.model_loop_boundary.begin_forward(
            self.model,
            distributed_state=distributed_state,
            exchange_mode=self.exchange_mode,
            observation=self.last_observation,
        )
        try:
            structural_view = self.structural_halo_builder.build(self, distributed_state)
            batch_dict, edge_count, edge_index = self.mace_graph_builder.build(
                self,
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
                    "The distributed MACE subclass did not return 'owned_local_energy'."
                )
            owned_local_forces, force_diagnostics = self.owned_force_recoverer.recover(
                owned_local_energy,
                structural_view,
            )
        finally:
            self.model_loop_boundary.end_forward(self.model)

        self.last_observation.update(
            self.validation_diagnostics.build_record(
                adapter=self,
                distributed_state=distributed_state,
                structural_view=structural_view,
                edge_count=edge_count,
                edge_index=edge_index,
                outputs=outputs,
                owned_local_energy=owned_local_energy,
                force_diagnostics=force_diagnostics,
            )
        )
        return owned_local_energy.detach(), owned_local_forces.detach()


from distnnp_model.nnp.MACE_Adaptation import phase1_raw_exchange_adapter as _phase1

DistMACEFullSubclassAdapter.AtomicData = _phase1.AtomicData
DistMACEFullSubclassAdapter.Batch = _phase1.Batch


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1_parser()
    parser.set_defaults(model="medium-mpa-0")
    parser.description = (
        "Full-model subclass DistMACE adapter: structural halo preparation in "
        "the adapter, but learned-feature exchange and owned-energy definition "
        "inside a subclassed ScaleShiftMACE forward loop."
    )
    parser.add_argument(
        "--exchange-layers",
        default="all_noninitial",
        help=(
            "Exchange policy. Supported presets: 'all_noninitial', "
            "'after_layer0', 'before_interaction0'. You can also pass a "
            "comma-separated list of zero-based interaction layers to exchange "
            "after their product block."
        ),
    )
    parser.add_argument(
        "--comparison-model",
        default="",
        help=(
            "Optional override for the exchange-off comparison run and the "
            "monolithic parity reference. Leave empty to reuse --model."
        ),
    )
    return parser


def parse_exchange_schedule(raw_value: str) -> FeatureExchangeSchedule:
    value = raw_value.strip().lower()
    if value == "all_noninitial":
        return FeatureExchangeSchedule()
    if value == "after_layer0":
        return FeatureExchangeSchedule(
            name="post_product_layer0_only",
            exchange_after_product_layers=(0,),
        )
    if value == "before_interaction0":
        return FeatureExchangeSchedule(
            name="pre_interaction_layer0_only",
            exchange_after_product_layers=(),
            exchange_before_interaction_layers=(0,),
        )
    if not value:
        raise ValueError("--exchange-layers cannot be empty.")
    layers = [int(token.strip()) for token in raw_value.split(",") if token.strip()]
    return FeatureExchangeSchedule(
        name=f"explicit_post_product_layers_{'_'.join(str(layer) for layer in layers)}",
        exchange_after_product_layers=layers,
    )


def resolve_comparison_model(args: argparse.Namespace) -> str:
    return args.comparison_model.strip() or args.model


def edge_index_to_global_pairs(
    edge_index: torch.Tensor,
    expanded_global_indices: torch.Tensor,
) -> list[list[int]]:
    src = expanded_global_indices.index_select(0, edge_index[0].to(torch.long))
    dst = expanded_global_indices.index_select(0, edge_index[1].to(torch.long))
    pairs = torch.stack([src, dst], dim=1).detach().cpu().tolist()
    return sorted([list(map(int, pair)) for pair in pairs])


def filter_owned_destination_pairs(
    edge_pairs: list[list[int]],
    owned_global_indices: list[int],
) -> list[list[int]]:
    owned_set = set(int(index) for index in owned_global_indices)
    return [pair for pair in edge_pairs if int(pair[1]) in owned_set]


def incoming_neighbor_count_map(edge_pairs: list[list[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, dst in edge_pairs:
        counts[int(dst)] = counts.get(int(dst), 0) + 1
    return counts


def build_monolithic_reference(
    raw_structure: Dict[str, Any],
    sim_config,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
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
    config = _phase1.Configuration(
        atomic_numbers=atomic_numbers.detach().cpu().numpy(),
        positions=positions.detach().cpu().to(torch.float64).numpy(),
        properties={},
        property_weights={},
        cell=_phase1.MACEPotentialModelWithRawExchange._cell_to_numpy(sim_config.cell),
        pbc=_phase1.MACEPotentialModelWithRawExchange._pbc_to_tuple(sim_config.pbc),
        weight=1.0,
        config_type="DistMACE_Monolithic_Parity",
        head="Default",
    )
    z_table = _phase1.AtomicNumberTable(
        [int(z) for z in reference_model.atomic_numbers.detach().cpu().tolist()]
    )
    heads = getattr(reference_model, "heads", ["Default"])
    cutoff = float(reference_model.r_max.detach().cpu().item())
    atomic_data = _phase1.AtomicData.from_config(
        config=config,
        z_table=z_table,
        cutoff=cutoff,
        heads=heads,
    )
    batch = _phase1.Batch.from_data_list([atomic_data]).to(device)
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
    forces = outputs["forces"]
    node_energy = outputs["node_energy"]
    edge_pairs = edge_index_to_global_pairs(
        atomic_data.edge_index.detach(),
        torch.arange(len(raw_structure["atomic_numbers"]), device=atomic_data.edge_index.device),
    )
    return {
        "reference_model": resolve_comparison_model(args),
        "num_interactions": int(reference_model.num_interactions.detach().cpu().item()),
        "total_energy": float(total_energy.detach().cpu().item()),
        "forces": forces.detach().cpu().tolist(),
        "force_shape": list(forces.shape),
        "node_energy": node_energy.reshape(-1).detach().cpu().tolist(),
        "node_energy_shape": list(node_energy.shape),
        "global_edge_pairs": edge_pairs,
        "edge_count": len(edge_pairs),
        "model_parameter_dtype": str(next(reference_model.parameters()).dtype),
        "batch_positions_dtype": str(batch_dict["positions"].dtype),
        "node_energy_dtype": str(node_energy.dtype),
        "total_energy_dtype": str(total_energy.dtype),
    }


def assemble_global_owned_forces(
    rank_records: List[Dict[str, Any]],
    num_atoms: int,
    force_key: str,
    index_key: str,
) -> torch.Tensor:
    global_forces = torch.zeros(num_atoms, 3, dtype=torch.float64)
    for record in rank_records:
        owned_indices = torch.tensor(record[index_key], dtype=torch.long)
        owned_forces = torch.tensor(record[force_key], dtype=torch.float64)
        if owned_forces.shape[0] != owned_indices.numel():
            raise ValueError(
                f"Owned force rows {owned_forces.shape[0]} did not match owned index count "
                f"{owned_indices.numel()}."
            )
        global_forces[owned_indices] = owned_forces
    return global_forces


def assemble_global_owned_scalars(
    rank_records: List[Dict[str, Any]],
    num_atoms: int,
    value_key: str,
    index_key: str,
) -> torch.Tensor:
    global_values = torch.zeros(num_atoms, dtype=torch.float64)
    for record in rank_records:
        owned_indices = torch.tensor(record[index_key], dtype=torch.long)
        owned_values = torch.tensor(record[value_key], dtype=torch.float64)
        if owned_values.shape[0] != owned_indices.numel():
            raise ValueError(
                f"Owned scalar rows {owned_values.shape[0]} did not match owned index count "
                f"{owned_indices.numel()}."
            )
        global_values[owned_indices] = owned_values
    return global_values


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: List[Dict[str, Any]],
    monolithic_reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    has_halo = any(record["n_halo"] > 0 for record in rank_records_sorted)
    num_atoms = len(raw_structure["atomic_numbers"])
    distributed_owned_forces = assemble_global_owned_forces(
        rank_records_sorted,
        num_atoms=num_atoms,
        force_key="returned_forces",
        index_key="global_owned_indices",
    )
    comparison_owned_forces = assemble_global_owned_forces(
        rank_records_sorted,
        num_atoms=num_atoms,
        force_key="comparison_returned_forces",
        index_key="global_owned_indices",
    )
    distributed_owned_node_energy = assemble_global_owned_scalars(
        rank_records_sorted,
        num_atoms=num_atoms,
        value_key="owned_node_energy_values",
        index_key="global_owned_indices",
    )
    exchange_force_delta = distributed_owned_forces - comparison_owned_forces
    monolithic_parity: Optional[Dict[str, Any]] = None
    graph_parity_records: List[Dict[str, Any]] = []
    global_graph_parity_exact = True
    max_owned_neighbor_count_mismatch = 0
    if monolithic_reference is not None:
        monolithic_force_tensor = torch.tensor(
            monolithic_reference["forces"],
            dtype=torch.float64,
        )
        monolithic_node_energy_tensor = torch.tensor(
            monolithic_reference["node_energy"],
            dtype=torch.float64,
        )
        force_error = distributed_owned_forces - monolithic_force_tensor
        node_energy_error = distributed_owned_node_energy - monolithic_node_energy_tensor
        monolithic_global_edge_pairs = monolithic_reference["global_edge_pairs"]
        distributed_owned_energy_sum = float(
            sum(record["owned_local_energy"] for record in rank_records_sorted)
        )
        for record in rank_records_sorted:
            owned_indices = record["global_owned_indices"]
            monolithic_owned_edges = filter_owned_destination_pairs(
                monolithic_global_edge_pairs,
                owned_indices,
            )
            distributed_owned_edges = record["owned_destination_global_edge_pairs"]
            exact_match = monolithic_owned_edges == distributed_owned_edges
            global_graph_parity_exact = global_graph_parity_exact and exact_match
            distributed_neighbor_counts = incoming_neighbor_count_map(distributed_owned_edges)
            monolithic_neighbor_counts = incoming_neighbor_count_map(monolithic_owned_edges)
            per_owned_atom_neighbor_delta: Dict[int, int] = {}
            for owned_index in owned_indices:
                delta = distributed_neighbor_counts.get(int(owned_index), 0) - monolithic_neighbor_counts.get(int(owned_index), 0)
                per_owned_atom_neighbor_delta[int(owned_index)] = delta
                max_owned_neighbor_count_mismatch = max(
                    max_owned_neighbor_count_mismatch,
                    abs(delta),
                )
            graph_parity_records.append(
                {
                    "rank": record["rank"],
                    "owned_indices": owned_indices,
                    "distributed_owned_destination_edge_count": len(distributed_owned_edges),
                    "monolithic_owned_destination_edge_count": len(monolithic_owned_edges),
                    "owned_destination_edge_sets_exact_match": exact_match,
                    "per_owned_atom_neighbor_delta": per_owned_atom_neighbor_delta,
                }
            )
        monolithic_parity = {
            "reference_model": monolithic_reference["reference_model"],
            "reference_num_interactions": monolithic_reference["num_interactions"],
            "monolithic_total_energy": monolithic_reference["total_energy"],
            "distributed_owned_energy_sum": distributed_owned_energy_sum,
            "absolute_energy_error": abs(
                distributed_owned_energy_sum - monolithic_reference["total_energy"]
            ),
            "distributed_force_shape": list(distributed_owned_forces.shape),
            "monolithic_force_shape": monolithic_reference["force_shape"],
            "max_abs_force_error": float(force_error.abs().max().item()),
            "l2_force_error": float(force_error.norm().item()),
            "max_abs_node_energy_error": float(node_energy_error.abs().max().item()),
            "l2_node_energy_error": float(node_energy_error.norm().item()),
            "owned_node_energy_exact_match_atol_1e-10": bool(
                torch.allclose(
                    distributed_owned_node_energy,
                    monolithic_node_energy_tensor,
                    atol=1e-10,
                    rtol=0.0,
                )
            ),
            "global_graph_parity_exact": global_graph_parity_exact,
            "max_owned_neighbor_count_mismatch": max_owned_neighbor_count_mismatch,
            "graph_parity_records": graph_parity_records,
            "monolithic_model_parameter_dtype": monolithic_reference[
                "model_parameter_dtype"
            ],
            "monolithic_batch_positions_dtype": monolithic_reference[
                "batch_positions_dtype"
            ],
            "monolithic_node_energy_dtype": monolithic_reference["node_energy_dtype"],
            "monolithic_total_energy_dtype": monolithic_reference[
                "total_energy_dtype"
            ],
        }
    checks = {
        "all_expected_halo_counts_match": all(
            record["n_halo"] == record["expected_total_recv"]
            for record in rank_records_sorted
        ),
        "all_recovered_force_shapes_match_owned_atoms": all(
            record["recovered_force_shape"] == [record["n_local"], 3]
            for record in rank_records_sorted
        ),
        "all_subclass_boundaries_installed": all(
            record["subclass_boundary_installed"] for record in rank_records_sorted
        ),
        "all_local_grads_finite": all(
            record["local_grad_is_finite"] for record in rank_records_sorted
        ),
        "all_exchanged_grads_finite": all(
            record["exchanged_grad_is_finite"] for record in rank_records_sorted
        ),
        "any_halo_gradient_active": any(
            record["halo_gradient_active"] for record in rank_records_sorted
        ),
        "all_internal_exchange_used_ipc_when_halo": all(
            record["feature_exchange_used_ipc"] for record in rank_records_sorted
        )
        if has_halo
        else True,
        "all_layer_row_counts_match_expected_total": all(
            all(
                (not layer_record.get("executed", False))
                or layer_record.get("row_count_matches_expected_total", False)
                for layer_record in record["layer_records"]
            )
            for record in rank_records_sorted
        ),
        "all_owned_destination_edge_sets_match_monolithic": all(
            item["owned_destination_edge_sets_exact_match"]
            for item in graph_parity_records
        )
        if monolithic_reference is not None
        else False,
        "exchange_schedule_names": sorted(
            {record["exchange_schedule_name"] for record in rank_records_sorted}
        ),
        "exchange_layers_before_interaction": sorted(
            {
                tuple(record["exchange_layers_before_interaction"])
                for record in rank_records_sorted
            }
        ),
        "exchange_layers_after_product": sorted(
            {
                tuple(record["exchange_layers_after_product"])
                for record in rank_records_sorted
            }
        ),
        "summed_owned_local_energy": float(
            sum(record["owned_local_energy"] for record in rank_records_sorted)
        ),
        "summed_expanded_rank_graph_energy": float(
            sum(record["expanded_rank_graph_energy"] for record in rank_records_sorted)
        ),
        "summed_no_exchange_owned_local_energy": float(
            sum(record["comparison_owned_local_energy"] for record in rank_records_sorted)
        ),
        "exchange_vs_no_exchange_energy_delta": float(
            sum(record["owned_local_energy"] for record in rank_records_sorted)
            - sum(record["comparison_owned_local_energy"] for record in rank_records_sorted)
        ),
        "exchange_vs_no_exchange_force_l2_delta": float(exchange_force_delta.norm().item()),
        "num_interactions": sorted(
            {record["num_interactions"] for record in rank_records_sorted}
        ),
        "dtype_policies": sorted(
            {
                (
                    record["model_parameter_dtype"],
                    record["batch_positions_dtype"],
                    record["node_embedding_dtype"],
                    record["edge_attr_dtype"],
                    record["edge_feat_dtype"],
                    record["node_energy_dtype"],
                    record["total_energy_dtype"],
                    record["owned_local_energy_dtype"],
                )
                for record in rank_records_sorted
            }
        ),
        "case_type": "halo" if has_halo else "no_halo",
    }
    return {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": num_atoms,
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "adapter_checks": checks,
        "monolithic_parity": monolithic_parity,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("Full-Model Subclass DistMACE Adapter")
    print("============================================================")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"exchange_layers: {args.exchange_layers}")
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
    print("Adapter Checks")
    print("------------------------------------------------------------")
    checks = summary["adapter_checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(
        "all_recovered_force_shapes_match_owned_atoms: "
        f"{checks['all_recovered_force_shapes_match_owned_atoms']}"
    )
    print(
        "all_subclass_boundaries_installed: "
        f"{checks['all_subclass_boundaries_installed']}"
    )
    print(f"all_local_grads_finite: {checks['all_local_grads_finite']}")
    print(f"all_exchanged_grads_finite: {checks['all_exchanged_grads_finite']}")
    print(f"any_halo_gradient_active: {checks['any_halo_gradient_active']}")
    print(
        "all_internal_exchange_used_ipc_when_halo: "
        f"{checks['all_internal_exchange_used_ipc_when_halo']}"
    )
    print(
        "all_layer_row_counts_match_expected_total: "
        f"{checks['all_layer_row_counts_match_expected_total']}"
    )
    print(
        "all_owned_destination_edge_sets_match_monolithic: "
        f"{checks['all_owned_destination_edge_sets_match_monolithic']}"
    )
    print(f"exchange_schedule_names: {checks['exchange_schedule_names']}")
    print(
        "exchange_layers_before_interaction: "
        f"{checks['exchange_layers_before_interaction']}"
    )
    print(f"exchange_layers_after_product: {checks['exchange_layers_after_product']}")
    print(f"num_interactions: {checks['num_interactions']}")
    print(f"summed_owned_local_energy: {checks['summed_owned_local_energy']}")
    print(
        "summed_expanded_rank_graph_energy: "
        f"{checks['summed_expanded_rank_graph_energy']}"
    )
    print(
        "summed_no_exchange_owned_local_energy: "
        f"{checks['summed_no_exchange_owned_local_energy']}"
    )
    print(
        "exchange_vs_no_exchange_energy_delta: "
        f"{checks['exchange_vs_no_exchange_energy_delta']}"
    )
    print(
        "exchange_vs_no_exchange_force_l2_delta: "
        f"{checks['exchange_vs_no_exchange_force_l2_delta']}"
    )
    print(f"dtype_policies: {checks['dtype_policies']}")
    print(f"case_type: {checks['case_type']}")
    print()
    if summary["monolithic_parity"] is not None:
        parity = summary["monolithic_parity"]
        print("------------------------------------------------------------")
        print("Monolithic Parity")
        print("------------------------------------------------------------")
        print(f"reference_model: {parity['reference_model']}")
        print(f"reference_num_interactions: {parity['reference_num_interactions']}")
        print(f"monolithic_total_energy: {parity['monolithic_total_energy']}")
        print(f"distributed_owned_energy_sum: {parity['distributed_owned_energy_sum']}")
        print(f"absolute_energy_error: {parity['absolute_energy_error']}")
        print(f"distributed_force_shape: {parity['distributed_force_shape']}")
        print(f"monolithic_force_shape: {parity['monolithic_force_shape']}")
        print(f"max_abs_force_error: {parity['max_abs_force_error']}")
        print(f"l2_force_error: {parity['l2_force_error']}")
        print(f"max_abs_node_energy_error: {parity['max_abs_node_energy_error']}")
        print(f"l2_node_energy_error: {parity['l2_node_energy_error']}")
        print(
            "owned_node_energy_exact_match_atol_1e-10: "
            f"{parity['owned_node_energy_exact_match_atol_1e-10']}"
        )
        print(f"global_graph_parity_exact: {parity['global_graph_parity_exact']}")
        print(
            "max_owned_neighbor_count_mismatch: "
            f"{parity['max_owned_neighbor_count_mismatch']}"
        )
        print(
            "monolithic_model_parameter_dtype: "
            f"{parity['monolithic_model_parameter_dtype']}"
        )
        print(
            "monolithic_batch_positions_dtype: "
            f"{parity['monolithic_batch_positions_dtype']}"
        )
        print(
            "monolithic_node_energy_dtype: "
            f"{parity['monolithic_node_energy_dtype']}"
        )
        print(
            "monolithic_total_energy_dtype: "
            f"{parity['monolithic_total_energy_dtype']}"
        )
        print()
        print("------------------------------------------------------------")
        print("Graph Parity By Rank")
        print("------------------------------------------------------------")
        for graph_record in parity["graph_parity_records"]:
            print(f"rank: {graph_record['rank']}")
            print(f"  owned_indices: {graph_record['owned_indices']}")
            print(
                "  distributed_owned_destination_edge_count: "
                f"{graph_record['distributed_owned_destination_edge_count']}"
            )
            print(
                "  monolithic_owned_destination_edge_count: "
                f"{graph_record['monolithic_owned_destination_edge_count']}"
            )
            print(
                "  owned_destination_edge_sets_exact_match: "
                f"{graph_record['owned_destination_edge_sets_exact_match']}"
            )
            print(
                "  per_owned_atom_neighbor_delta: "
                f"{graph_record['per_owned_atom_neighbor_delta']}"
            )
            print()
    print("------------------------------------------------------------")
    print("Per-Rank Observation Points")
    print("------------------------------------------------------------")
    for record in summary["rank_records"]:
        print(f"rank: {record['rank']}")
        print(f"  n_local: {record['n_local']}")
        print(f"  n_total_after_exchange: {record['n_total_after_exchange']}")
        print(f"  n_halo: {record['n_halo']}")
        print(f"  expected_total_recv: {record['expected_total_recv']}")
        print(f"  exchange_mode: {record['exchange_mode']}")
        print(f"  exchange_reason: {record['exchange_reason']}")
        print(f"  total_send: {record['total_send']}")
        print(f"  total_recv: {record['total_recv']}")
        print(f"  feature_exchange_used_ipc: {record['feature_exchange_used_ipc']}")
        print(
            "  executed_exchange_boundaries: "
            f"{record['executed_exchange_boundaries']}"
        )
        print(f"  executed_exchange_layers: {record['executed_exchange_layers']}")
        print(f"  num_interactions: {record['num_interactions']}")
        print(
            "  exchange_layers_before_interaction: "
            f"{record['exchange_layers_before_interaction']}"
        )
        print(f"  exchange_layers_after_product: {record['exchange_layers_after_product']}")
        print(f"  ownership_boundary: {record['ownership_boundary']}")
        print(f"  ownership_rule: {record['ownership_rule']}")
        print(f"  force_rule: {record['force_rule']}")
        print(f"  global_owned_indices: {record['global_owned_indices']}")
        print(f"  halo_global_indices: {record['halo_global_indices']}")
        print(f"  expanded_rank_graph_energy: {record['expanded_rank_graph_energy']}")
        print(f"  owned_local_energy: {record['owned_local_energy']}")
        print(f"  expanded_minus_owned_energy: {record['expanded_minus_owned_energy']}")
        print(f"  comparison_owned_local_energy: {record['comparison_owned_local_energy']}")
        print(
            "  comparison_energy_delta_vs_exchange_on: "
            f"{record['owned_local_energy'] - record['comparison_owned_local_energy']}"
        )
        print(f"  owned_destination_edge_count: {record['owned_destination_edge_count']}")
        print(f"  owned_node_energy_values: {record['owned_node_energy_values']}")
        print(f"  recovered_force_shape: {record['recovered_force_shape']}")
        print(f"  halo_gradient_active: {record['halo_gradient_active']}")
        print(f"  model_parameter_dtype: {record['model_parameter_dtype']}")
        print(f"  batch_positions_dtype: {record['batch_positions_dtype']}")
        print(f"  graph_vectors_dtype: {record['graph_vectors_dtype']}")
        print(f"  graph_lengths_dtype: {record['graph_lengths_dtype']}")
        print(f"  node_embedding_dtype: {record['node_embedding_dtype']}")
        print(f"  edge_attr_dtype: {record['edge_attr_dtype']}")
        print(f"  edge_feat_dtype: {record['edge_feat_dtype']}")
        print(f"  node_energy_dtype: {record['node_energy_dtype']}")
        print(f"  total_energy_dtype: {record['total_energy_dtype']}")
        print(f"  owned_local_energy_dtype: {record['owned_local_energy_dtype']}")
        print(f"  layer_records: {record['layer_records']}")
        print()


def main() -> None:
    try:
        args = build_parser().parse_args()
        schedule = parse_exchange_schedule(args.exchange_layers)
        raw_structure = parse_raw_structure(args)
        structure_diagnostics = analyze_structure(raw_structure)

        torch_dist_env = TorchDistEnv(is_distributed=True)
        device = (
            torch_dist_env.device if args.device == "cuda" else torch.device(args.device)
        )

        if not dist.is_initialized():
            raise RuntimeError(
                "DistMACEFullSubclassAdapter requires a distributed launch. "
                "Run it under torchrun or an equivalent SLURM multi-rank launch."
            )

        raw_model = load_raw_mace_model(
            model_family=args.model_family,
            model=args.model,
            device=str(device),
        )
        raw_model.to(device)
        raw_model.eval()

        global_info = build_global_molecular_info(raw_structure, device)
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
            global_molecular_info=global_info,
            partitioner=partitioner,
            sim_config=sim_config,
            dist_env=torch_dist_env.dist_env,
            device=device,
        )

        adapter = DistMACEFullSubclassAdapter(raw_model, schedule=schedule)
        adapter.initialize_buffers(
            distributed_state.local_partition_info,
            device=device,
            world_size=int(distributed_state.dist_env.world_size),
        )
        energy, forces = adapter(distributed_state)

        comparison_model = load_raw_mace_model(
            model_family=args.model_family,
            model=resolve_comparison_model(args),
            device=str(device),
        )
        comparison_model.to(device)
        comparison_model.eval()
        comparison_adapter = DistMACEFullSubclassAdapter(
            comparison_model,
            schedule=FeatureExchangeSchedule(
                name="no_internal_exchange",
                exchange_after_product_layers=(),
            ),
        )
        comparison_adapter.initialize_buffers(
            distributed_state.local_partition_info,
            device=device,
            world_size=int(distributed_state.dist_env.world_size),
        )
        comparison_energy, comparison_forces = comparison_adapter(distributed_state)

        record = dict(adapter.last_observation)
        record["returned_energy"] = float(energy.detach().cpu().item())
        record["returned_force_shape"] = list(forces.shape)
        record["returned_forces"] = forces.detach().cpu().tolist()
        record["global_owned_indices"] = (
            distributed_state.local_partition_info.partition_indices.detach().cpu().tolist()
        )
        record["comparison_owned_local_energy"] = float(
            comparison_energy.detach().cpu().item()
        )
        record["comparison_returned_forces"] = comparison_forces.detach().cpu().tolist()
        gathered = maybe_gather_records(record)

        if dist.get_rank() == 0:
            monolithic_reference = build_monolithic_reference(
                raw_structure=raw_structure,
                sim_config=sim_config,
                args=args,
                device=device,
            )
            summary = summarize_rank_results(
                raw_structure,
                structure_diagnostics,
                gathered,
                monolithic_reference=monolithic_reference,
            )
            print_summary(summary, args)
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print()
                print(f"Wrote JSON summary to {output_path}")
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
