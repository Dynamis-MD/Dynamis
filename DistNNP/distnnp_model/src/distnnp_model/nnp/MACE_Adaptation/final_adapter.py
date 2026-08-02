"""Single-layer distributed MACE milestone adapter.

This module now represents the merged milestone path: structural halo support,
one subclass-owned learned-feature exchange layer, ownership-aware local
energy, autograd-recovered owned local forces, and validation for both halo
and no-halo runs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.nnp.MACE_Adaptation.mace_subclass_boundary import (
    DistMACEModelSubclassBoundary,
)
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


@dataclass(frozen=True)
class FeatureExchangePolicy:
    """Frozen single-layer policy surface for subclass-owned feature exchange."""

    name: str = "layer_1_only"
    interaction_layer: int = 1
    boundary: str = "post_linear_up_pre_conv_tp"

    def validate(self) -> None:
        if self.name != "layer_1_only":
            raise ValueError(
                "The single-layer milestone freezes a single policy: 'layer_1_only'."
            )
        if self.interaction_layer != 1:
            raise ValueError(
                "The single-layer milestone freezes the adapter to interaction_layer=1."
            )


@dataclass
class StructuralHaloView:
    local_positions: torch.Tensor
    local_atom_types: torch.Tensor
    exchanged_positions: torch.Tensor
    exchanged_atom_types: torch.Tensor
    exchanged_atomic_numbers: torch.Tensor
    num_local: int
    num_total: int
    num_halo: int


class StructuralHaloBuilder:
    """Build the owned-local plus halo structural view used by the adapter."""

    def build(
        self,
        adapter: "DistMACEAdapter",
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

        num_local = int(local_positions.shape[0])
        num_total = int(exchanged_positions.shape[0])
        return StructuralHaloView(
            local_positions=local_positions,
            local_atom_types=local_atom_types,
            exchanged_positions=exchanged_positions,
            exchanged_atom_types=exchanged_atom_types,
            exchanged_atomic_numbers=exchanged_atomic_numbers,
            num_local=num_local,
            num_total=num_total,
            num_halo=int(num_total - num_local),
        )


class MACEGraphBuilder:
    """Build a differentiable public-MACE batch dictionary."""

    def build(
        self,
        adapter: "DistMACEAdapter",
        structural_view: StructuralHaloView,
        distributed_state: DistNNPState,
    ) -> tuple[Dict[str, torch.Tensor], int]:
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
        return batch_dict, edge_count


class OwnedEnergyHead:
    """Freeze the milestone ownership rule at the public node-energy boundary."""

    boundary = "public_node_energy"
    rule = "owned_local_energy = outputs['node_energy'][:num_local].sum()"

    def build(
        self,
        outputs: Dict[str, torch.Tensor],
        num_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "node_energy" not in outputs:
            raise KeyError(
                "MACE output dict does not expose 'node_energy'; the single-layer "
                "milestone cannot "
                "freeze the ownership rule without that tensor."
            )
        node_energy = outputs["node_energy"]
        if not torch.is_tensor(node_energy):
            raise TypeError(
                f"Expected outputs['node_energy'] to be a tensor, got {type(node_energy)!r}."
            )

        node_energy_flat = node_energy.reshape(-1)
        owned_node_energy = node_energy_flat[:num_local]
        owned_local_energy = owned_node_energy.sum()
        return node_energy, owned_node_energy, owned_local_energy


class OwnedForceRecoverer:
    """Recover owned local forces by differentiating owned local energy."""

    rule = "owned_local_forces = -grad(owned_local_energy, local_positions)"

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

        diagnostics = {
            "recovered_force_shape": list(owned_local_forces.shape),
            "local_grad_is_finite": local_grad_is_finite,
            "exchanged_grad_is_finite": exchanged_grad_is_finite,
            "local_grad_norm": local_grad_norm,
            "halo_grad_norm": halo_grad_norm,
            "local_force_norm": local_force_norm,
            "halo_gradient_active": halo_gradient_active,
            "backward_depends_on_halo": halo_gradient_active,
        }
        return owned_local_forces, diagnostics


class ValidationDiagnostics:
    """Centralize the milestone reporting contract for the new adapter."""

    adapter_contract_name = "dist_mace_single_layer_milestone"

    def build_record(
        self,
        adapter: "DistMACEAdapter",
        distributed_state: DistNNPState,
        structural_view: StructuralHaloView,
        edge_count: int,
        node_energy: torch.Tensor,
        owned_node_energy: torch.Tensor,
        owned_local_energy: torch.Tensor,
        raw_total_energy: torch.Tensor,
        force_diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        validation = adapter.partition_validation or {}
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
            "feature_exchange_policy": adapter.feature_exchange_policy.name,
            "feature_exchange_layer": adapter.feature_exchange_policy.interaction_layer,
            "feature_exchange_boundary": adapter.feature_exchange_policy.boundary,
            "feature_exchange_used_ipc": adapter.last_observation.get(
                "feature_exchange_used_ipc", False
            ),
            "subclass_boundary_executed": adapter.last_observation.get(
                "subclass_boundary_executed", False
            ),
            "input_feature_shape": adapter.last_observation.get("input_feature_shape"),
            "owned_local_feature_shape": adapter.last_observation.get(
                "owned_local_feature_shape"
            ),
            "exchanged_feature_shape": adapter.last_observation.get(
                "exchanged_feature_shape"
            ),
            **adapter.model_subclass_boundary.describe(),
            "ownership_boundary": adapter.owned_energy_head.boundary,
            "ownership_rule": adapter.owned_energy_head.rule,
            "force_rule": adapter.owned_force_recoverer.rule,
            "node_energy_shape": list(node_energy.shape),
            "owned_node_energy_shape": list(owned_node_energy.shape),
            "raw_total_energy": float(raw_total_energy.detach().cpu().item()),
            "owned_local_energy": float(owned_local_energy.detach().cpu().item()),
            "raw_minus_owned_energy": float(
                (raw_total_energy - owned_local_energy).detach().cpu().item()
            ),
            **force_diagnostics,
        }


class DistMACEAdapter(MACEPotentialModelWithRawExchange):
    """Subclass-boundary distributed MACE adapter with explicit model seam."""

    adapter_target = (
        "single-layer structural_halo + layer_1_only feature exchange + "
        "owned node_energy + autograd-owned local forces"
    )
    adapter_status = "single_layer_milestone_implemented"

    def __init__(
        self,
        model,
        feature_exchange_policy: FeatureExchangePolicy | None = None,
    ) -> None:
        policy = feature_exchange_policy or FeatureExchangePolicy()
        policy.validate()

        super().__init__(model)
        self.feature_exchange_policy = policy
        self.learned_feature_exchanger = GPUDirectExchanger()
        self._feature_exchange_observation: Dict[str, Any] = {}
        self.model_subclass_boundary = DistMACEModelSubclassBoundary(policy)
        self.model_subclass_boundary.install(self.model)
        self.structural_halo_builder = StructuralHaloBuilder()
        self.mace_graph_builder = MACEGraphBuilder()
        self.owned_energy_head = OwnedEnergyHead()
        self.owned_force_recoverer = OwnedForceRecoverer()
        self.validation_diagnostics = ValidationDiagnostics()

    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        self._feature_exchange_observation = {
            "feature_exchange_layer": self.feature_exchange_policy.interaction_layer,
            "feature_exchange_boundary": self.feature_exchange_policy.boundary,
            "feature_exchange_used_ipc": False,
            "subclass_boundary_executed": False,
        }
        self.model_subclass_boundary.begin_forward(
            distributed_state=distributed_state,
            exchange_mode=self.exchange_mode,
            learned_feature_exchanger=self.learned_feature_exchanger,
            observation=self._feature_exchange_observation,
        )
        try:
            structural_view = self.structural_halo_builder.build(self, distributed_state)
            batch_dict, edge_count = self.mace_graph_builder.build(
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
            node_energy, owned_node_energy, owned_local_energy = self.owned_energy_head.build(
                outputs,
                structural_view.num_local,
            )
            raw_total_energy = outputs["energy"]
            if torch.is_tensor(raw_total_energy) and raw_total_energy.numel() == 1:
                raw_total_energy = raw_total_energy.reshape(())

            owned_local_forces, force_diagnostics = self.owned_force_recoverer.recover(
                owned_local_energy,
                structural_view,
            )
        finally:
            self.model_subclass_boundary.end_forward()

        self.last_observation = self.validation_diagnostics.build_record(
            adapter=self,
            distributed_state=distributed_state,
            structural_view=structural_view,
            edge_count=edge_count,
            node_energy=node_energy,
            owned_node_energy=owned_node_energy,
            owned_local_energy=owned_local_energy,
            raw_total_energy=raw_total_energy,
            force_diagnostics=force_diagnostics,
        )
        self.last_observation.update(self._feature_exchange_observation)
        return owned_local_energy.detach(), owned_local_forces.detach()


from distnnp_model.nnp.MACE_Adaptation import phase1_raw_exchange_adapter as _phase1

DistMACEAdapter.AtomicData = _phase1.AtomicData
DistMACEAdapter.Batch = _phase1.Batch


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase1_parser()
    parser.description = (
        "Single-layer DistMACEAdapter milestone: structural halo support, one "
        "subclass-owned learned-feature exchange layer, ownership-aware local "
        "energy, autograd-owned local forces, and validation for halo and "
        "no-halo distributed runs."
    )
    return parser


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    has_halo = any(record["n_halo"] > 0 for record in rank_records_sorted)
    no_halo = all(record["n_halo"] == 0 for record in rank_records_sorted)
    all_feature_shapes_match_total = all(
        (
            record["exchanged_feature_shape"] is None
            or record["exchanged_feature_shape"][0] == record["n_total_after_exchange"]
        )
        for record in rank_records_sorted
    )
    all_owned_energy_matches_raw_when_no_halo = all(
        abs(record["raw_minus_owned_energy"]) <= 1e-9 for record in rank_records_sorted
    )
    any_owned_energy_excludes_halo_contribution = any(
        abs(record["raw_minus_owned_energy"]) > 1e-9 for record in rank_records_sorted
    )
    checks = {
        "all_expected_halo_counts_match": all(
            record["n_halo"] == record["expected_total_recv"]
            for record in rank_records_sorted
        ),
        "all_owned_energies_finite": all(
            torch.isfinite(torch.tensor(record["owned_local_energy"])).item()
            for record in rank_records_sorted
        ),
        "all_raw_energies_finite": all(
            torch.isfinite(torch.tensor(record["raw_total_energy"])).item()
            for record in rank_records_sorted
        ),
        "all_recovered_force_shapes_match_owned_atoms": all(
            record["recovered_force_shape"] == [record["n_local"], 3]
            for record in rank_records_sorted
        ),
        "all_subclass_boundaries_installed": all(
            record["subclass_boundary_installed"] for record in rank_records_sorted
        ),
        "any_subclass_boundaries_executed": any(
            record["subclass_boundary_executed"] for record in rank_records_sorted
        ),
        "all_internal_exchange_used_ipc": all(
            record["feature_exchange_used_ipc"] for record in rank_records_sorted
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
        "feature_exchange_policy": sorted(
            {record["feature_exchange_policy"] for record in rank_records_sorted}
        ),
        "feature_exchange_layers": sorted(
            {record["feature_exchange_layer"] for record in rank_records_sorted}
        ),
        "model_subclass_contracts": sorted(
            {record["model_subclass_contract"] for record in rank_records_sorted}
        ),
        "adapter_status": sorted({record["adapter_status"] for record in rank_records_sorted}),
        "summed_owned_local_energy": float(
            sum(record["owned_local_energy"] for record in rank_records_sorted)
        ),
        "summed_raw_total_energy": float(
            sum(record["raw_total_energy"] for record in rank_records_sorted)
        ),
        "case_type": "halo" if has_halo else "no_halo",
        "all_feature_shapes_match_total": all_feature_shapes_match_total,
        "all_owned_energy_matches_raw_when_no_halo": all_owned_energy_matches_raw_when_no_halo,
        "any_owned_energy_excludes_halo_contribution": any_owned_energy_excludes_halo_contribution,
        "milestone_objectives_satisfied": (
            all(
                [
                    all(
                        record["recovered_force_shape"] == [record["n_local"], 3]
                        for record in rank_records_sorted
                    ),
                    all(record["subclass_boundary_installed"] for record in rank_records_sorted),
                    all_feature_shapes_match_total,
                    all(record["local_grad_is_finite"] for record in rank_records_sorted),
                    all(record["exchanged_grad_is_finite"] for record in rank_records_sorted),
                ]
            )
            and (
                (
                    has_halo
                    and any(record["subclass_boundary_executed"] for record in rank_records_sorted)
                    and all(record["feature_exchange_used_ipc"] for record in rank_records_sorted)
                    and any(record["halo_gradient_active"] for record in rank_records_sorted)
                    and any_owned_energy_excludes_halo_contribution
                )
                or (
                    no_halo
                    and not any(record["subclass_boundary_executed"] for record in rank_records_sorted)
                    and not any(record["feature_exchange_used_ipc"] for record in rank_records_sorted)
                    and all_owned_energy_matches_raw_when_no_halo
                )
            )
        ),
    }
    return {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "adapter_checks": checks,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("DistMACEAdapter Single-Layer Distributed MACE Milestone")
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
    print("Adapter Checks")
    print("------------------------------------------------------------")
    checks = summary["adapter_checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(f"all_owned_energies_finite: {checks['all_owned_energies_finite']}")
    print(f"all_raw_energies_finite: {checks['all_raw_energies_finite']}")
    print(
        "all_recovered_force_shapes_match_owned_atoms: "
        f"{checks['all_recovered_force_shapes_match_owned_atoms']}"
    )
    print(
        "all_subclass_boundaries_installed: "
        f"{checks['all_subclass_boundaries_installed']}"
    )
    print(
        "any_subclass_boundaries_executed: "
        f"{checks['any_subclass_boundaries_executed']}"
    )
    print(f"all_internal_exchange_used_ipc: {checks['all_internal_exchange_used_ipc']}")
    print(f"all_local_grads_finite: {checks['all_local_grads_finite']}")
    print(f"all_exchanged_grads_finite: {checks['all_exchanged_grads_finite']}")
    print(f"any_halo_gradient_active: {checks['any_halo_gradient_active']}")
    print(f"feature_exchange_policy: {checks['feature_exchange_policy']}")
    print(f"feature_exchange_layers: {checks['feature_exchange_layers']}")
    print(f"model_subclass_contracts: {checks['model_subclass_contracts']}")
    print(f"adapter_status: {checks['adapter_status']}")
    print(f"case_type: {checks['case_type']}")
    print(f"all_feature_shapes_match_total: {checks['all_feature_shapes_match_total']}")
    print(
        "all_owned_energy_matches_raw_when_no_halo: "
        f"{checks['all_owned_energy_matches_raw_when_no_halo']}"
    )
    print(
        "any_owned_energy_excludes_halo_contribution: "
        f"{checks['any_owned_energy_excludes_halo_contribution']}"
    )
    print(f"milestone_objectives_satisfied: {checks['milestone_objectives_satisfied']}")
    print(f"summed_owned_local_energy: {checks['summed_owned_local_energy']}")
    print(f"summed_raw_total_energy: {checks['summed_raw_total_energy']}")
    print()
    print("------------------------------------------------------------")
    print("Per-Rank Observation Points")
    print("------------------------------------------------------------")
    for record in summary["rank_records"]:
        print(f"rank: {record['rank']}")
        print(f"  adapter_contract: {record['adapter_contract']}")
        print(f"  adapter_target: {record['adapter_target']}")
        print(f"  adapter_status: {record['adapter_status']}")
        print(f"  model_subclass_contract: {record['model_subclass_contract']}")
        print(f"  model_subclass_status: {record['model_subclass_status']}")
        print(f"  installed_model_class_name: {record['installed_model_class_name']}")
        print(
            f"  installed_interaction_class_name: "
            f"{record['installed_interaction_class_name']}"
        )
        print(
            f"  installed_interaction_index: {record['installed_interaction_index']}"
        )
        print(f"  n_local: {record['n_local']}")
        print(f"  n_total_after_exchange: {record['n_total_after_exchange']}")
        print(f"  n_halo: {record['n_halo']}")
        print(f"  expected_total_recv: {record['expected_total_recv']}")
        print(f"  exchange_mode: {record['exchange_mode']}")
        print(f"  exchange_reason: {record['exchange_reason']}")
        print(f"  total_send: {record['total_send']}")
        print(f"  total_recv: {record['total_recv']}")
        print(f"  send_counts_by_peer: {record['send_counts_by_peer']}")
        print(f"  recv_counts_by_peer: {record['recv_counts_by_peer']}")
        print(f"  edge_count_after_atomic_data: {record['edge_count_after_atomic_data']}")
        print(f"  local_atomic_numbers: {record['local_atomic_numbers']}")
        print(f"  halo_atomic_numbers: {record['halo_atomic_numbers']}")
        print(f"  feature_exchange_policy: {record['feature_exchange_policy']}")
        print(f"  feature_exchange_layer: {record['feature_exchange_layer']}")
        print(f"  feature_exchange_boundary: {record['feature_exchange_boundary']}")
        print(f"  feature_exchange_used_ipc: {record['feature_exchange_used_ipc']}")
        print(f"  subclass_boundary_installed: {record['subclass_boundary_installed']}")
        print(f"  subclass_boundary_executed: {record['subclass_boundary_executed']}")
        print(f"  input_feature_shape: {record['input_feature_shape']}")
        print(f"  owned_local_feature_shape: {record['owned_local_feature_shape']}")
        print(f"  exchanged_feature_shape: {record['exchanged_feature_shape']}")
        print(f"  ownership_boundary: {record['ownership_boundary']}")
        print(f"  ownership_rule: {record['ownership_rule']}")
        print(f"  force_rule: {record['force_rule']}")
        print(f"  node_energy_shape: {record['node_energy_shape']}")
        print(f"  owned_node_energy_shape: {record['owned_node_energy_shape']}")
        print(f"  recovered_force_shape: {record['recovered_force_shape']}")
        print(f"  raw_total_energy: {record['raw_total_energy']}")
        print(f"  owned_local_energy: {record['owned_local_energy']}")
        print(f"  raw_minus_owned_energy: {record['raw_minus_owned_energy']}")
        print(f"  local_grad_is_finite: {record['local_grad_is_finite']}")
        print(f"  exchanged_grad_is_finite: {record['exchanged_grad_is_finite']}")
        print(f"  local_grad_norm: {record['local_grad_norm']}")
        print(f"  halo_grad_norm: {record['halo_grad_norm']}")
        print(f"  local_force_norm: {record['local_force_norm']}")
        print(f"  halo_gradient_active: {record['halo_gradient_active']}")
        print(f"  backward_depends_on_halo: {record['backward_depends_on_halo']}")
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

        if not dist.is_initialized():
            raise RuntimeError(
                "DistMACEAdapter single-layer milestone requires a distributed launch. "
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

        adapter = DistMACEAdapter(raw_model)
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
