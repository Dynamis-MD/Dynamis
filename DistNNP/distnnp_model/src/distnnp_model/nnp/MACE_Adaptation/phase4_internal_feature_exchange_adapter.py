"""Phase 4 internal learned-feature exchange prototype for distributed MACE.

This prototype extends the owned-energy / owned-force path by inserting one
learned-feature exchange point inside the MACE interaction stack:

local DistNNP partition
-> exchange raw positions / atomic numbers
-> local + halo molecular view
-> standard MACE graph build
-> internal learned-feature exchange after `linear_up(...)`
   and before `conv_tp(...)` on one chosen interaction layer
-> owned local energy from outputs["node_energy"][:num_local].sum()
-> owned local forces from -d(owned_local_energy)/d(local_positions)

The insertion point follows the Step 6 analysis: exchange the flattened
per-node tensor consumed by the message-construction path, while preserving the
existing local-then-ghost ordering already used by the structural halo path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_state import SFCCurveType, SFCPartitioner, TorchDistEnv

from distnnp_model.nnp.MACE_Adaptation.phase1_raw_exchange_adapter import (
    analyze_structure,
    build_global_molecular_info,
    build_parser as build_phase1_parser,
    build_phase1_distnnp_state,
    build_sim_config,
    finalize_distributed,
    load_raw_mace_model,
    maybe_gather_records,
    parse_raw_structure,
)
from distnnp_model.nnp.MACE_Adaptation.phase3_owned_force_adapter import (
    MACEPotentialModelWithOwnedEnergyForces,
)


def build_parser():
    parser = build_phase1_parser()
    parser.description = (
        "Phase 4 internal learned-feature exchange prototype: insert one "
        "feature exchange point after linear_up(...) and before conv_tp(...) "
        "on a chosen interaction layer, then recover owned local forces from "
        "the owned scalar energy."
    )
    parser.add_argument(
        "--feature-exchange-layer",
        type=int,
        default=1,
        help=(
            "Interaction layer index at which to insert the learned-feature "
            "exchange. Defaults to 1, the first non-initial interaction layer."
        ),
    )
    return parser


class MACEPotentialModelWithInternalFeatureExchange(
    MACEPotentialModelWithOwnedEnergyForces
):
    """Prototype wrapper with one internal learned-feature exchange boundary."""

    internal_exchange_boundary = "post_linear_up_pre_conv_tp"

    def __init__(self, model, feature_exchange_layer: int = 1) -> None:
        super().__init__(model)
        self.feature_exchange_layer = int(feature_exchange_layer)
        self.learned_feature_exchanger = GPUDirectExchanger()
        self._current_distributed_state = None
        self._feature_exchange_observation: Dict[str, Any] = {}
        self._install_internal_feature_exchange_hook()

    def _install_internal_feature_exchange_hook(self) -> None:
        interactions = getattr(self.model, "interactions", None)
        if interactions is None:
            raise AttributeError(
                "Loaded MACE model does not expose an 'interactions' stack; "
                "cannot install the internal feature exchange hook."
            )
        if self.feature_exchange_layer < 0 or self.feature_exchange_layer >= len(interactions):
            raise ValueError(
                f"feature_exchange_layer={self.feature_exchange_layer} is out of range for "
                f"{len(interactions)} interaction blocks."
            )

        target_block = interactions[self.feature_exchange_layer]
        original_handle_lammps = target_block.handle_lammps
        wrapper = self
        layer_index = self.feature_exchange_layer

        def _distnnp_handle_lammps(
            block_self,
            node_feats: torch.Tensor,
            lammps_class,
            lammps_natoms,
            first_layer: bool,
        ) -> torch.Tensor:
            # Preserve MACE's own LAMMPS path and only intervene at the chosen
            # non-initial DistNNP prototype layer.
            if lammps_class is not None:
                return original_handle_lammps(
                    node_feats,
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                    first_layer=first_layer,
                )
            if first_layer:
                return node_feats
            if wrapper.exchange_mode != "ipc":
                return node_feats
            if wrapper._current_distributed_state is None:
                raise RuntimeError(
                    "Internal feature exchange hook was invoked without an active DistNNPState."
                )

            distributed_state = wrapper._current_distributed_state
            n_local = int(
                distributed_state.local_partition_info.partition_indices.shape[0]
            )
            learned_local_node_feats = node_feats[:n_local].contiguous()
            input_dtype = node_feats.dtype
            exchanged_node_feats = FeaturePushToExchange.apply(
                learned_local_node_feats,
                distributed_state,
                wrapper.learned_feature_exchanger,
            )
            if exchanged_node_feats.dtype != input_dtype:
                exchanged_node_feats = exchanged_node_feats.to(dtype=input_dtype)
            wrapper._feature_exchange_observation = {
                "feature_exchange_layer": layer_index,
                "feature_exchange_boundary": wrapper.internal_exchange_boundary,
                "input_feature_shape": list(node_feats.shape),
                "owned_local_feature_shape": list(learned_local_node_feats.shape),
                "exchanged_feature_shape": list(exchanged_node_feats.shape),
                "input_feature_dtype": str(input_dtype),
                "exchanged_feature_dtype": str(exchanged_node_feats.dtype),
                "feature_exchange_used_ipc": True,
            }
            return exchanged_node_feats

        target_block.handle_lammps = MethodType(_distnnp_handle_lammps, target_block)

    def __call__(self, distributed_state, *args, **kwargs):
        self._current_distributed_state = distributed_state
        self._feature_exchange_observation = {
            "feature_exchange_layer": self.feature_exchange_layer,
            "feature_exchange_boundary": self.internal_exchange_boundary,
            "feature_exchange_used_ipc": False,
        }
        try:
            owned_local_energy, owned_local_forces = super().__call__(
                distributed_state,
                *args,
                **kwargs,
            )
        finally:
            self._current_distributed_state = None

        self.last_observation.update(self._feature_exchange_observation)
        return owned_local_energy, owned_local_forces


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    exchanged_shapes = [
        tuple(record.get("exchanged_feature_shape", []))
        for record in rank_records_sorted
        if record.get("exchanged_feature_shape")
    ]
    shape_consistent = (
        len(exchanged_shapes) > 0
        and all(shape == exchanged_shapes[0] for shape in exchanged_shapes)
    )
    feature_exchange_checks = {
        "all_expected_halo_counts_match": all(
            record["n_halo"] == record["expected_total_recv"]
            for record in rank_records_sorted
        ),
        "all_owned_energies_finite": all(
            np.isfinite(record["owned_local_energy"]) for record in rank_records_sorted
        ),
        "all_raw_energies_finite": all(
            np.isfinite(record["raw_total_energy"]) for record in rank_records_sorted
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
        "all_internal_exchange_shapes_consistent": shape_consistent,
        "all_internal_exchange_used_ipc": all(
            record.get("feature_exchange_used_ipc", False) for record in rank_records_sorted
        ),
        "any_halo_gradient_active": any(
            record["halo_gradient_active"] for record in rank_records_sorted
        ),
        "feature_exchange_boundary": sorted(
            {record["feature_exchange_boundary"] for record in rank_records_sorted}
        ),
        "feature_exchange_layers": sorted(
            {record["feature_exchange_layer"] for record in rank_records_sorted}
        ),
        "summed_owned_local_energy": float(
            sum(record["owned_local_energy"] for record in rank_records_sorted)
        ),
        "summed_raw_total_energy": float(
            sum(record["raw_total_energy"] for record in rank_records_sorted)
        ),
    }
    return {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "feature_exchange_checks": feature_exchange_checks,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args) -> None:
    print("============================================================")
    print("Phase 4 Internal Feature-Exchange MACE Prototype")
    print("============================================================")
    print(f"model_family: {args.model_family}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"feature_exchange_layer: {args.feature_exchange_layer}")
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
    print("Feature Exchange Checks")
    print("------------------------------------------------------------")
    checks = summary["feature_exchange_checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(f"all_owned_energies_finite: {checks['all_owned_energies_finite']}")
    print(f"all_raw_energies_finite: {checks['all_raw_energies_finite']}")
    print(
        "all_recovered_force_shapes_match_owned_atoms: "
        f"{checks['all_recovered_force_shapes_match_owned_atoms']}"
    )
    print(f"all_local_grads_finite: {checks['all_local_grads_finite']}")
    print(f"all_exchanged_grads_finite: {checks['all_exchanged_grads_finite']}")
    print(
        "all_internal_exchange_shapes_consistent: "
        f"{checks['all_internal_exchange_shapes_consistent']}"
    )
    print(f"all_internal_exchange_used_ipc: {checks['all_internal_exchange_used_ipc']}")
    print(f"any_halo_gradient_active: {checks['any_halo_gradient_active']}")
    print(f"feature_exchange_boundary: {checks['feature_exchange_boundary']}")
    print(f"feature_exchange_layers: {checks['feature_exchange_layers']}")
    print(f"summed_owned_local_energy: {checks['summed_owned_local_energy']}")
    print(f"summed_raw_total_energy: {checks['summed_raw_total_energy']}")
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
        print(f"  send_counts_by_peer: {record['send_counts_by_peer']}")
        print(f"  recv_counts_by_peer: {record['recv_counts_by_peer']}")
        print(f"  edge_count_after_atomic_data: {record['edge_count_after_atomic_data']}")
        print(f"  local_atomic_numbers: {record['local_atomic_numbers']}")
        print(f"  halo_atomic_numbers: {record['halo_atomic_numbers']}")
        print(f"  feature_exchange_boundary: {record['feature_exchange_boundary']}")
        print(f"  feature_exchange_layer: {record['feature_exchange_layer']}")
        print(f"  feature_exchange_used_ipc: {record['feature_exchange_used_ipc']}")
        print(f"  input_feature_shape: {record.get('input_feature_shape')}")
        print(f"  owned_local_feature_shape: {record.get('owned_local_feature_shape')}")
        print(f"  exchanged_feature_shape: {record.get('exchanged_feature_shape')}")
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
                "Phase 4 internal feature-exchange adapter requires a distributed launch. "
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

        adapter = MACEPotentialModelWithInternalFeatureExchange(
            raw_model,
            feature_exchange_layer=args.feature_exchange_layer,
        )
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
