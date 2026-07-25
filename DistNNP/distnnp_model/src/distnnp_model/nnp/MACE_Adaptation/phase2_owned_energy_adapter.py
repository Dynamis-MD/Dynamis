"""Phase 2 ownership-aware local-energy prototype for distributed MACE.

This prototype builds directly on the stabilized Phase 1 raw-exchange wrapper:

local DistNNP partition
-> exchange raw positions / atomic numbers
-> local + halo molecular view
-> standard MACE forward
-> owned local energy from outputs["node_energy"][:num_local].sum()

It intentionally keeps the Phase 1 force path for now. The goal of this file is
to make the ownership rule explicit at the model boundary without yet solving
the autograd-correct local-force story from the owned scalar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from distnnp_state import SFCCurveType, SFCPartitioner, TorchDistEnv

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


def build_parser():
    parser = build_phase1_parser()
    parser.description = (
        "Phase 2 ownership-aware local-energy prototype: build the Phase 1 "
        "local+halo MACE view, then return a local owned scalar energy from "
        "the public per-atom node_energy tensor."
    )
    return parser


class MACEPotentialModelWithOwnedEnergy(MACEPotentialModelWithRawExchange):
    """Phase 2 wrapper that imposes ownership on public per-atom node energies."""

    ownership_boundary = "public_node_energy"
    ownership_rule = "owned_local_energy = outputs['node_energy'][:num_local].sum()"

    def __call__(self, distributed_state, *args, **kwargs):
        validation = validate_phase1_state(distributed_state)
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

        exchanged_atomic_numbers = self._map_atomic_numbers(exchanged_atom_types)
        n_local = int(local_positions.shape[0])
        n_total = int(exchanged_positions.shape[0])
        n_halo = int(n_total - n_local)

        config = self._build_configuration(
            exchanged_atomic_numbers,
            exchanged_positions,
            distributed_state,
        )
        atomic_data = self.AtomicData.from_config(  # type: ignore[attr-defined]
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.heads,
        )
        edge_count = int(atomic_data.edge_index.shape[1])

        batch = self.Batch.from_data_list([atomic_data]).to(local_positions.device)  # type: ignore[attr-defined]
        batch_dict = align_batch_dtype(batch.to_dict(), self.model)
        outputs = self.model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_stress=False,
            compute_hessian=False,
        )

        if "node_energy" not in outputs:
            raise KeyError(
                "MACE output dict does not expose 'node_energy'; this prototype "
                "cannot impose ownership at the public per-atom boundary."
            )

        node_energy = outputs["node_energy"]
        if not torch.is_tensor(node_energy):
            raise TypeError(
                f"Expected outputs['node_energy'] to be a tensor, got {type(node_energy)!r}."
            )

        node_energy_flat = node_energy.reshape(-1)
        if int(node_energy_flat.numel()) != n_total:
            raise ValueError(
                "node_energy length does not match the expanded local+halo atom count: "
                f"{int(node_energy_flat.numel())} vs {n_total}."
            )

        owned_node_energy = node_energy_flat[:n_local]
        owned_local_energy = owned_node_energy.sum()

        raw_total_energy = outputs["energy"]
        if torch.is_tensor(raw_total_energy) and raw_total_energy.numel() == 1:
            raw_total_energy = raw_total_energy.reshape(())

        forces = outputs["forces"]

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
            "ownership_boundary": self.ownership_boundary,
            "ownership_rule": self.ownership_rule,
            "node_energy_shape": list(node_energy.shape),
            "owned_node_energy_shape": list(owned_node_energy.shape),
            "raw_total_energy": float(raw_total_energy.detach().cpu().item()),
            "owned_local_energy": float(owned_local_energy.detach().cpu().item()),
            "raw_minus_owned_energy": float(
                (raw_total_energy - owned_local_energy).detach().cpu().item()
            ),
        }

        return owned_local_energy.detach(), forces[:n_local].detach()


# Phase 1 imports these through optional runtime MACE imports.
# Rebind them on the subclass so the type checker and runtime both stay happy.
from distnnp_model.nnp.MACE_Adaptation import phase1_raw_exchange_adapter as _phase1

MACEPotentialModelWithOwnedEnergy.AtomicData = _phase1.AtomicData
MACEPotentialModelWithOwnedEnergy.Batch = _phase1.Batch


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    ownership_checks = {
        "all_expected_halo_counts_match": all(
            record["n_halo"] == record["expected_total_recv"]
            for record in rank_records_sorted
        ),
        "all_force_shapes_match_owned_atoms": all(
            record["returned_force_shape"] == [record["n_local"], 3]
            for record in rank_records_sorted
        ),
        "all_owned_energies_finite": all(
            np.isfinite(record["owned_local_energy"]) for record in rank_records_sorted
        ),
        "all_raw_energies_finite": all(
            np.isfinite(record["raw_total_energy"]) for record in rank_records_sorted
        ),
        "all_node_energy_shapes_cover_expanded_atoms": all(
            record["owned_node_energy_shape"] == [record["n_local"]]
            for record in rank_records_sorted
        ),
        "ownership_boundary": sorted(
            {record["ownership_boundary"] for record in rank_records_sorted}
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
        "ownership_checks": ownership_checks,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args) -> None:
    print("============================================================")
    print("Phase 2 Owned-Energy MACE Prototype")
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
    print("Ownership Checks")
    print("------------------------------------------------------------")
    checks = summary["ownership_checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(
        f"all_force_shapes_match_owned_atoms: {checks['all_force_shapes_match_owned_atoms']}"
    )
    print(f"all_owned_energies_finite: {checks['all_owned_energies_finite']}")
    print(f"all_raw_energies_finite: {checks['all_raw_energies_finite']}")
    print(
        "all_node_energy_shapes_cover_expanded_atoms: "
        f"{checks['all_node_energy_shapes_cover_expanded_atoms']}"
    )
    print(f"ownership_boundary: {checks['ownership_boundary']}")
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
        print(f"  ownership_boundary: {record['ownership_boundary']}")
        print(f"  ownership_rule: {record['ownership_rule']}")
        print(f"  node_energy_shape: {record['node_energy_shape']}")
        print(f"  owned_node_energy_shape: {record['owned_node_energy_shape']}")
        print(f"  raw_total_energy: {record['raw_total_energy']}")
        print(f"  owned_local_energy: {record['owned_local_energy']}")
        print(f"  raw_minus_owned_energy: {record['raw_minus_owned_energy']}")
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
                "Phase 2 owned-energy adapter requires a distributed launch. "
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

        adapter = MACEPotentialModelWithOwnedEnergy(raw_model)
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
