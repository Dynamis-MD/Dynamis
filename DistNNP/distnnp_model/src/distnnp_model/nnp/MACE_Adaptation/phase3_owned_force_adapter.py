"""Phase 3 owned-energy / owned-force prototype for distributed MACE.

This prototype extends the Phase 2 ownership-aware local-energy wrapper by
rebuilding the local force path around the owned scalar energy:

local DistNNP partition
-> exchange raw positions / atomic numbers
-> local + halo molecular view
-> standard MACE forward on a differentiable position tensor
-> owned local energy from outputs["node_energy"][:num_local].sum()
-> owned local forces from -d(owned_local_energy)/d(local_positions)

The central purpose of this file is to verify that gradient flow reaches the
owned local positions through the exchange-aware computation graph.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

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
        "Phase 3 owned-force prototype: build the Phase 1 local+halo MACE view, "
        "form owned local energy from public per-atom node_energy, and recover "
        "owned local forces by differentiating that scalar with respect to local positions."
    )
    return parser


class MACEPotentialModelWithOwnedEnergyForces(MACEPotentialModelWithRawExchange):
    """Prototype wrapper that differentiates owned local energy for forces."""

    ownership_boundary = "public_node_energy"
    ownership_rule = "owned_local_energy = outputs['node_energy'][:num_local].sum()"
    force_rule = "owned_local_forces = -grad(owned_local_energy, local_positions)"

    def _build_differentiable_batch_dict(
        self,
        exchanged_atomic_numbers: torch.Tensor,
        exchanged_positions: torch.Tensor,
        distributed_state,
    ) -> tuple[Dict[str, torch.Tensor], int]:
        """Build a MACE batch dict while preserving autograd on positions.

        AtomicData.from_config needs NumPy positions to construct the graph
        topology, which severs autograd from the live local/halo position
        tensor. For Step 5 we keep that topology-building path, then replace the
        resulting batch_dict["positions"] with the differentiable
        ``exchanged_positions`` tensor before the MACE forward.
        """

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

        batch = self.Batch.from_data_list([atomic_data]).to(exchanged_positions.device)  # type: ignore[attr-defined]
        batch_dict = batch.to_dict()
        batch_dict["positions"] = exchanged_positions
        batch_dict = align_batch_dtype(batch_dict, self.model)
        return batch_dict, edge_count

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
        exchanged_positions.retain_grad()

        exchanged_atomic_numbers = self._map_atomic_numbers(exchanged_atom_types)
        n_local = int(local_positions.shape[0])
        n_total = int(exchanged_positions.shape[0])
        n_halo = int(n_total - n_local)

        batch_dict, edge_count = self._build_differentiable_batch_dict(
            exchanged_atomic_numbers,
            exchanged_positions,
            distributed_state,
        )
        outputs = self.model(
            batch_dict,
            training=False,
            compute_force=False,
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

        local_position_grads, exchanged_position_grads = torch.autograd.grad(
            owned_local_energy,
            [local_positions, exchanged_positions],
            retain_graph=False,
            allow_unused=False,
        )

        owned_local_forces = -local_position_grads
        halo_position_grads = exchanged_position_grads[n_local:]

        local_grad_is_finite = bool(torch.isfinite(local_position_grads).all().item())
        exchanged_grad_is_finite = bool(torch.isfinite(exchanged_position_grads).all().item())
        halo_grad_norm = float(halo_position_grads.norm().detach().cpu().item()) if n_halo > 0 else 0.0
        local_grad_norm = float(local_position_grads.norm().detach().cpu().item())
        local_force_norm = float(owned_local_forces.norm().detach().cpu().item())
        halo_gradient_active = bool(halo_grad_norm > 0.0) if n_halo > 0 else False

        self.last_observation = {
            "rank": int(distributed_state.dist_env.global_rank),
            "n_local": n_local,
            "n_total_after_exchange": n_total,
            "n_halo": n_halo,
            "expected_total_recv": self.total_recv,
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
            "force_rule": self.force_rule,
            "node_energy_shape": list(node_energy.shape),
            "owned_node_energy_shape": list(owned_node_energy.shape),
            "recovered_force_shape": list(owned_local_forces.shape),
            "raw_total_energy": float(raw_total_energy.detach().cpu().item()),
            "owned_local_energy": float(owned_local_energy.detach().cpu().item()),
            "raw_minus_owned_energy": float(
                (raw_total_energy - owned_local_energy).detach().cpu().item()
            ),
            "local_grad_is_finite": local_grad_is_finite,
            "exchanged_grad_is_finite": exchanged_grad_is_finite,
            "local_grad_norm": local_grad_norm,
            "halo_grad_norm": halo_grad_norm,
            "local_force_norm": local_force_norm,
            "halo_gradient_active": halo_gradient_active,
            "backward_depends_on_halo": halo_gradient_active,
        }

        return owned_local_energy.detach(), owned_local_forces.detach()


from distnnp_model.nnp.MACE_Adaptation import phase1_raw_exchange_adapter as _phase1

MACEPotentialModelWithOwnedEnergyForces.AtomicData = _phase1.AtomicData
MACEPotentialModelWithOwnedEnergyForces.Batch = _phase1.Batch


def summarize_rank_results(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    observation_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rank_records_sorted = sorted(observation_records, key=lambda item: item["rank"])
    force_checks = {
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
        "any_halo_gradient_active": any(
            record["halo_gradient_active"] for record in rank_records_sorted
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
        "force_checks": force_checks,
        "rank_records": rank_records_sorted,
    }


def print_summary(summary: Dict[str, Any], args) -> None:
    print("============================================================")
    print("Phase 3 Owned-Force MACE Prototype")
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
    print("Force Checks")
    print("------------------------------------------------------------")
    checks = summary["force_checks"]
    print(f"all_expected_halo_counts_match: {checks['all_expected_halo_counts_match']}")
    print(f"all_owned_energies_finite: {checks['all_owned_energies_finite']}")
    print(f"all_raw_energies_finite: {checks['all_raw_energies_finite']}")
    print(
        "all_recovered_force_shapes_match_owned_atoms: "
        f"{checks['all_recovered_force_shapes_match_owned_atoms']}"
    )
    print(f"all_local_grads_finite: {checks['all_local_grads_finite']}")
    print(f"all_exchanged_grads_finite: {checks['all_exchanged_grads_finite']}")
    print(f"any_halo_gradient_active: {checks['any_halo_gradient_active']}")
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
                "Phase 3 owned-force adapter requires a distributed launch. "
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

        adapter = MACEPotentialModelWithOwnedEnergyForces(raw_model)
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
