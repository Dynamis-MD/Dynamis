"""Validate Phase 3 owned local forces against monolithic MACE forces.

This script is designed for the current MACE adaptation experiments rather than
for broad default test execution. It can be run either:

1. Directly under ``torchrun --standalone --nproc_per_node=2`` for the current
   two-rank prototype topology, or
2. Through pytest in an environment that has:
   - at least 2 CUDA GPUs
   - ``mace-torch`` installed
   - the DistNNP extensions available

The validation flow is:

- build the same distributed state used by the Phase 3 owned-force adapter
- run the distributed owned-force prototype on each rank
- gather owned local forces and partition indices back to rank 0
- reconstruct a full-system force tensor in canonical atom order
- compute a monolithic single-rank MACE force reference on the same structure
- compare distributed owned forces against the monolithic forces
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist

from distnnp_state import SFCCurveType, SFCPartitioner, TorchDistEnv

from distnnp_model.nnp.MACE_Adaptation.phase1_raw_exchange_adapter import (
    analyze_structure,
    build_global_molecular_info,
    build_phase1_distnnp_state,
    build_sim_config,
    finalize_distributed,
    load_raw_mace_model,
    parse_raw_structure,
)
from distnnp_model.nnp.MACE_Adaptation.phase3_owned_force_adapter import (
    MACEPotentialModelWithOwnedEnergyForces,
    build_parser as build_phase3_parser,
)
from distnnp_model.nnp.MACE_Adaptation.standalone_direct_model_driver import (
    align_batch_dtype,
    build_batch,
    build_configuration,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for distributed owned-force validation.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_phase3_parser()
    parser.description = (
        "Validate Phase 3 distributed owned local forces against monolithic "
        "single-rank MACE forces on the same structure."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for distributed-vs-monolithic force comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance for distributed-vs-monolithic force comparison.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the validation summary from rank 0.",
    )
    return parser


def compute_monolithic_reference_forces(
    raw_structure: Dict[str, Any],
    raw_model,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the full unpartitioned structure through monolithic MACE."""

    config = build_configuration(raw_structure)
    batch = build_batch(config, raw_model, str(device))
    batch_dict = align_batch_dtype(batch.to_dict(), raw_model)

    outputs = raw_model(
        batch_dict,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )

    forces = outputs["forces"].detach()
    energy = outputs["energy"].detach()
    if torch.is_tensor(energy) and energy.numel() == 1:
        energy = energy.reshape(())

    return {
        "energy": float(energy.cpu().item()),
        "forces": forces.cpu(),
        "forces_shape": list(forces.shape),
    }


def maybe_gather_objects(local_object: Dict[str, Any]) -> List[Dict[str, Any]]:
    if dist.is_initialized():
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_object)
        return [item for item in gathered if item is not None]
    return [local_object]


def reconstruct_owned_force_tensor(
    gathered_rank_records: List[Dict[str, Any]],
    total_atoms: int,
) -> torch.Tensor:
    full_forces = torch.zeros(total_atoms, 3, dtype=torch.float32)
    for record in gathered_rank_records:
        partition_indices = torch.tensor(record["partition_indices"], dtype=torch.long)
        owned_forces = torch.tensor(record["owned_forces"], dtype=torch.float32)
        full_forces.index_copy_(0, partition_indices, owned_forces)
    return full_forces


def summarize_validation(
    raw_structure: Dict[str, Any],
    structure_diagnostics: Dict[str, Any],
    gathered_rank_records: List[Dict[str, Any]],
    monolithic_reference: Dict[str, Any],
    reconstructed_owned_forces: torch.Tensor,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    monolithic_forces = monolithic_reference["forces"].to(torch.float32)
    force_delta = reconstructed_owned_forces - monolithic_forces
    max_abs_force_error = float(force_delta.abs().max().item())
    mean_abs_force_error = float(force_delta.abs().mean().item())
    distributed_force_norm = float(reconstructed_owned_forces.norm().item())
    monolithic_force_norm = float(monolithic_forces.norm().item())
    allclose = bool(
        torch.allclose(
            reconstructed_owned_forces,
            monolithic_forces,
            atol=atol,
            rtol=rtol,
        )
    )

    return {
        "source_path": raw_structure.get("source_path", "unknown"),
        "num_atoms": len(raw_structure["atomic_numbers"]),
        "atomic_numbers": list(raw_structure["atomic_numbers"]),
        "positions_A": raw_structure["positions"],
        "structure_diagnostics": structure_diagnostics,
        "atol": atol,
        "rtol": rtol,
        "distributed_checks": {
            "all_ranks_report_halo_dependency": all(
                record["backward_depends_on_halo"] for record in gathered_rank_records
            ),
            "all_ranks_local_grads_finite": all(
                record["local_grad_is_finite"] for record in gathered_rank_records
            ),
            "all_ranks_exchanged_grads_finite": all(
                record["exchanged_grad_is_finite"] for record in gathered_rank_records
            ),
        },
        "comparison": {
            "forces_allclose": allclose,
            "max_abs_force_error": max_abs_force_error,
            "mean_abs_force_error": mean_abs_force_error,
            "distributed_force_norm": distributed_force_norm,
            "monolithic_force_norm": monolithic_force_norm,
            "monolithic_energy": monolithic_reference["energy"],
            "summed_owned_local_energy": float(
                sum(record["owned_local_energy"] for record in gathered_rank_records)
            ),
        },
        "rank_records": gathered_rank_records,
        "reconstructed_owned_forces": reconstructed_owned_forces.tolist(),
        "monolithic_forces": monolithic_forces.tolist(),
        "force_delta": force_delta.tolist(),
    }


def print_summary(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print("============================================================")
    print("Phase 3 Distributed vs Monolithic Force Validation")
    print("============================================================")
    print(f"source_path: {summary['source_path']}")
    print(f"num_atoms: {summary['num_atoms']}")
    print(f"atol: {summary['atol']}")
    print(f"rtol: {summary['rtol']}")
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
    print("Distributed Checks")
    print("------------------------------------------------------------")
    checks = summary["distributed_checks"]
    print(f"all_ranks_report_halo_dependency: {checks['all_ranks_report_halo_dependency']}")
    print(f"all_ranks_local_grads_finite: {checks['all_ranks_local_grads_finite']}")
    print(f"all_ranks_exchanged_grads_finite: {checks['all_ranks_exchanged_grads_finite']}")
    print()
    print("------------------------------------------------------------")
    print("Force Comparison")
    print("------------------------------------------------------------")
    comparison = summary["comparison"]
    print(f"forces_allclose: {comparison['forces_allclose']}")
    print(f"max_abs_force_error: {comparison['max_abs_force_error']}")
    print(f"mean_abs_force_error: {comparison['mean_abs_force_error']}")
    print(f"distributed_force_norm: {comparison['distributed_force_norm']}")
    print(f"monolithic_force_norm: {comparison['monolithic_force_norm']}")
    print(f"monolithic_energy: {comparison['monolithic_energy']}")
    print(f"summed_owned_local_energy: {comparison['summed_owned_local_energy']}")


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    raw_structure = parse_raw_structure(args)
    structure_diagnostics = analyze_structure(raw_structure)

    is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    torch_dist_env = TorchDistEnv(is_distributed=is_distributed)
    device = torch_dist_env.device if args.device == "cuda" else torch.device(args.device)

    if not dist.is_initialized():
        raise RuntimeError(
            "This validation requires a distributed launch. "
            "Run it under torchrun or an equivalent multi-rank launcher."
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
    partitioner = SFCPartitioner(SFCCurveType.Morton)
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
    owned_energy, owned_forces = adapter(distributed_state)

    local_record = adapter.last_observation | {
        "partition_indices": distributed_state.local_partition_info.partition_indices.detach()
        .cpu()
        .tolist(),
        "owned_forces": owned_forces.detach().cpu().tolist(),
        "owned_energy_returned": float(owned_energy.detach().cpu().item()),
    }
    gathered_rank_records = maybe_gather_objects(local_record)

    if dist.get_rank() != 0:
        return {}

    monolithic_reference = compute_monolithic_reference_forces(
        raw_structure,
        raw_model,
        device,
    )
    reconstructed_owned_forces = reconstruct_owned_force_tensor(
        gathered_rank_records,
        total_atoms=len(raw_structure["atomic_numbers"]),
    )
    return summarize_validation(
        raw_structure=raw_structure,
        structure_diagnostics=structure_diagnostics,
        gathered_rank_records=gathered_rank_records,
        monolithic_reference=monolithic_reference,
        reconstructed_owned_forces=reconstructed_owned_forces,
        atol=args.atol,
        rtol=args.rtol,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run_validation(args)
        if dist.is_initialized() and dist.get_rank() == 0:
            print_summary(summary, args)
            if args.output_json:
                output_path = Path(args.output_json)
                output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print()
                print(f"Wrote JSON summary to {output_path}")
    finally:
        finalize_distributed()


def test_phase3_owned_forces_match_monolithic_mace():
    """Pytest entry point for environments that already provide torchrun-style setup.

    This function is intentionally conservative: the main validation path is the
    direct script execution under ``torchrun``. Keeping this pytest entry point
    small avoids adding noisy custom process orchestration on top of the current
    adaptation experiments.
    """

    if not dist.is_initialized():
        pytest.skip("Run this validation under torchrun or an initialized distributed context.")

    args = build_parser().parse_args([])
    summary = run_validation(args)
    if dist.get_rank() == 0:
        assert summary["distributed_checks"]["all_ranks_report_halo_dependency"]
        assert summary["distributed_checks"]["all_ranks_local_grads_finite"]
        assert summary["distributed_checks"]["all_ranks_exchanged_grads_finite"]
        assert summary["comparison"]["forces_allclose"], (
            "Distributed owned forces do not match monolithic MACE forces "
            f"within atol={summary['atol']} rtol={summary['rtol']}."
        )


if __name__ == "__main__":
    main()

