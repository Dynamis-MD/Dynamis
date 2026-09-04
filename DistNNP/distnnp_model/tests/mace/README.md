# MACE Tests

The primary MACE validation surface is the thin adapter, run through SLURM or
the equivalent `torchrun` commands. These drivers are not pytest entrypoints.
For pre-upstream Simulator builds, the trajectory helper supplies only the
missing `os` and `is_pathlike` module imports. It does not patch native
bindings, state tensors, or repartitioning behavior.

## Primary workflows

`slurm_distmace_thin_static.sh` generates an immutable standalone MACE
calculator/core-model artifact and runs strict distributed thin-adapter parity.
Set `ENABLE_BENCHMARK=1` to collect timings; benchmark values are observations,
not correctness assertions.

`slurm_distmace_thin_trajectory.sh` runs the standard two-rank trajectory and
then validates it against a singleton public MACE calculator trajectory. Both
runs use the deterministic 216-atom periodic diamond-Si fixture, 100 steps,
0.5 fs timestep, and scheduled rebalance every seven steps.

Run either script from a SLURM allocation after setting `PROJECT_ROOT` if it is
not the default cluster checkout. Results are written under `results/` with the
SLURM job ID.

## Direct commands

Static parity:

```bash
python distnnp_model/tests/mace/mace_standalone_reference.py --output tmp/reference.pt
torchrun --nproc_per_node=2 distnnp_model/tests/mace/test_distmace_thin_adapter.py \
  run-distributed --reference-artifact tmp/reference.pt --strict-parity
```

Trajectory acceptance:

```bash
torchrun --nproc_per_node=2 distnnp_model/tests/mace/run_distributed_trajectory.py \
  --path ./tmp/mace_md_si
python distnnp_model/tests/mace/test_md.py --path ./tmp/mace_md_si
```

`test_md.py` defaults to `--reference-mode monolithic`. Its explicit
`--reference-mode thin` option is for developer comparison only and is not the
acceptance oracle.

The producer saves `dist_traj.pt` as metadata plus frames. The reference runner
rejects mismatches in checkpoint identity, structure hash, step settings, rank
count, adapter identity, or force-unit convention before numerical comparison.

## Coverage boundaries

Static parity anchors the thin adapter to independent MACE calculator and raw
core-model values. Trajectory validation checks distributed partitioning, halo
and feature exchange, rank-local energy aggregation, repartitioning, and Verlet
integration against monolithic MACE. Adapter energy is rank-local; the producer
sums rank contributions into each saved total energy.

## Diagnostic fallback

`test_distmace_top_down_adapter.py` and its existing SLURM script remain
available for diagnostic investigation. They are not primary validation or
benchmark commands for the supported thin-adapter path.
