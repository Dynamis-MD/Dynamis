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

## SLURM orchestration

Submit the default acceptance workflows from the repository root:

```bash
sbatch distnnp_model/scripts/slurm_distmace_thin_static.sh
sbatch distnnp_model/scripts/slurm_distmace_thin_trajectory.sh
```

The scripts use the cluster checkout by default. Override settings at submit
time rather than editing the scripts:

```bash
PROJECT_ROOT=/path/to/DistNNP/DistNNP \
MODEL_FAMILY=mp MODEL_NAME=medium-mpa-0 \
sbatch distnnp_model/scripts/slurm_distmace_thin_static.sh

RESULT_DIR=/scratch/$USER/distmace_trajectory_run \
sbatch distnnp_model/scripts/slurm_distmace_thin_trajectory.sh
```

| Variable | Static parity | Trajectory | Meaning |
| --- | --- | --- | --- |
| `PROJECT_ROOT` | yes | yes | Repository root; defaults to the cluster checkout. |
| `RESULT_DIR` | yes | yes | Artifact directory; defaults to a job-ID-specific directory under `results/`. |
| `MODEL_FAMILY` | yes | yes | MACE foundation family; default `mp`. |
| `MODEL_NAME` | yes | yes | Checkpoint name; default `medium-mpa-0`. |
| `NPROC_PER_NODE` | yes | yes | GPU rank count; trajectory rejects any value other than `2`. |
| `XYZ_PATH` | yes | no | Optional static-parity XYZ/extxyz input; overrides the built-in structure. |
| `ENABLE_BENCHMARK` | yes | no | Set to `1` to record benchmark samples after strict parity. |
| `MASTER_PORT` | yes | yes | Optional explicit torchrun rendezvous port. |

Slurm writes standard output to its scheduler-default `slurm-<jobid>.out` file
unless an `--output` setting is supplied at submission. The scripts also print
the final result directory on success.

Static results contain `standalone_reference.pt` and `thin_summary.json`.
Trajectory results contain `silicon_supercell.xyz`, `dist_traj.pt`, `dist_out/`,
and `ref_out/comparison_report.csv` after the monolithic comparison succeeds.

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

Static benchmark collection is explicit on both sides of the comparison:

```bash
python distnnp_model/tests/mace/mace_standalone_reference.py \
  --xyz-path distnnp_model/tests/mace/data/benchmark_scaling_molecular_3750.xyz \
  --model-family mp --model medium-mpa-0 --output tmp/reference_3750.pt \
  --benchmark --benchmark-warmup 5 --benchmark-iterations 20 --benchmark-samples 7

torchrun --nproc_per_node=2 distnnp_model/tests/mace/test_distmace_thin_adapter.py \
  run-distributed \
  --xyz-path distnnp_model/tests/mace/data/benchmark_scaling_molecular_3750.xyz \
  --model-family mp --model medium-mpa-0 --reference-artifact tmp/reference_3750.pt \
  --strict-parity --benchmark --benchmark-warmup 5 --benchmark-iterations 20 \
  --benchmark-samples 7
```

Useful static-driver options are `--builtin-system natural_chain12`,
`--xyz-path`, `--head-name`, `--ghost-cutoff-buffer`,
`--nonperiodic-padding`, and `--output-json`. Keep
`--disable-nonperiodic-canonicalization` for local investigation only; it is
not part of the standard acceptance workflow.

`test_md.py` defaults to `--reference-mode monolithic`. Its explicit
`--reference-mode thin` option is for developer comparison only and is not the
acceptance oracle.

The producer saves `dist_traj.pt` as metadata plus frames. The reference runner
rejects mismatches in checkpoint identity, structure hash, step settings, rank
count, adapter identity, or force-unit convention before numerical comparison.

The standardized trajectory CLI intentionally exposes only `--path`,
`--model-family`, `--model`, and `--head-name`. It always uses the Si fixture,
100 steps, a 0.5 fs timestep, and scheduled rebalancing. The singleton command
uses monolithic MACE by default; do not pass `--head-name` to that mode because
the public calculator reference does not currently select a named head.

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
