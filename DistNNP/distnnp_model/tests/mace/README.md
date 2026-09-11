# MACE validation

`MACEModel` is validated with direct Python and `torchrun` commands. Run the
commands below from the `DistNNP` repository root in an environment where the
project packages and the optional MACE dependency are installed.

## Static parity

Generate an independent public-MACE reference artifact, then require the
two-rank DistNNP model to match it:

```bash
RESULT_DIR=./tmp/mace_static
mkdir -p "${RESULT_DIR}"

python distnnp_model/tests/mace/mace_standalone_reference.py \
  --model-family mp --model medium-mpa-0 \
  --output "${RESULT_DIR}/reference.pt"

torchrun --standalone --nproc_per_node=2 \
  distnnp_model/tests/mace/test_mace_model.py \
  run-distributed \
  --model-family mp --model medium-mpa-0 \
  --reference-artifact "${RESULT_DIR}/reference.pt" \
  --strict-parity \
  --output-json "${RESULT_DIR}/mace_summary.json"
```

Use `--xyz-path` to supply an XYZ/extxyz structure. The static driver also
supports `--builtin-system`, `--head-name`, and `--ghost-cutoff-buffer`.

## Static-driver controls

Use `--builtin-system` or `--xyz-path` to select a structure, and keep
`--model-family`, `--model`, and optional `--head-name` identical in both
commands. `--ghost-cutoff-buffer` controls the distributed halo margin.
`--benchmark` with its warmup, iteration, and sample flags records timing
statistics; it does not relax `--strict-parity`. `--time-initialize-buffers`
additionally profiles buffer setup. `--output-json PATH` writes the
corresponding summary.

The reference artifact signature includes positions, cell, PBC flags, and
model identity. Both commands in a reference/distributed pair must therefore
use the same structure and model arguments. Nonperiodic XYZ/extxyz inputs must
provide a valid cell containing their coordinates.

## Distributed trajectory

Run the two-rank MACEModel trajectory, then compare it with a one-rank public
MACE calculator trajectory:

```bash
RESULT_DIR=./tmp/mace_trajectory
mkdir -p "${RESULT_DIR}"

torchrun --standalone --nproc_per_node=2 \
  distnnp_model/tests/mace/run_distributed_trajectory.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0

python distnnp_model/tests/mace/test_md.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0 \
  --reference-mode monolithic
```

The acceptance trajectory is a deterministic 216-atom periodic diamond-Si
system with 100 steps, a 0.5 fs timestep, and rebalancing every seven steps.
The generated artifact records model identity, structure hash, rank count, and
force-unit convention before numerical comparison begins.

## Benchmark collection

Benchmarking is opt-in and does not change parity acceptance. Add
`--benchmark --benchmark-warmup 5 --benchmark-iterations 20
--benchmark-samples 7` to both static-parity commands. Use the same model and
structure on both sides of the comparison.

## Coverage boundary

Static parity anchors `MACEModel` to public MACE calculator and raw-core model
values. Trajectory validation covers distributed partitioning, halo and feature
exchange, rank-local energy aggregation, repartitioning, and Verlet
integration against monolithic MACE. `MACEModel` returns raw MACE energy and
force units; the trajectory harness applies its explicit Verlet force-unit
conversion only for integration.
