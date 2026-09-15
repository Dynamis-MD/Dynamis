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

## cuEquivariance

The static reference and distributed drivers both accept `--enable-cueq`.
When selected, the raw MACE model is converted with MACE's
`convert_e3nn_cueq` conversion before either path is timed or evaluated. The
flag is part of the reference-artifact signature, so a CuEq reference can only
be compared with a CuEq distributed run. Omitting the flag retains the e3nn
implementation.

Install mutually compatible versions of all four CuEq packages. The validated
cluster environment currently uses version `0.9.1` for
`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-cu12`, and
`cuequivariance-ops-torch-cu12` with Python 3.12, PyTorch 2.8.0+cu129, and
MACE 0.3.16.

Generate a CuEq-enabled standalone reference and run the two-GPU benchmark:

```bash
RESULT_DIR=./tmp/mace_cueq_2gpu
XYZ_PATH=distnnp_model/tests/mace/data/benchmark_scaling_molecular_7500.xyz
mkdir -p "${RESULT_DIR}"

python distnnp_model/tests/mace/mace_standalone_reference.py \
  --xyz-path "${XYZ_PATH}" --model-family mp --model medium-mpa-0 \
  --enable-cueq --output "${RESULT_DIR}/reference.pt" \
  --benchmark --benchmark-warmup 10 --benchmark-iterations 50 \
  --benchmark-samples 7 --output-json "${RESULT_DIR}/reference.json"

torchrun --standalone --nproc_per_node=2 \
  distnnp_model/tests/mace/test_mace_model.py run-distributed \
  --xyz-path "${XYZ_PATH}" --model-family mp --model medium-mpa-0 \
  --enable-cueq --reference-artifact "${RESULT_DIR}/reference.pt" \
  --strict-parity --benchmark --benchmark-warmup 10 \
  --benchmark-iterations 50 --benchmark-samples 7 \
  --output-json "${RESULT_DIR}/distributed.json"
```

For a four-GPU strong-scaling run, generate the same standalone reference and
change only the distributed launch to `--nproc_per_node=4`. Keep the XYZ,
model, head, dtype, CuEq setting, warmup, iterations, and sample count fixed
when comparing rank counts. The 7,500-atom fixture is preferred over the
3,750-atom fixture for two-versus-four-GPU scaling because the smaller system
does not provide enough per-rank compute to amortize halo and synchronization
costs.

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

The uninterrupted `distributed_median_ms_per_step` is the primary end-to-end
measurement and uses the slowest rank. The static driver also records a
diagnostic per-rank decomposition into structural/position exchange, graph
construction, MACE forward/feature exchange, and force recovery/backward
exchange. These diagnostic stages synchronize the device after every boundary
and therefore should be used to attribute costs, not substituted for the
uninterrupted result. Workload fields include owned and halo atoms, graph-edge
counts, rows exchanged per peer, feature width, and logical feature payload
bytes.

## Coverage boundary

Static parity anchors `MACEModel` to public MACE calculator and raw-core model
values. Trajectory validation covers distributed partitioning, halo and feature
exchange, rank-local energy aggregation, repartitioning, and Verlet
integration against monolithic MACE.

`MACEModel` returns DistNNP MD units — energy in `amu*A^2/fs^2` and forces in
`amu*A/fs^2` — matching `UMAModel` and `ORBModel`, so it drives `Simulator`
directly with no wrapper. Public MACE predicts eV and eV/Angstrom, so every
reference in this directory is scaled by
`distnnp_model.units.EV_PER_A_TO_AMU_A_PER_FS2` before it is compared. That
constant is defined once and imported by all three backends; do not re-spell
the literal.
