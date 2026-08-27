# DistMACE Top-Down Adapter

------------------------------------------------------------

## Purpose

This folder now centers on the current MP/MPA distributed MACE adapter:

- `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/distmace_top_down_adapter.py`
- `distnnp_model/tests/test_distmace_top_down_adapter.py`

The adapter is designed to let DistNNP drive:

- partition ownership
- halo communication
- rank-local structural reconstruction
- owned-energy semantics

while MACE still provides:

- node embedding
- edge featurization
- interaction and product blocks
- per-node energy prediction

This README is meant to help collaborators understand where the adapter is
today, how its internal pieces fit together, and how to run validation and
benchmark experiments against standalone MACE.

------------------------------------------------------------

## Current Scope

The current implementation is intentionally scoped to MP/MPA-style foundation
models loaded through:

- `load_raw_mace_model(...)` in `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/distmace_top_down_adapter.py`

and validated against the `ScaleShiftMACE`-compatible contract enforced in:

- `DistMACETopDownAdapter._validate_model_contract(...)` in `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/distmace_top_down_adapter.py`

In practice, this means the adapter is aimed at checkpoints such as:

- `medium-mpa-0`
- `medium-0b3`
- `large`

It is not yet a fully generic MACE-family adapter across every model family or
specialized conditioning path.

------------------------------------------------------------

## File Map

Primary implementation:

- `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/distmace_top_down_adapter.py`

Primary distributed test and benchmark driver:

- `distnnp_model/tests/test_distmace_top_down_adapter.py`

Historical design notes and earlier milestones:

- `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/TOP_DOWN_SINGLE_FILE_ADAPTER_DESIGN.md`
- `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/DISTMACE PERFORMANCE REDESIGN.txt`
- `distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/FINAL_ADAPTER_README.md`

Those older files are still useful for history, but the current operational
surface is the top-down adapter and its dedicated test driver.

------------------------------------------------------------

## Utilities And Top-Level Helpers

The top of the adapter file contains the utilities that the rest of the
implementation builds on.

### Model loading

- `load_raw_mace_model(...)`

This loads a raw public MACE foundation model without the ASE calculator
wrapper. It currently supports the `mp`, `off`, `omol`, and `polar` loader
names, although the adapter contract itself is only validated for the MP/MPA
style path.

### Structural view containers

- `StructuralHaloView`
- `ExplicitOwnedGraph`

These are the two main adapter-side dataclasses:

- `StructuralHaloView` stores the local positions, exchanged positions,
  exchanged atomic numbers, and local/halo/global indexing information.
- `ExplicitOwnedGraph` stores the rank-local MACE graph tensors that are
  ultimately passed into the model forward.

### Geometry and partition helpers

- `canonicalize_nonperiodic_geometry(...)`
- `validate_partition_info(...)`

`canonicalize_nonperiodic_geometry(...)` translates nonperiodic coordinates
into a padded orthorhombic box so DistNNP’s current radius-graph behavior does
not wrap atoms unexpectedly.

`validate_partition_info(...)` checks the send/recv CSR metadata used by both
structural exchange and learned-feature exchange. This is one of the most
important early correctness gates in the file.

### Head and contract helpers

- `resolve_model_head_name(...)`
- `resolve_model_head_index(...)`
- `summarize_model_contract(...)`

These helpers are shared by both the adapter and the test file so that:

- head selection stays consistent across distributed and monolithic paths
- model structure is summarized the same way in test output and diagnostics

### Diagnostic record helpers

- `build_graph_authority_record(...)`
- `make_adapter_observation(...)`
- `finalize_adapter_observation(...)`

These build the structured per-rank observation record that later feeds the
validation summary in the test harness.

------------------------------------------------------------

## Runtime Mixin

The model-internal distributed behavior lives in:

- `DistScaleShiftMACERuntimeMixin`

This mixin is dynamically installed onto the loaded raw MACE model so the
adapter can keep the normal MACE execution loop while inserting one learned
feature exchange boundary.

### What the mixin owns

Runtime state and exchange context:

- `_configure_dist_runtime(...)`
- `_configure_feature_exchange_runtime(...)`

Debug and execution modes:

- `_set_debug_tracing(...)`
- `_set_execution_mode(...)`
- `_emit_debug_log(...)`

Interaction output normalization and exchanger setup:

- `_normalize_interaction_output(...)`
- `_ensure_feature_exchanger_ready(...)`

Exchange policy:

- `_should_exchange_after_product(...)`
- `_exchange_after_product(...)`

Forward override:

- `forward(...)`

### What the mixin actually changes

The mixin keeps the native MACE sequence:

1. `prepare_graph(...)`
2. node embedding
3. spherical harmonics and radial embedding
4. interaction/product loop
5. readouts
6. per-node energy accumulation

The distributed change is that after each product block with a downstream
consumer, the mixin may exchange owned local node features through:

- `FeaturePushToExchange.apply(...)`

Today, for the current two-interaction MP/MPA checkpoints, this means the
exchange happens once after layer 0 and before layer 1 consumes the updated
remote features.

### Execution modes

The mixin supports three execution modes:

- `validate`
- `benchmark`
- `debug`

The most important practical difference is in:

- `forward(...)`

In `benchmark` mode, the forward returns early with:

- `owned_node_energy`
- `owned_local_energy`

and skips richer graph-level outputs that the adapter does not consume during
timed runs.

------------------------------------------------------------

## Adapter Class

The DistNNP-facing orchestrator is:

- `DistMACETopDownAdapter`

This is the class collaborators should think of as "the adapter".

### Construction and contract

Key entry points:

- `__init__(...)`
- `_validate_model_contract(...)`
- `_install_runtime_subclass(...)`

Important design points:

- the adapter currently accepts a raw MACE model object
- it validates the MP/MPA-facing `ScaleShiftMACE` contract up front
- it dynamically installs the runtime mixin onto that model

### Buffer and mode control

- `set_diagnostics(...)`
- `set_debug_tracing(...)`
- `set_execution_mode(...)`
- `initialize_buffers(...)`

`initialize_buffers(...)` is where the adapter binds the current partition to:

- position exchange
- atom-type exchange
- learned-feature exchange runtime metadata

### Structural reconstruction

The main structural helpers are:

- `_prepare_static_graph_cache(...)`
- `_halo_global_indices(...)`
- `_build_structural_view(...)`

This stage:

- takes the owned local positions from `DistNNPState`
- exchanges them into a `[local | halo]` structural view
- exchanges and caches atom-type-derived metadata
- optionally reconstructs halo global IDs for diagnostics

### Rank-local graph construction

- `_radius_graph(...)`
- `_build_owned_graph(...)`

This is the point where the adapter turns the structural halo view into the
actual MACE batch dictionary. The current policy is:

- build the graph over `[local | halo]`
- keep only edges whose destination is locally owned
- pass the resulting tensors to the MACE forward

### Force recovery and observations

- `_recover_forces(...)`
- `_build_observation(...)`
- `_execute_model(...)`
- `__call__(...)`

The core adapter contract is:

1. run the distributed MACE forward
2. recover owned local forces from autograd on `owned_local_energy`
3. return:
   - owned local energy
   - owned local forces

This is the main reason the adapter does not use the raw MACE force tensor as
its final distributed answer.

------------------------------------------------------------

## End-To-End Flow

The current top-down execution path is:

1. load a raw model through `load_raw_mace_model(...)`
2. build a `DistNNPState` in the test harness
3. construct `DistMACETopDownAdapter`
4. call `initialize_buffers(...)`
5. call `adapter(state)`
6. inside the adapter:
   - build the structural halo view
   - build the owned-destination graph
   - run the runtime-mixed MACE forward
   - recover forces by autograd

------------------------------------------------------------

## Generalizability Status

The adapter is more generic than the earlier single-exchange prototype, but it
is still intentionally limited.

### What is generalized already

- it no longer hardcodes a single head name; head resolution now goes through
  `resolve_model_head_name(...)`
- it supports readout layouts with either one final readout or one readout per
  interaction layer, validated in `DistMACETopDownAdapter._validate_model_contract(...)`
- it supports two-interaction MP/MPA checkpoints that differ in skip-connection
  behavior and hidden widths

### What is still intentionally narrow

- the model contract is still tied to MP/MPA-style `ScaleShiftMACE`
- force recovery is still centered on owned-energy autograd
- the graph policy is still owned-destination filtering after a full local plus
  halo radius-graph build
- the exchange policy is still effectively "after each product block with a
  downstream consumer", which for current MP/MPA models means one exchange

### What is left for a future generic adapter

- broader support beyond MP/MPA
- explicit support for charge/spin-conditioned variants
- a formal capability layer across different MACE families
- deeper exchange-policy generalization

------------------------------------------------------------

## Performance Notes

The current adapter includes a benchmark-oriented execution mode, but it is not
yet a thin, performance-first rewrite.

Relevant code:

- benchmark execution mode inside `DistScaleShiftMACERuntimeMixin.forward(...)`
- benchmark control in `benchmark_speedup(...)` from `distnnp_model/tests/test_distmace_top_down_adapter.py`

### Improvements already present

- debug-only CPU conversions are guarded
- diagnostics are disabled during timed runs
- static species-side graph metadata is cached
- benchmark mode skips richer outputs the adapter does not use

### Major remaining bottlenecks

- full radius-graph build over `[local | halo]`
- owned-destination edge filtering after graph construction
- synchronous feature exchange
- force recovery through a real backward pass

So collaborators should view the current implementation as:

- correctness-first
- benchmark-capable
- not yet the final performance architecture

------------------------------------------------------------

## Test Driver Overview

The main experimental entrypoint is:

- `build_parser(...)`
- `prepare_experiment(...)`
- `run_experiment(...)`

### What the test driver does

It combines four responsibilities:

1. parse CLI inputs
2. build the distributed `DistNNPState`
3. run the distributed adapter
4. optionally compare against monolithic MACE and benchmark the result

### Validation modes

CLI option:

- `--validation-mode {full,minimal}`

`minimal`:

- runs the distributed adapter
- checks structural and ownership diagnostics
- skips monolithic parity references

`full`:

- runs the distributed adapter
- builds monolithic MACE references through `monolithic_references(...)`
- compares energy, forces, node energies, and graph agreement

### Benchmark mode

Benchmark entrypoint:

- `benchmark_speedup(...)`

The benchmark:

- requires successful full validation first
- switches the adapter to `benchmark` mode
- disables diagnostics and debug tracing
- times the slowest rank as the distributed step metric
- compares that against a single-GPU standalone MACE baseline

------------------------------------------------------------

## Recommended Workflows

### 1. Quick distributed sanity check

Use this when you want to confirm the adapter still runs on a small built-in
system without paying the full monolithic reference cost.

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --builtin-system natural_chain12 \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode minimal \
  --no-benchmark
```

### 2. Full parity validation on the user’s target model

This is the recommended command for comparing the distributed adapter against
standalone MACE on `medium-mpa-0`.

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --builtin-system natural_chain12 \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity \
  --no-benchmark
```

### 3. Full validation on the 3750-atom benchmark structure

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --xyz-path distnnp_model/tests/data/mace/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity \
  --no-benchmark
```

### 4. Full validation plus benchmark

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --xyz-path distnnp_model/tests/data/mace/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-iterations 20 \
  --benchmark-samples 7
```

### 5. Save a JSON report for collaborators

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --xyz-path distnnp_model/tests/data/mace/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-iterations 20 \
  --benchmark-samples 7 \
  --output-json results/distmace_medium_mpa0_2gpu.json
```

### 6. Debug a failing exchange

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/test_distmace_top_down_adapter.py \
  --builtin-system natural_chain12 \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode minimal \
  --no-benchmark \
  --debug-exchange
```

------------------------------------------------------------

## Notes For Standalone MACE Comparison

If the immediate goal is to compare the user’s standalone MACE
`medium-mpa-0` result against the distributed adapter, the most relevant path
is:

- distributed adapter run through `run_experiment(...)`
- monolithic references from `monolithic_references(...)`

During `full` validation, rank 0 reports:

- `distributed_energy`
- `explicit_distnnp_graph_energy`
- `standalone_mace_energy`
- `explicit_force_mae`
- `standalone_force_mae`

Those are the most useful fields for direct collaborator-facing comparison.

------------------------------------------------------------

## Practical Limitations

Collaborators should keep these caveats in mind:

- the adapter currently requires at least two ranks in the distributed test
  driver
- the performance benchmark is useful for trend comparison, but it is not yet
  the final thin-adapter performance story
- the validation path is intentionally richer and heavier than the benchmark
  path
- the current model support target is MP/MPA, not every MACE-family variant

------------------------------------------------------------

## Recommended Reading Order

For someone new to this adaptation, the most efficient reading order is:

1. this README
2. `DistMACETopDownAdapter`
3. `DistScaleShiftMACERuntimeMixin.forward(...)`
4. `build_parser(...)`
5. `run_experiment(...)`

That sequence usually gives enough context to run, validate, and interpret the
current adapter without first reading the older phase files.
