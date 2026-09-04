# MACE Tests

This directory contains the MACE-specific distributed validation and trajectory workflows.

## Main Files

- `test_distmace_thin_adapter.py`: thin-adapter parity driver, detached-reference comparison, and benchmark flow
- `mace_standalone_reference.py`: standalone MACE artifact generator used by the thin-adapter test
- `test_distmace_top_down_adapter.py`: top-down adapter parity and benchmark driver
- `run_distributed_trajectory.py`: distributed MD trajectory writer for the thin adapter
- `test_md.py`: singleton MD reference runner and trajectory comparison entrypoint
- `md_test_common.py`: shared MACE trajectory helpers, structure generation, comparison utilities, and simulator-compatibility shims
- `data/`: bundled MACE XYZ fixtures

## Typical Workflows

Thin parity:

```bash
python distnnp_model/tests/mace/mace_standalone_reference.py \
  --output tmp/reference.pt

torchrun --nproc_per_node=2 \
  distnnp_model/tests/mace/test_distmace_thin_adapter.py \
  run-distributed \
  --reference-artifact tmp/reference.pt \
  --strict-parity
```

Thin benchmark:

```bash
python distnnp_model/tests/mace/mace_standalone_reference.py \
  --xyz-path distnnp_model/tests/mace/data/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --output tmp/reference_3750.pt \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-iterations 20 \
  --benchmark-samples 7

torchrun --nproc_per_node=2 \
  distnnp_model/tests/mace/test_distmace_thin_adapter.py \
  run-distributed \
  --xyz-path distnnp_model/tests/mace/data/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --reference-artifact tmp/reference_3750.pt \
  --strict-parity \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-iterations 20 \
  --benchmark-samples 7
```

Top-down parity:

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/mace/test_distmace_top_down_adapter.py \
  --builtin-system natural_chain12 \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity
```

Top-down benchmark:

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/mace/test_distmace_top_down_adapter.py \
  --xyz-path distnnp_model/tests/mace/data/benchmark_scaling_molecular_3750.xyz \
  --model-family mp \
  --model medium-mpa-0 \
  --validation-mode full \
  --strict-parity \
  --benchmark \
  --benchmark-warmup 5 \
  --benchmark-iterations 20 \
  --benchmark-samples 7
```

Trajectory validation:

The trajectory workflow generates a deterministic, slightly perturbed 216-atom
periodic diamond-Si supercell for the MPA-0 materials baseline.

```bash
torchrun --nproc_per_node=2 \
  -m distnnp_model.tests.mace.run_distributed_trajectory \
  --path ./tmp/mace_md_si

python -m distnnp_model.tests.mace.test_md \
  --path ./tmp/mace_md_si
```

Trajectory validation with explicit script paths:

```bash
torchrun --nproc_per_node=2 \
  distnnp_model/tests/mace/run_distributed_trajectory.py \
  --path ./tmp/mace_md_si

python distnnp_model/tests/mace/test_md.py \
  --path ./tmp/mace_md_si
```
