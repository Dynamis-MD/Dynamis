# Running the `distnnp` tests

These tests exercise the `Simulator` end to end: partitioning, halo exchange,
force evaluation and integration. They are **distributed tests** and must be
launched with `torchrun` on at least 2 GPUs.

All commands below are run from the repo's `DistNNP/` directory.

## Prerequisites

### 1. Install the sub-packages

```bash
pip install -r requirements-dev.txt
```

Reinstall the affected package after any C++/CUDA change:

```bash
pip install -e ./dynamis_state
pip install -e ./distnnp_integrator
pip install -e ./distnnp_model
pip install -e ./distnnp
```

### 2. Get 2 GPUs

The tests need a real multi-rank setup — on a single rank the exchanger
short-circuits to identity and nothing meaningful is being tested. On SLURM:

```bash
salloc --account=<account> --partition=<gpu-partition> \
       --nodes=1 --gpus-per-node=2 --time=01:00:00 --mem=64g
```

### 3. Generate the reference fixtures

The regression tests compare against a single-GPU baseline that is **not**
checked in. Generate it once (and again whenever the physics changes):

```bash
torchrun --nproc_per_node=1 distnnp/tests/fixtures/generate_reference.py
```

Note `--nproc_per_node=1` — this is the single-rank reference. It writes
`lj_reference_euler.pt`, `lj_reference_verlet.pt` and `lj_reference_langevin.pt`
into `distnnp/tests/fixtures/`. The Langevin file takes considerably longer
than the other two, since it runs several long trajectories to build its
statistical baseline.

If you skip this step the regression tests fail on a missing file rather than
on a real regression.

## Running

```bash
# everything
torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v

# skip the noisy timing tests (recommended day to day)
torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v -m "not perf"

# only the timing tests
torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v -m perf

# a single file, or a single test
torchrun --nproc_per_node=2 -m pytest distnnp/tests/test_simulation_regression.py -v
torchrun --nproc_per_node=2 -m pytest distnnp/tests/test_simulation_regression.py -v -k euler
```

To run the other sub-packages' tests as well, drop the path argument —
`pytest.ini` sets `testpaths` to all four packages:

```bash
torchrun --nproc_per_node=2 -m pytest -v
```

## Reading the output

**Every rank runs the whole suite.** pytest is not rank-aware, so `torchrun
--nproc_per_node=2` gives you two independent pytest sessions and therefore two
of everything — two progress lines, two summaries. This is expected. The ranks
are doing genuinely different work (each owns half the atoms); they just both
report it.

**Tests that assert only on rank 0** (`test_energy_matches_baseline`, the
Langevin KS test) return early on rank 1, so a mixed PASSED/PASSED-but-quieter
picture across the two summaries is normal.

**A non-zero exit from `torchrun`** shows up as a `ChildFailedError` traceback
with `<NO_OTHER_FAILURES>`. That is just torchrun reporting that one rank's
pytest exited non-zero; the real failure is in that rank's pytest output above
it. Scroll up.

## What the tests cover

| File | What it checks |
|---|---|
| `test_simulation_distributed.py` | Smoke tests — the simulator constructs and steps without error, energy is a scalar, forces have the right shape, nothing is NaN/Inf. |
| `test_simulation_regression.py` | Positions, forces and energy match the single-GPU reference after 1/5/10/15/20 steps. Plus a Kolmogorov-Smirnov test for Langevin. |
| `test_simulation_performance.py` | Per-step and 100-step wall time against recorded bounds. Marked `perf`. |

Every test is parametrized over all three integrators (`euler`, `verlet`,
`langevin`) by the `simulation_factory` fixture in `conftest.py`.

### Why Langevin is skipped in most regression tests

Langevin is stochastic, so a multi-GPU run cannot reproduce a single-GPU
trajectory step for step — the noise draws differ. Those tests `skip` for
Langevin by design and it gets `test_langevin_matches_baseline` instead, which
compares the *distributions* of time-averaged KE and PE across independent runs
using a 2-sample KS test. Seeing Langevin skips in the summary is correct
behaviour, not a broken fixture.

### Why the perf tests fail sometimes

The bounds are `mean + 1.5*std` over 20 baseline runs on 2x A100s, so a single
run can exceed them without anything being wrong — single-step CUDA-event
timings in particular are noisy. Treat one failure as inconclusive and re-run a
couple of times; only investigate if it fails consistently. Re-baseline the
numbers in `test_simulation_performance.py` if the hardware or system size
changes.

## The reference system

Defined in `conftest.py` and mirrored exactly in `generate_reference.py`:
16 argon atoms (atomic number 18) placed randomly in an 8x8x8 Å box with
`seed=100`, `dt=1e-3`, cutoff 3.5 Å, 0.5 Å ghost buffer, `simple` partitioning.
It is deliberately tiny — these are correctness tests, not a benchmark.

If you change any of that, the checked-in bounds and the reference fixtures
both become invalid and need regenerating.

## Troubleshooting

**Everything skips with "Not running under torchrun"** — you ran bare `pytest`.
The `dist_init` fixture in `conftest.py` needs `LOCAL_RANK`, which `torchrun`
sets.

**"Requires world_size >= 2"** — you used `--nproc_per_node=1`. That is only for
`generate_reference.py`.

**`FileNotFoundError: ..._reference_*.pt`** — the fixtures were never generated;
see step 3.
