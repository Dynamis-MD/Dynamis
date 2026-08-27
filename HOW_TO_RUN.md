# How to Run DistNNP on HPC systems NCSA Delta

This document assumes you have just logged into Delta and cloned the
repository. It covers every step needed to install dependencies, build
the C++ / CUDA extensions, and submit the synthetic LJ experiment.

Last verified end-to-end (build + tests + multi-GPU run) on 2026-08-26.

---

## 0. System overview

| Item | Value |
|------|-------|
| Cluster | NCSA Delta |
| GPU nodes | A100 40 GB (`gpuA100x4` partition) |
| SLURM account | `bcpt-delta-gpu` |
| OS | RHEL 9 |
| Python (build) | 3.13.5 via the `python/3.13.5-gcc13.3.1` module |
| CUDA (build + run) | 12.9 via `cudatoolkit/26.5_12.9-unsupported` — **not** the default `cudatoolkit/26.5_13.2` module, see note below |
| Package manager | [`uv`](https://docs.astral.sh/uv/) |

> **Why CUDA 12.9, not the default 13.2 module:** `distnnp_state` and
> `distnnp_model` hardcode `CUDA_ARCHITECTURES "70;80;90"` in their
> `CMakeLists.txt` (Volta/Ampere/Hopper). CUDA 13.x dropped support for
> `compute_70` (Volta) entirely, so building against the default
> `cudatoolkit/26.5_13.2` module fails with
> `nvcc fatal: Unsupported gpu architecture 'compute_70'`. The
> `cudatoolkit/26.5_12.9-unsupported` module (CUDA 12.9) still supports it and
> is what actually builds successfully on this cluster. Use the same module
> at run time too — the compiled `.so` extensions link against this toolkit's
> CUDA runtime libraries.

---

## 1. Clone the repository

```bash
git clone https://github.com/szaman19/DistNNP.git
cd DistNNP
```

Everything below is run from this top-level `DistNNP/` directory unless
stated otherwise.

---

## 2. Load required modules

```bash
module load python/3.13.5-gcc13.3.1
module unload cudatoolkit 2>/dev/null || true   # drop the default 13.2 module
module load cudatoolkit/26.5_12.9-unsupported
module load gcc-native/14
module load aws-ofi-nccl                        # bridges NCCL to Slingshot 11 for multi-node jobs
```

Add these to a small `env.sh` you `source` at the start of every session (and
in every sbatch script) rather than to `~/.bashrc` — cudatoolkit versions here
have moved out from under old docs before (see the CUDA note above), so it's
easier to track/update in one file than to hunt through shell init scripts.

---

## 3. Create a `uv` venv and install

```bash
cd DistNNP   # the inner DistNNP/ that contains requirements-dev.txt
uv venv --python "$(which python3)" .venv
source .venv/bin/activate
```

**Install torch first, standalone.** `distnnp_state`/`distnnp_integrator`
don't declare torch as a dependency (they only need its CMake config at build
time), and `requirements-dev.txt` computes `CMAKE_PREFIX_PATH` by importing
`torch` from the *current* interpreter — so torch must already be installed
and importable before you run the main install command:

```bash
uv pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install ninja   # not on PATH by default; system cmake (3.26.5) already satisfies the >=3.25 requirement
```

**Pin torch with a constraint file before the main install.** Without this,
`uv pip install -r requirements-dev.txt` re-resolves the unpinned
`torch>=2.0.0` line against the default PyPI index, tries to replace the
already-installed `+cu128` build, and can corrupt the venv if interrupted
mid-download:

```bash
echo "torch==2.11.0+cu128" > /tmp/distnnp-constraints.txt
```

Then build and install everything, with `CMAKE_PREFIX_PATH` pointing at
torch's CMake config and `CMAKE_ARGS` pinning the CUDA architectures (this
also fixes `distnnp_integrator/CMakeLists.txt`, which — unlike its two
siblings — doesn't hardcode `CUDA_ARCHITECTURES` at all, so on a login node
with no GPU visible, CMake's auto-detection can silently pick the wrong
target):

```bash
export CMAKE_PREFIX_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))'):$CMAKE_PREFIX_PATH"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=70;80;90"

uv pip install -r requirements-dev.txt --constraint /tmp/distnnp-constraints.txt
```

This installs pytest/scipy and all four sub-packages in editable mode
(`distnnp_state` → `distnnp_model[all]` → `distnnp_integrator` → `distnnp`,
per `requirements-dev.txt`'s own order — `distnnp_state`'s headers must be
installed before `distnnp_model` can build). Expect the `distnnp_model[all]`
step (pulls `fairchem-core` and its dependency tree, including a from-source
`numpy` build since Python 3.13 has no prebuilt wheel for the resolved
`numpy==1.26.4`) to take **10–20 minutes** the first time — this is the slow
step, not a hang. Run it under `sbatch` or with `run_in_background` if you're
scripting this rather than typing it interactively; it comfortably exceeds a
10-minute foreground shell.

### Verifying the install

```bash
python -c "
import distnnp_state, distnnp_integrator, distnnp_model, distnnp
print('All packages imported successfully')
"
```

If any import fails, re-run the install step — a partial build sometimes
leaves the `.so` in the wrong location on the first attempt.

---

## 4. Confirm the test suite passes

Before trusting the environment for benchmarking, confirm the existing suite
is green in it:

```bash
salloc --account=bcpt-delta-gpu --partition=gpuA100x4 --nodes=1 \
       --gpus-per-node=2 --time=00:30:00 --mem=64g
# (inside the allocation)
module load python/3.13.5-gcc13.3.1
module unload cudatoolkit 2>/dev/null || true
module load cudatoolkit/26.5_12.9-unsupported gcc-native/14
cd /u/$USER/DistNNP/DistNNP
source .venv/bin/activate

torchrun --nproc_per_node=1 distnnp/tests/fixtures/generate_reference.py   # one-time reference fixtures
torchrun --nproc_per_node=2 -m pytest distnnp/tests/ -v -m "not perf"
```

Expect `49 passed, 17 skipped, 6 deselected` (skips are the Langevin
regression cases; deselected are the `perf` tests).

---

## 5. Run the synthetic experiment

`distnnp/experiments/synthetic/main.py` now takes CLI args and reports
throughput instead of running one hardcoded, unconfigurable pass:

```bash
cd distnnp/experiments/synthetic
torchrun --standalone --nproc_per_node=2 main.py --num-atoms 1000 --steps 200
```

Flags: `--num-atoms`, `--box-size`, `--steps`, `--timestep-fs`,
`--rebalance-interval`, `--log-interval`, `--traj-interval` (0 = no
trajectory snapshots, the default — enable it if you actually want
`trajectory.traj` output), `--output-dir` (defaults to `./results` next to
the script).

---

## 6. Multi-GPU baseline sweep (SLURM)

Use `scripts/run_synthetic_baseline.sbatch` — it loads the correct modules,
sets the NCCL/libfabric env vars, and calls the venv's `torchrun` **by
absolute path** rather than `source .venv/bin/activate` (activation was
observed to silently no-op under `sbatch` on this cluster — jobs fell back to
system Python 3.9 — most likely a conda shell hook in the login profile
interfering with plain venv activation in a non-interactive shell):

```bash
for n in 1 2 4; do
  sbatch --gpus-per-node=$n scripts/run_synthetic_baseline.sbatch $n --num-atoms 4000 --steps 1000
done
```

Each job writes `distnnp/experiments/synthetic/results/synthetic_baseline_<N>gpu_<atoms>atoms.json`
with `steps_per_sec` and `peak_memory_mb`. Reference numbers from the last
run (4000 atoms, 1000 steps): 1 GPU → 47.3 steps/s, 2 GPU → 76.7 steps/s,
4 GPU → 88.9 steps/s. Sub-linear scaling at this atom count is expected —
halo-exchange overhead dominates at small system sizes (see `CLAUDE.md`).

Check job status with `squeue -u $USER`; logs land in
`DistNNP/DistNNP/logs/synthetic_baseline_<jobid>.{out,err}`.

---

## 7. Re-running after a code change

If you change any **Python-only** file (anything under `distnnp/`,
`distnnp_model/`, or the experiment scripts), no rebuild is needed —
editable installs pick up changes immediately.

If you change any **C++ or CUDA** file (anything under
`distnnp_state/src/csrc/` or `distnnp_integrator/src/csrc/`), reinstall
that sub-package with the same env vars as the initial build:

```bash
export CMAKE_PREFIX_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))'):$CMAKE_PREFIX_PATH"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=70;80;90"
uv pip install -e ./distnnp_integrator
# or
uv pip install -e ./distnnp_state
```

---

## 8. Troubleshooting

### `nvcc fatal: Unsupported gpu architecture 'compute_70'`
You're building against the default `cudatoolkit/26.5_13.2` module (CUDA
13.x dropped Volta support). `module unload cudatoolkit && module load
cudatoolkit/26.5_12.9-unsupported` first — see §0.

### `Could not find Torch` during build
Torch isn't installed/importable yet in the current interpreter, or
`CMAKE_PREFIX_PATH` wasn't exported before the install command. Install torch
standalone first (§3), then export `CMAKE_PREFIX_PATH` in the same shell.

### `uv pip install -r requirements-dev.txt` re-downloads a ~500MB torch and/or corrupts the venv
You skipped the `--constraint` file in §3 — the unpinned `torch>=2.0.0` line
lets the resolver try to replace your already-installed `+cu128` build. If
the venv's torch is already corrupted (a `torch-*.dist-info/` directory with
no `METADATA` file inside), delete `.venv/lib/python3.13/site-packages/torch*`
and reinstall it per §3.

### `AttributeError: 'MolecularInfo' object has no attribute 'atom_forces'`
Fixed as of 2026-08-26 in `setup_system.py` — `atom_forces` was never a bound
attribute on the compiled `MolecularInfo` binding, so this would fire on the
very first call to `generate_synthetic_dataset`. If you see this again,
someone re-added that assignment; just remove it.

### One rank exits with no traceback while writing results (`ChildFailedError`, no Python traceback)
The C++ `TrajectoryWriter` constructor does `stat()`-then-`mkdir()` with a
hard `exit(1)` on `EEXIST` — a race when multiple ranks construct it
concurrently against a not-yet-existing `output_dir`. Fixed as of
2026-08-26 in `main.py`, which now pre-creates `output_dir` before
constructing the `Simulator`. If you write a new script that constructs a
`Simulator` directly, do the same:
`os.makedirs(config.output_dir, exist_ok=True)` before construction.

### `No module named 'distnnp_state'` or similar after install
The `.so` was not placed correctly. Run the install command again — on
the second attempt scikit-build-core usually places it correctly.

### `CUDA error: device-side assert triggered` immediately
Add `CUDA_LAUNCH_BLOCKING=1` before `torchrun` to get a more precise
traceback. This slows the run but makes CUDA errors synchronous.

### Job stays in `PD` (pending) state
The `gpuA100x4` partition is busy, especially for `--gpus-per-node=4`
(whole-node) requests. Check with:
```bash
squeue -u $USER
```
A `(Priority)` or `(Resources)` reason means it will start automatically once
resources free up; `(QOSMaxJobsPerUserLimit)` means you have too many jobs
queued already.

### `Connection closed by peer` in the output
One rank crashed before the other. Scroll up in the output file to
find the rank that actually printed the traceback — the
`Connection closed` message is from the *surviving* rank noticing that
the other died.

### `undefined reference to 'dlvsym@GLIBC_2.2.5'` during CMake build
CMake looks for libraries in `$LD_LIBRARY_PATH`. Make sure it includes the
CUDA and standard system library paths:
```bash
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
```
