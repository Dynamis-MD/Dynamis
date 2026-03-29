# How to Run DistNNP on HPC systems NCSA Delta 

This document assumes you have just logged into Delta and cloned the
repository. It covers every step needed to install dependencies, build
the C++ / CUDA extensions, and submit the synthetic LJ experiment.

---

## 0. System overview

| Item | Value |
|------|-------|
| Cluster | NCSA Delta |
| GPU nodes | A100 40 GB (`gpuA100x4` partition) |
| SLURM account | `bcpt-delta-gpu` |
| OS | RHEL 9 |
| Python (system) | 3.9.18 at `/usr/bin/python3` |
| CUDA | 12.8 (via `cudatoolkit/25.3_12.8` module) |

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

Only three modules are actually needed. Load them now and add them to
your `~/.bashrc` or a setup script so they are available in batch jobs
too.

```bash
module load cudatoolkit/25.3_12.8   # CUDA 12.8 runtime + nvcc
module load gcc-native/13.2         # C++ compiler for building extensions
module load slurm-env/0.1           # sbatch / squeue / etc.
```

---

## 3. Install Python dependencies and build the C++ extensions

The `requirements-dev.txt` file installs PyTorch, pytest, and all four
sub-packages in editable mode.  The C++ sub-packages
(`distnnp_state`, `distnnp_integrator`) are built automatically by
`pip` using `scikit-build-core` and CMake.

First install `scikit-build-core`, which is the build backend used by
the C++ packages:

```bash
pip install scikit-build-core --user
```

Then install everything else.  The `CMAKE_PREFIX_PATH` must point at
the PyTorch CMake directory so that CMake can locate Torch headers and
libraries when building the CUDA extensions:

```bash
cd DistNNP   # the inner DistNNP/ that contains requirements-dev.txt

CMAKE_PREFIX_PATH=$(python3 -c \
  'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))') \
pip install -r requirements-dev.txt --user
```

> **Why `CMAKE_PREFIX_PATH` is needed:**  On Delta, PyTorch is
> installed in a non-standard user location.  Without this hint, CMake
> cannot find `TorchConfig.cmake` and the build fails with
> `Could not find Torch`.

This single command installs PyTorch (if not already present),
compiles the CUDA kernels for `distnnp_state` and
`distnnp_integrator`, and installs all four packages as editable
installs that point back into the source tree.

Expect the build to take **3–8 minutes** the first time.  Subsequent
installs are faster because CMake caches the build artefacts under
`distnnp_state/skbuild/` and `distnnp_integrator/skbuild/`.

### Verifying the install

```bash
python3 -c "
import distnnp_state
import distnnp_integrator
import distnnp_model
import distnnp
print('All packages imported successfully')
"
```

If any import fails, re-run the install step — a partial build
sometimes leaves the `.so` in the wrong location on the first attempt.

---

## 4. Create the SLURM batch script

Create a file called `run_test.sh` in the top-level `DistNNP/`
directory (the one that contains the two `DistNNP/` and `run_test.sh`
after cloning):

```bash
cat > run_test.sh << 'EOF'
#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --job-name=distnnp-test
#SBATCH --output=/u/$USER/DistNNP/distnnp_test_%j.out

module load cudatoolkit/25.3_12.8
module load gcc-native/13.2

cd /u/$USER/DistNNP/DistNNP/distnnp/experiments/synthetic
torchrun --standalone --nproc_per_node=2 main.py
EOF
```

> **Note:** Replace `/u/$USER/DistNNP/` with your actual path if you
> cloned to a different location.

---

## 5. Submit the job

```bash
cd /u/$USER/DistNNP   # top-level directory containing run_test.sh
sbatch run_test.sh
```

Check job status:

```bash
squeue -u $USER
```

Watch the output as the job runs (replace `<JOBID>` with the number
printed by `sbatch`):

```bash
tail -f distnnp_test_<JOBID>.out
```

---

## 6. What a successful run looks like

The job runs for roughly **8–10 seconds** for 1 000 Verlet steps with
100 atoms on 2 GPUs.

The output file will contain only a few PyTorch distributed info lines
and one warning about `destroy_process_group` — no tracebacks:

```
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[rank0]:[W...] WARNING: destroy_process_group() was not called before program exit...
```

Checkpoint files are written every 10 steps into the experiment
directory:

```
DistNNP/distnnp/experiments/synthetic/
    rank_0_step_0.pt
    rank_0_step_10.pt
    ...
    rank_0_step_990.pt
    rank_1_step_0.pt
    ...
    rank_1_step_990.pt
```

Each `.pt` file is a dict with keys `pos`, `vel`, `atom` holding the
local partition's positions, velocities, and global atom indices for
that rank at that step.

---

## 7. Re-running after a code change

If you change any **Python-only** file (anything under
`distnnp/`, `distnnp_model/`, or the experiment scripts), no rebuild
is needed — editable installs pick up changes immediately.

If you change any **C++ or CUDA** file (anything under
`distnnp_state/src/csrc/` or `distnnp_integrator/src/csrc/`), reinstall
that sub-package:

```bash
CMAKE_PREFIX_PATH=$(python3 -c \
  'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))') \
pip install -e ./distnnp_integrator --user
# or
pip install -e ./distnnp_state --user
```

---

## 8. Troubleshooting

### `Could not find Torch` during build
You forgot to set `CMAKE_PREFIX_PATH`. Use the full command from §3.

### `No module named 'distnnp_state'` or similar after install
The `.so` was not placed correctly. Run the install command again — on
the second attempt scikit-build-core usually places it correctly.

### `CUDA error: device-side assert triggered` immediately
Add `CUDA_LAUNCH_BLOCKING=1` before `torchrun` in the SLURM script to
get a more precise traceback. This slows the run but makes CUDA errors
synchronous.

### Job stays in `PD` (pending) state
The `gpuA100x4` partition is busy. Check with:
```bash
squeue -u $USER
```
A `(Resources)` reason means nodes are allocated but not yet available;
it will start automatically. A `(QOSMaxJobsPerUserLimit)` reason means
you have too many jobs in the queue already.

### `Connection closed by peer` in the output
One rank crashed before the other. Scroll up in the output file to
find the rank that actually printed the traceback — the
`Connection closed` message is from the *surviving* rank noticing that
the other died.
</content>
</invoke>