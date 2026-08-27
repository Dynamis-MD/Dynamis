# exp0 — Synthetic End-to-End MD Benchmark:

UMA-based MD on synthetic CH₄:O₂ boxes (~16k → ~1M atoms) on a single GPU node
(1/2/4 GPUs), comparing three engines:

| Engine | What it is |
|---|---|
| **DistNNP-UMA** | `distnnp_model.nnp.UMAModel` + `distnnp.Simulator`, one rank/GPU via `torchrun`; atoms spatially partitioned (SFC-Morton, edge-weighted), only ghost atoms/features exchanged per message-passing layer. |
| **FairChem-UMA** (baseline) | one ASE Langevin driver whose force/energy call is graph-parallel across the node via `fairchem` `get_predict_unit(..., workers=N)` → Ray `ParallelMLIPPredictUnit` (contiguous **index** partition of atoms, node embeddings all-gathered per layer, energy all-reduced). |
| **LAMMPS-MACE** (baseline) | `pair_style mliap unified` (MACE `medium-mpa-0` via cuEquivariance, Kokkos/CUDA); LAMMPS owns the spatial domain decomposition + ghost communication + integrator. Holds the *model* fixed against a second family so the comparison is decomposition strategy only. See [LAMMPS-MACE baseline](#lammps-mace-baseline-workstream-b). |

Both engines: model **`uma-s-1p2`**, task **`omol`**, **Langevin** thermostat,
`dt = 0.5 fs`, `T = 300 K`, `friction = 0.01 fs⁻¹`, both start from a fresh
Maxwell-Boltzmann velocity draw.

The result (see [Results](#results)) is that DistNNP is a few percent faster on
every multi-GPU cell and scales slightly cleaner, but FairChem's balanced
edge-sharding gives it a **higher single-node memory ceiling** on uniform
systems. 

---

## Layout

```
drivers/
  bench_common.py       args, warmup+timed loop, OOM capture, JSON schema, unit constants
  run_distnnp_uma.py    DistNNP-UMA driver (torchrun, 1 rank/GPU)
  run_fairchem_uma.py   FairChem-UMA driver (1 proc + Ray workers)
  run_lammps.py         LAMMPS-MACE driver (spawns `lmp` / `mpirun`, parses Loop time)
  templates/lammps_nvt.in   templated Langevin-NVT input deck
scripts/
  cluster.env           SITE CONFIG — Slurm accounts/partitions, modules, fabric
                        (Delta defaults; override via cluster.local.env or env vars)
  cluster.env.example   template for a non-Delta cluster
  _lib.sh               shared bootstrap: path resolution, module/runtime helpers
  sbatch_exp0.sh        cluster-agnostic `sbatch` wrapper (injects account/partition/mem/output)
  env_setup.sh          build the .venv-uma venv + the 4 CUDA extensions
  build_lammps_env.sh   build the SEPARATE .venv-lammps-mace (torch cu126 + mace + cuEq)
  build_lammps.sh       build LAMMPS from source (ML-IAP + KOKKOS/CUDA + PYTHON)
  spike_lammps_mace.sbatch  single- vs multi-GPU gate
  gen_datasets.sh       wrapper over exp_datasets/scripts/03_generate_ch4.py
  smoke.sbatch          ≤3 min: 1 tiny cell per UMA engine on 2 GPUs
  smoke_lammps.sbatch   ≤5 min: 1 tiny LAMMPS-MACE cell
  run_cell.sbatch       run ONE (engine,xyz,ngpu) cell; failure → oom/error stub JSON
  submit_matrix.sh      enumerate the matrix, submit one job per cell (throttled)
analysis/
  summarize.py          merge results/*.json → summary.csv + figures + headline_table.md
results/                one JSON per cell (git-ignored)
```

---

## How to run

All commands are given from the repo root. Every Slurm job goes through
`scripts/sbatch_exp0.sh`, which reads `scripts/cluster.env` and injects the
site-specific `--account` / `--partition` / `--mem` / `--output` flags — the
`.sbatch` files themselves contain **no** cluster-specific `#SBATCH` lines.

```bash
cd experiments/exp0_end_to_end
```

### 1. Build the environment (once, CPU node, ~15–25 min)

```bash
scripts/sbatch_exp0.sh --cpu scripts/env_setup.sh
```

Prerequisite: accept the `facebook/UMA` license on huggingface.co and
`huggingface-cli login` (or `export HF_TOKEN=...`) once. The script fresh-builds
a uv venv at `DistNNP/.venv-uma` (torch 2.8.0+cu128, fairchem-core 2.21, ray
2.58), builds the 4 distnnp CUDA extensions with `--no-build-isolation` against
*this* venv's torch (arch from `EXP0_CUDA_ARCH`, default `80;90`), and pre-warms
the `uma-s-1p2` checkpoint so compute nodes need no network. Resolved versions
land in `resolved_versions.txt`. Requires `uv` on PATH (or `EXP0_UV=/path/to/uv`).

### 2. Generate datasets (once, seconds, login node — no Slurm)

```bash
scripts/gen_datasets.sh                 # default: k ∈ {1,2,32,64,128,192,256}
scripts/gen_datasets.sh 32 64 128 192   # a subset
```

Writes `exp_datasets/baselines/ch4_<k>k_splited.xyz`. Positions are
bit-reproducible; pinned sizes are checksum-verified against
`exp_datasets/scripts/checksums.txt`, newer sizes are generated and reported.

### 3. Smoke-test the drivers (2 GPUs, ~3 min, optional)

```bash
scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke.sbatch
```

One tiny cell per UMA engine — just confirms each driver launches and writes a
schema-valid `status:"ok"` JSON. This is **not** a correctness check: model
parity vs a single-process baseline lives in `distnnp_model/tests/` and is not
exp0's job to re-run.

### 4. Run the matrix

```bash
scripts/submit_matrix.sh --dry-run                    # inspect the job list
scripts/submit_matrix.sh                              # submit all (~16 jobs)
scripts/submit_matrix.sh --only fill                  # just one sub-experiment
scripts/submit_matrix.sh --only strong --engine distnnp-uma
```

| Sub-experiment (`--only`) | cells | timed steps |
|---|---|---|
| `strong` | both × `32k @ {1,2,4}` | 20 |
| `fill` | both × `{32k@1, 64k@2, 128k@4}` (≈32k atoms/GPU, compute-bound) | 20 |
| `ceiling` | both × `{192k, 256k} @ 4` (push to OOM) | 12 |

`submit_matrix.sh` never requests more GPUs/node than `EXP0_GPUS_PER_NODE_MAX`.
The timed window stays below the first repartition
(`warmup + timed < rebalance_interval`) so every timed step does identical work;
warmup covers CUDA init + first-call graph/allocation. FairChem OOM cells hang
(Ray) rather than raising, so each carries a hard `timeout` and `run_cell.sbatch`
writes an `oom` stub if no JSON appears.

### 5. Analyse

```bash
DistNNP/.venv-uma/bin/python experiments/exp0_end_to_end/analysis/summarize.py
```

→ `analysis/summary.csv`, `analysis/headline_table.md`, and
`analysis/figures/{strong_scaling,fill_scaling,oom_ceiling}.png`.

---

## LAMMPS-MACE baseline

`implementation_plan.md` §12. Adds a third engine, `lammps-mace`, that runs the
MACE `medium-mpa-0` foundation model through LAMMPS's `pair_style mliap unified`
(cuEquivariance + Kokkos/CUDA). Same model as the DistNNP-side MACE adapter
(`origin/MACE-Adaptation`), so a DistNNP-MACE vs LAMMPS-MACE comparison is
apples-to-apples once that wrapper lands.

### Why a separate build + env

The stock `lammps/*` modules on Delta have `ML-IAP`, `KOKKOS` and `PYTHON` all
**off**, so LAMMPS must be built from source. MACE ML-IAP also needs its own
python stack (torch `cu126` + `mace-torch` + `cuequivariance` + `numpy<2`),
which conflicts with `.venv-uma` (torch 2.8 / numpy 2.4).

A first full run was done at **float64** (archived, `results/_archive_float64/`)


### How to run

```bash
cd experiments/exp0_end_to_end
bash scripts/gen_datasets.sh                                       # datasets (gitignored)
scripts/build_lammps_env.sh                                        # ~15 min, login node OK
scripts/sbatch_exp0.sh --cpu --time=03:00:00 scripts/build_lammps.sh   # ~1–2 h
# put the resulting  EXP0_LAMMPS_BIN=<...>/install/bin/lmp  in cluster.local.env

scripts/sbatch_exp0.sh --gpus-per-node=4 scripts/spike_lammps_mace.sbatch   # → results/spike/VERDICT.txt
scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke_lammps.sbatch        # schema check

# the LAMMPS-MACE matrix (strong / fill / ceiling / 1-GPU sweep / naive):
scripts/submit_matrix.sh --with-lammps --engine lammps-mace
```

Already built the LAMMPS binary + venv and just need to re-run at a different
precision? Skip straight to `RESTART.md`.

`analysis/summarize.py` picks up `lammps-mace` automatically (N-engine headline
table + a third series on every figure). Cross-engine steps/s ratios are
**suppressed when the two engines ran different models** (UMA vs MACE) — the
table legend names the model per engine.

### Results

---

## Running on another cluster

Nothing above is Delta-specific except the values in `scripts/cluster.env`.
To run elsewhere, **do not edit tracked files** — instead:

```bash
cp scripts/cluster.env.example scripts/cluster.local.env
$EDITOR scripts/cluster.local.env      # git-ignored; sourced first, wins
```

or export the `EXP0_*` variables in your shell. Only the values that differ from
the Delta defaults need to be set:

| variable | default (Delta) | meaning |
|---|---|---|
| `EXP0_GPU_ACCOUNT` / `EXP0_CPU_ACCOUNT` | `bcpt-delta-gpu` / `bcpt-delta-cpu` | Slurm `--account` |
| `EXP0_GPU_PARTITION` / `EXP0_CPU_PARTITION` | `gpuA100x4` / `cpu` | Slurm `--partition` |
| `EXP0_GPUS_PER_NODE_MAX` | `4` | cap on `--gpus-per-node` per cell |
| `EXP0_GPU_MEM` / `EXP0_CPU_MEM` | `200g` / `48g` | Slurm `--mem` |
| `EXP0_CPUS_PER_TASK` | `16` | Slurm `--cpus-per-task` |
| `EXP0_SBATCH_EXTRA` | *(empty)* | extra flags for every job, e.g. `--qos=high` |
| `EXP0_MODULES` | `python/3.13.5-gcc13.3.1 cudatoolkit/26.5_12.9-unsupported gcc-native/14 aws-ofi-nccl` | args to one `module load`; **set empty** on conda/container clusters |
| `EXP0_MODULE_UNLOAD` | `cudatoolkit` | `module unload` these first |
| `EXP0_MODULE_PURGE` | `0` | `module purge` before loading |
| `EXP0_NCCL_SOCKET_IFNAME` | `hsn` | high-speed fabric iface; empty ⇒ NCCL auto-detects |
| `EXP0_CUDA_ARCH` | `80;90` | extension build target (A100=80, H100/H200=90, A40=86) |
| `EXP0_VENV` | `<repo>/DistNNP/.venv-uma` | reuse an existing venv |
| `EXP0_UV` | `uv` | uv binary for `env_setup.sh` |

**LAMMPS-MACE engine only** (skip unless running `lammps-mace`):

| variable | default (Delta) | meaning |
|---|---|---|
| `EXP0_LAMMPS_ROOT` | `/projects/bcpt/$USER` | parent dir for the env + LAMMPS build tree (not `$HOME`) |
| `EXP0_LAMMPS_VENV` / `EXP0_LAMMPS_BIN` | `<ROOT>/venvs/.venv-lammps-mace` / `<ROOT>/lammps/install/bin/lmp` | derived from ROOT |
| `EXP0_LAMMPS_MODULES` | `cuda/12.9 gcc-native/14 cmake/3.31.8 nvhpc-hpcx-cuda12-openmpi5/26.5 fftw` | module set for the LAMMPS build + run |
| `EXP0_LAMMPS_TORCH` / `EXP0_LAMMPS_TORCH_INDEX` | `2.7.1` / `.../whl/cu126` | torch wheel for the LAMMPS venv (drop to `2.6.0` if cuEq-ops has no wheel) |
| `EXP0_LAMMPS_REF` | `release` | LAMMPS git ref (pin a Sept-2025+ commit once validated) |
| `EXP0_MACE_MODEL_NAME` / `EXP0_MACE_MODEL` | `medium-mpa-0` / `<ROOT>/mace_models/mace-mpa-0-medium.model` | MACE foundation model |
| `EXP0_MACE_ELEMENTS` | `C H O` | element→LAMMPS-type order for `pair_coeff` |

`REPO_ROOT` and the inner monorepo path are derived from each script's own
location — no need to set them unless your checkout layout differs
(`EXP0_REPO_ROOT`).

`scripts/cluster.env.example` is a worked example for a hypothetical 8×H100
cluster with different partition/module names. Sanity-check your config with:

```bash
EXP0_DRY_RUN=1 scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke.sbatch
scripts/submit_matrix.sh --dry-run
```

---

## Resolved environment

From `resolved_versions.txt`; python 3.13.5, CUDA 12.9
(`cudatoolkit/26.5_12.9-unsupported`), extensions compiled for arch `80;90`.

| package | version | | package | version |
|---|---|---|---|---|
| torch | 2.8.0+cu128 | | ray | 2.58.0 |
| torchvision | 0.23.0+cu128 | | e3nn | 0.6.0 |
| fairchem-core | 2.21.0 | | numpy | 2.4.6 |
| ase | 3.29.0 | | triton | 3.4.0 |
| numba | 0.67.0 | | scipy | 1.18.1 |

---

## Results

**Cluster:** NCSA Delta `gpuA100x4` (4× A100-SXM4-**40 GB**, NVLink).
**System:** isotropic CH₄:O₂ dense fluid, ρ ≈ 0.037 atoms/Å³.
Regenerate with `analysis/summarize.py` (→ `analysis/headline_table.md`).

### Throughput (steps/s) — corrected physics, both engines at ~250 K

| atoms | GPUs | atoms/GPU | DistNNP | FairChem | speedup |
|------:|:---:|:---:|---:|---:|:---:|
| 16k | 1 | 16k | 0.285 | 0.286 | 1.00× |
| 32k | 1 | 32k | 0.138 | 0.139 | 1.00× |
| 32k | 2 | 16k | 0.273 | 0.254 | **1.07×** |
| 32k | 4 | 8k | 0.536 | 0.498 | **1.08×** |
| 64k | 2 | 32k | 0.135 | 0.127 | **1.06×** |
| 128k | 4 | 32k | 0.138 | 0.132 | **1.04×** |

1 GPU is a tie by construction (`--workers 1` FairChem = plain `MLIPPredictUnit`,
no Ray). On every cell where FairChem actually runs graph-parallel, DistNNP is
4–8 % faster.

### Parallel efficiency

| | DistNNP | FairChem |
|---|---:|---:|
| strong, 4 GPU (32k) | **97 %** | 90 % |
| weak, 4 GPU (32k/rank) | **99 %** | 96 % |
