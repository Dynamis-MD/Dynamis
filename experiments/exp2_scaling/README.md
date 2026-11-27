# Experiment 2: Multi-GPU Strong, Weak & Hero Scaling Harness

Shared, **model-agnostic** scaling benchmark infrastructure for **Dynamis**
(supports ORB, UMA, MACE, and any other `dynamis_model` NNP backend) using the
MD5-verified Zenodo benchmark datasets ([record 24487955](https://zenodo.org/records/24487955)).

> **Experiment Plan & Status:** See [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)
> for the cross-model scaling methodology, per-dataset run matrices (`ch4_o2`,
> `li3po4`, `2w49`, `apoa1`), per-model guidance (ORB, UMA, MACE), and the live
> completed vs. to-do tracker.

## 1. Obtain Shared Datasets (Once, Login Node)

Datasets live centrally under `experiments/exp_datasets/zenodo/` so every model
shares a single verified copy:

```bash
# Download both scaling suites (ch4_o2: 8k..10M atoms, li3po4: 32k..1.01M atoms)
bash experiments/exp_datasets/scripts/download_zenodo.sh --scaling

# Download biomolecular benchmarks or list all 8 archives on Zenodo record 24487955
bash experiments/exp_datasets/scripts/download_zenodo.sh 2w49 apoa1
bash experiments/exp_datasets/scripts/download_zenodo.sh --list
```

On shared clusters, set `DYNAMIS_ZENODO_CACHE_DIR=/path/to/shared_cache` to avoid
redundant downloads across teammates.

## 2. Cluster Configuration & Slurm Submission Rules

`cluster.env` and `scripts/sbatch_submit.sh` encode the default Empire AI `beta`
(GB200 NVL72, 4× B200/node) rules and can be customized per user via an ignored
`cluster.local.env`:

- Always passes `--account` (`su_kchiu_molecular` by default).
- Always allocates all **4 GPUs/node** (`--gpus-per-node=4 --mem=0 --cpus-per-task=32`).
- Automatically sets `--segment=<nodes>` so multi-node jobs (`N <= 18`) land in a
  single NVL72 rack block for MNNVL `GPUDirectExchanger` Fabric handles.
- Uses `--partition=beta --qos=priority` by default (`--test` targets `beta_test` / `test`).
- Set `EXP2_DRY_RUN=1` to inspect exact `sbatch` commands without submitting.

## 3. Shared Harness Components

| File | Purpose |
|------|---------|
| `EXPERIMENT_PLAN.md` | Cross-model experiment plan, per-dataset matrices, model guidance, and done/to-do status tracker |
| `experiments/exp_datasets/scripts/download_zenodo.sh` | MD5-verified Zenodo record 24487955 downloader (`ch4_o2`, `li3po4`, `2w49`, `apoa1`, `CH4_*`, `RP3_1320`) |
| `cluster.env` + `scripts/_lib.sh` | Cluster/CUDA/NCCL/MNNVL bootstrap and dataset discovery (`EXP2_DATASETS_ROOT`) |
| `scripts/sbatch_submit.sh` | Site-policy Slurm submission wrapper |
| `drivers/bench_common.py` | Reusable CLI base parser, `TrajectoryReader` cell/PBC loader, `ScalableHaloPartitioner`, `kinetic_temperature`, `TimedLoop`, OOM/timeout watchdogs, `gather_rank_metrics`, and `Dynamis-experiments-log` (`run.schema.json` v1.0) record + gzipped timing writer |

## 4. Output Format & Post-Analysis (`Dynamis-experiments-log`)

All post-analysis, summary tables, and publication figures live in
**`Dynamis-experiments-log`** (do not add ad-hoc plotting scripts here).

Drivers built on `bench_common.py` emit per-run directories directly matching
`Dynamis-experiments-log`'s canonical layout and `run.schema.json` v1.0 (with
portable `dataset_file` basenames, run-relative `"timings_dir": "timings"`,
gzipped `timing_rank*.jsonl.gz`, and zero machine-local absolute paths):

```text
<output_dir>/<study>/<run_id>/
├── result.json
└── timings/
    ├── timing_rank0.jsonl.gz
    └── ...
```

To record and analyze a completed campaign in `Dynamis-experiments-log`:

1. Copy `<output_dir>/{strong,weak,hero,...}/` into
   `Dynamis-experiments-log/records/<model_family>/<system>/<campaign_id>/`.
2. Add `campaign.yaml` (conforming to `schemas/campaign.schema.json`).
3. Validate, summarize, and generate figures in `Dynamis-experiments-log`:
   ```bash
   dynamis-log validate  records/<model_family>/<system>/<campaign_id> --check-timings --strict
   dynamis-log summarize records/<model_family>/<system>/<campaign_id>
   ```
   Use `figures/scripts/` with `dynamis_log.viz` + `figures/registry.yaml` for
   all plots and cross-model comparisons.

## 5. Adding a New Model Driver (UMA, MACE, ORB, …)

1. Create `drivers/run_<model>_scaling.py` importing `build_base_parser`,
   `load_system`, `inject_cell`, `wrap_scalable_halo`, `kinetic_temperature`,
   `TimedLoop`, `capture_oom`, `gather_rank_metrics`, `make_run_id`,
   `run_directory`, `compress_timings`, `BenchRecord`, and `write_result` from
   `bench_common`.
2. Pass `output_dir=os.path.join(run_dir, "timings")` to `SimulatorConfig`,
   call `compress_timings(...)` on rank 0 after `TimingRegistry.instance().close()`,
   and store model-specific fields in `BenchRecord(..., metrics={...})`.
3. Submit batches through `scripts/sbatch_submit.sh` (pointing `EXP2_DATA_DIR` at
   `experiments/exp_datasets/zenodo/<dataset>`), then ingest the resulting
   `<study>/<run_id>/` directories into `Dynamis-experiments-log`.

## 6. ORB Scaling Quick Start (CH₄:O₂ & Li₃PO₄)

From `experiments/exp2_scaling/`:

### Build the Python 3.12 ORB environment & pre-warm checkpoints (once, login node)
```bash
bash scripts/setup_orb_env.sh
```

### Run Preflight Gates
```bash
# Distributed parity + neighbor-cap tests (1 node, beta_test)
scripts/sbatch_submit.sh --test --time=00:15:00 --job-name=orb-parity \
    scripts/probe_orb_gate.sbatch parity

# 1-GPU HBM ceiling probe (CH4:O2)
scripts/sbatch_submit.sh --time=00:20:00 --job-name=orb-ceiling \
    scripts/probe_orb_gate.sbatch ceiling-direct

# 2-node MNNVL GPUDirect Fabric vs NCCL verification
scripts/sbatch_submit.sh --nodes=2 --segment=2 --time=00:15:00 --job-name=orb-mnnvl \
    scripts/probe_orb_gate.sbatch mnnvl

# Li3PO4 chunked BF16 + Blackwell fused kernels gate (requires PRs #156 & #158)
scripts/sbatch_submit.sh --test --time=00:20:00 --job-name=li3po4-gate \
    scripts/probe_orb_gate.sbatch li3po4-fused
```

### Submit Scaling & Hero Sweeps (1 → 32 GPUs)
```bash
# Inspect exact sbatch commands without submitting
scripts/submit_orb_scaling.sh --dry-run --dataset ch4_o2 --study all --model both
scripts/submit_orb_scaling.sh --dry-run --dataset li3po4 --study all

# CH4:O2 Orb-Direct strong, weak, and 10M-atom hero runs
scripts/submit_orb_scaling.sh --dataset ch4_o2 --study strong --model both --exchanger gpudirect
scripts/submit_orb_scaling.sh --dataset ch4_o2 --study weak   --model both --weak-density both
scripts/submit_orb_scaling.sh --dataset ch4_o2 --study hero   --model both

# Li3PO4 Orb-Direct k=20 (chunked BF16 + fused kernels) weak32k + strong1m + strong61k
scripts/submit_orb_scaling.sh --dataset li3po4 --study all
```
