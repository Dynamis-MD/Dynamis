# Experiment 2: Multi-GPU Strong, Weak & Hero Scaling Harness

Shared, **model-agnostic** scaling benchmark infrastructure for **Dynamis**
(supports ORB, UMA, MACE, and any other `dynamis_model` NNP backend) using the
MD5-verified Zenodo benchmark datasets ([record 24487955](https://zenodo.org/records/24487955)).

## 1. Obtain Shared Datasets (Once, Login Node)

Datasets live centrally under `experiments/exp_datasets/zenodo/` so every model
shares a single verified copy:

```bash
# Download both scaling suites (ch4_o2: 8k..10M atoms, li3po4: 32k..1.01M atoms)
bash experiments/exp_datasets/scripts/download_zenodo.sh --scaling

# Or pick individual archives / list all 8 archives on Zenodo record 24487955
bash experiments/exp_datasets/scripts/download_zenodo.sh --list
bash experiments/exp_datasets/scripts/download_zenodo.sh ch4_o2
bash experiments/exp_datasets/scripts/download_zenodo.sh li3po4
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
| `experiments/exp_datasets/scripts/download_zenodo.sh` | MD5-verified Zenodo record 24487955 downloader (`ch4_o2`, `li3po4`, `2w49`, `apoa1`, `CH4_*`, `RP3_1320`) |
| `cluster.env` + `scripts/_lib.sh` | Cluster/CUDA/NCCL/MNNVL bootstrap and dataset discovery (`EXP2_DATASETS_ROOT`) |
| `scripts/sbatch_submit.sh` | Site-policy Slurm submission wrapper |
| `drivers/bench_common.py` | Reusable CLI base parser, `TrajectoryReader` cell/PBC loader, `ScalableHaloPartitioner`, `thermalize`, `kinetic_temperature`, `TimedLoop`, OOM/timeout watchdogs, `gather_rank_metrics`, and `BenchRecord` JSON writer |
| `analysis/summarize_scaling.py` | Aggregates `BenchRecord` JSONs + `TimingRegistry` JSONLs into `summary.csv`, `scaling_report.md`, `scaling_table.tex`, and strong/weak/memory/breakdown figures |

## 4. Adding a New Model Driver (UMA, MACE, ORB, …)

1. Create `drivers/run_<model>_scaling.py` that imports `build_base_parser`,
   `load_system`, `inject_cell`, `wrap_scalable_halo`, `thermalize`,
   `kinetic_temperature`, `TimedLoop`, `capture_oom`, `gather_rank_metrics`,
   `BenchRecord`, and `write_result` from `bench_common`.
2. Add model-specific CLI arguments to `build_base_parser(...)`, instantiate your
   model, and pass model-specific metadata via `BenchRecord(..., extra={...})`.
3. Submit batches through `scripts/sbatch_submit.sh` (pointing `EXP2_DATA_DIR` at
   `experiments/exp_datasets/zenodo/<dataset>`).
4. Summarize any results directory with:

```bash
python experiments/exp2_scaling/analysis/summarize_scaling.py \
    --results-dir experiments/exp2_scaling/results \
    --out-dir experiments/exp2_scaling/analysis \
    --title "<Model> Scaling"
```
