#!/bin/bash
# Enumerate the exp0 job matrix and submit one job per cell (throttled), via
# the cluster-agnostic wrapper scripts/sbatch_exp0.sh.
#
#   bash scripts/submit_matrix.sh [--dry-run] [--with-lammps] \
#        [--only strong|fill|ceiling|lammps|lammps-1gpu|lammps-naive] \
#        [--engine distnnp-uma|distnnp-uma-opt|fairchem-uma|lammps-mace]
#
# distnnp-uma / fairchem-uma run the MATCHED kernel config (k1: tf32 + no
# activation checkpointing -- the fastest config FairChem's Ray path survives).
# distnnp-uma-opt is the SAME DistNNP driver at `--inference-settings turbo`
# (+ merge_mole + compile) -- its own curve, ~1.6x faster in the triage.
#
# Matrix (see implementation_plan.md / README). This is a *performance*
# benchmark on synthetic isotropic dense-fluid boxes -- short warmup that still
# covers CUDA init + first-call graph/alloc, a timed window below the first
# repartition (warmup+timed < rebalance_interval), tight walltime caps so
# FairChem OOM cells (which hang rather than raise) are reaped fast.
#
#   strong      : all engines x 32k @ {1,2,4}              -- strong scaling
#   fill        : all engines x {32k@1, 64k@2, 128k@4}     -- ~32k atoms/GPU
#   ceiling     : all engines x {192k, 256k} @ 4           -- push to OOM on 4 GPU
#   lammps-1gpu : lammps-mace x {2,16,32,64,128,192,256}k @ 1  -- single-GPU OOM sweep
#                 (the fallback headline if MACE ML-IAP multi-GPU is broken, mace#1294)
#   lammps-naive: lammps-mace x {128,192,256}k @ 4, --comm-cutoff  -- naive ghost shell
#
# lammps-mace is only enumerated when its `lmp` binary exists (or --with-lammps).
# The number of GPUs requested for any cell is capped at EXP0_GPUS_PER_NODE_MAX.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.env
. "$SCRIPTS_DIR/cluster.env"
EXP_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
BASELINES="$EXP_DIR/exp_datasets/baselines"
RUN_CELL="$SCRIPTS_DIR/run_cell.sbatch"
SBATCH="$SCRIPTS_DIR/sbatch_exp0.sh"

WARMUP=8
# distnnp-uma-opt runs `--inference-settings turbo` -> torch.compile. fairchem
# compiles with `dynamic=True` (predict.py), so varying node/edge counts do NOT
# retrace; the only extra warmup cost is the one-time first trace plus a few
# dynamo 0/1-dim specializations, which a warmup of 8 already covers. The real
# compile budget is wall-time, handled by `hard` below, not by warmup count.
OPT_WARMUP="${OPT_WARMUP:-8}"
GPU_MAX="${EXP0_GPUS_PER_NODE_MAX:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"

DRY_RUN=0; ONLY=""; ENGINE_FILTER=""; WITH_LAMMPS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --with-lammps) WITH_LAMMPS=1 ;;
    --only) ONLY="$2"; shift ;;
    --engine) ENGINE_FILTER="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# lammps-mace is opt-in: only enumerate it when its binary is built (or forced).
LAMMPS_ENGINES=()
if [ -x "${EXP0_LAMMPS_BIN:-/nonexistent}" ] || [ "$WITH_LAMMPS" = 1 ] \
   || [ "$ENGINE_FILTER" = "lammps-mace" ]; then
  LAMMPS_ENGINES=(lammps-mace)
fi

xyz_for() { echo "$BASELINES/ch4_${1}k_splited.xyz"; }
want()    { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
want_eng(){ [ -z "$ENGINE_FILTER" ] || [ "$ENGINE_FILTER" = "$1" ]; }

throttle() {
  [ "$DRY_RUN" -eq 1 ] && return 0
  while [ "$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^exp0-')" -ge "$MAX_INFLIGHT" ]; do
    sleep 30
  done
}

submit() {
  local subexp="$1" engine="$2" size_k="$3" ngpu="$4" comm_cutoff="${5:-0}"
  want "$subexp" || return 0
  want_eng "$engine" || return 0
  if [ "$ngpu" -gt "$GPU_MAX" ]; then
    echo "SKIP ${subexp}/${engine}/${size_k}k/${ngpu}gpu -- needs $ngpu > EXP0_GPUS_PER_NODE_MAX=$GPU_MAX"
    return 0
  fi
  local xyz; xyz="$(xyz_for "$size_k")"
  if [ ! -f "$xyz" ]; then
    echo "MISSING $xyz -- run scripts/gen_datasets.sh $size_k"; return 1
  fi
  local timed=20;  [ "$size_k" -ge 192 ] && timed=12
  local hard=700;  [ "$size_k" -ge 192 ] && hard=900
  # torch.compile's first trace on eSCN is minutes -- the -opt cells need a much
  # larger wall budget so a slow compile is not misreported as an OOM/hang.
  case "$engine" in distnnp-uma-opt) hard=$((hard + 1200));; esac
  local model_tag="uma-s-1p2"; case "$engine" in lammps-*) model_tag="${EXP0_MACE_MODEL_NAME:-mace-mpa-0}";; esac
  local name="exp0-${subexp}-${engine}-${size_k}k-${ngpu}gpu"
  local extra=(--timed-steps "$timed" --warmup-steps "$WARMUP" --rebalance-interval 1000)
  case "$engine" in
    lammps-*)
      extra=(--timed-steps "$timed" --warmup-steps "$WARMUP")
      [ "$comm_cutoff" != "0" ] && extra+=(--comm-cutoff "$comm_cutoff")
      ;;
    distnnp-uma|fairchem-uma)
      # The MATCHED axis: k1 -- the highest kernel config FairChem can run
      # (merge_mole deadlocks its Ray workers, compile breaks MOLE under dynamo;
      # see analysis/tune_report.py). Applied identically to both engines.
      extra+=(--tf32 1 --activation-checkpointing 0)
      ;;
    distnnp-uma-opt)
      # DistNNP's fastest safe config -- run_cell.sbatch injects
      # `--inference-settings turbo` (tf32 + no-ackpt + merge_mole + compile).
      # No kernel flags here; only the (env-overridable) warmup.
      extra=(--timed-steps "$timed" --warmup-steps "$OPT_WARMUP" --rebalance-interval 1000)
      ;;
  esac
  # Walltime tracks the hard cell timeout + model-load slack; the -opt compile
  # cells run long enough to need more than the 25 min the rest use.
  local wall="00:25:00"; case "$engine" in distnnp-uma-opt) wall="00:50:00";; esac
  local cmd=("$SBATCH" --job-name="$name" --gpus-per-node="$ngpu" --time="$wall"
             --export=ALL,CELL_TAG="${subexp}",HARD_TIMEOUT="${hard}",CELL_MODEL="${model_tag}"
             "$RUN_CELL" "$engine" "$xyz" "$ngpu" "${extra[@]}")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY: ${cmd[*]}"
  else
    throttle
    "${cmd[@]}"
  fi
}

for engine in distnnp-uma distnnp-uma-opt fairchem-uma "${LAMMPS_ENGINES[@]}"; do
  # strong scaling -- fixed 32k, vary GPUs
  for ngpu in 1 2 4; do submit strong "$engine" 32 "$ngpu"; done
  # fill scaling -- ~32k atoms/GPU
  submit fill "$engine" 32  1
  submit fill "$engine" 64  2
  submit fill "$engine" 128 4
  # OOM ceiling on 4 GPU (128k@4 already covered by `fill`)
  for size_k in 192 256; do submit ceiling "$engine" "$size_k" 4; done
done

# --- LAMMPS-MACE: strong-scaling on small boxes + single-GPU OOM sweep +
#     the naive extended-ghost-shell series -----------------------------------
# MACE-MPA-medium's cuEq TP kernel is memory-hungry: at float64 the envelope was
# only ~8-12k atoms/GPU (see results/_archive_float64/). At float32 (the exp0
# baseline -- the DistNNP-MACE adapter is FP32) it should roughly double. The
# small-box strong sweep gives a clean 1->2->4 scaling curve wherever the
# envelope lands; the 1-GPU sweep finds the single-GPU ceiling.
if [ "${#LAMMPS_ENGINES[@]}" -gt 0 ]; then
  # 32k@{1,2,4} strong is already emitted by the per-engine loop above; add the
  # smaller boxes here (needs ch4_{4,8}k -- gen_datasets.sh / 03_generate_ch4.py).
  for size_k in 8 16; do
    for ngpu in 1 2 4; do submit strong lammps-mace "$size_k" "$ngpu"; done
  done
  for size_k in 2 4 8 16 32 64 128 192 256; do
    submit lammps-1gpu lammps-mace "$size_k" 1
  done
  # ~2 * r_max for a 2-layer medium-mpa-0 (r_max ~ 5.5 A); override via CLI.
  NAIVE_CUTOFF="${EXP0_MACE_COMM_CUTOFF:-11.0}"
  for size_k in 128 192 256; do
    submit lammps-naive lammps-mace "$size_k" 4 "$NAIVE_CUTOFF"
  done
fi

echo "submitted (dry_run=$DRY_RUN). watch: squeue -u $USER"
