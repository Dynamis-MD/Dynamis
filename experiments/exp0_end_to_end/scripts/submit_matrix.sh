#!/bin/bash
# Enumerate the exp0 job matrix and submit one job per cell (throttled), via
# the cluster-agnostic wrapper scripts/sbatch_exp0.sh.
#
#   bash scripts/submit_matrix.sh [--dry-run] [--only strong|fill|ceiling] [--engine distnnp-uma|fairchem-uma]
#
# Matrix (see implementation_plan.md / README). This is a *performance*
# benchmark on synthetic isotropic dense-fluid boxes -- short warmup that still
# covers CUDA init + first-call graph/alloc, a timed window below the first
# repartition (warmup+timed < rebalance_interval), tight walltime caps so
# FairChem OOM cells (which hang rather than raise) are reaped fast.
#
#   strong   : both x  32k @ {1,2,4}                 -- strong scaling
#   fill     : both x  {32k@1, 64k@2, 128k@4}        -- ~32k atoms/GPU (compute-bound)
#   ceiling  : both x  {192k, 256k} @ 4              -- push to OOM on 4 GPU
#                                                       (128k@4 comes from `fill`)
#
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
GPU_MAX="${EXP0_GPUS_PER_NODE_MAX:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"

DRY_RUN=0; ONLY=""; ENGINE_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --only) ONLY="$2"; shift ;;
    --engine) ENGINE_FILTER="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

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
  local subexp="$1" engine="$2" size_k="$3" ngpu="$4"
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
  local name="exp0-${subexp}-${engine}-${size_k}k-${ngpu}gpu"
  local cmd=("$SBATCH" --job-name="$name" --gpus-per-node="$ngpu" --time="00:20:00"
             --export=ALL,CELL_TAG="${subexp}",HARD_TIMEOUT="${hard}"
             "$RUN_CELL" "$engine" "$xyz" "$ngpu"
             --timed-steps "$timed" --warmup-steps "$WARMUP" --rebalance-interval 1000)
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY: ${cmd[*]}"
  else
    throttle
    "${cmd[@]}"
  fi
}

for engine in distnnp-uma fairchem-uma; do
  # strong scaling -- fixed 32k, vary GPUs
  for ngpu in 1 2 4; do submit strong "$engine" 32 "$ngpu"; done
  # fill scaling -- ~32k atoms/GPU
  submit fill "$engine" 32  1
  submit fill "$engine" 64  2
  submit fill "$engine" 128 4
  # OOM ceiling on 4 GPU (128k@4 already covered by `fill`)
  for size_k in 192 256; do submit ceiling "$engine" "$size_k" 4; done
done

echo "submitted (dry_run=$DRY_RUN). watch: squeue -u $USER"
