#!/bin/bash
# ---------------------------------------------------------------------------
# Cluster-agnostic `sbatch` wrapper. Injects --account / --partition / --mem /
# --cpus-per-task / --output from scripts/cluster.env so the .sbatch files
# themselves carry NO site-specific #SBATCH lines.
#
#   scripts/sbatch_exp0.sh [--cpu] [extra sbatch flags] <script.sbatch> [script args]
#
#   --cpu   use EXP0_CPU_ACCOUNT / EXP0_CPU_PARTITION / EXP0_CPU_MEM
#           (default is the GPU queue). env_setup.sh is the only --cpu job.
#
# Examples:
#   scripts/sbatch_exp0.sh --cpu scripts/env_setup.sh
#   scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke.sbatch
#   scripts/sbatch_exp0.sh --gpus-per-node=4 --time=00:20:00 \
#       scripts/run_cell.sbatch distnnp-uma exp_datasets/baselines/ch4_32k_splited.xyz 4
# ---------------------------------------------------------------------------
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.env
. "$D/cluster.env"
EXP_DIR="$(cd "$D/.." && pwd)"
mkdir -p "$EXP_DIR/results/logs"

QUEUE=gpu
case "${1:-}" in
  --cpu) QUEUE=cpu; shift ;;
  --gpu) shift ;;
esac

if [ "$QUEUE" = cpu ]; then
  ACCT="$EXP0_CPU_ACCOUNT"; PART="$EXP0_CPU_PARTITION"; MEM="$EXP0_CPU_MEM"
else
  ACCT="$EXP0_GPU_ACCOUNT"; PART="$EXP0_GPU_PARTITION"; MEM="$EXP0_GPU_MEM"
fi

# Slurm runs the batch script from a spool copy, so ${BASH_SOURCE[0]} inside it
# no longer points at scripts/. Hand the real location through the job env
# (propagated by sbatch's default --export=ALL and by --export=ALL,VAR=VAL).
export EXP0_SCRIPTS_DIR="$D"

RUN=(sbatch)
[ "${EXP0_DRY_RUN:-0}" = "1" ] && RUN=(echo "+ sbatch")

# EXP0_SBATCH_EXTRA is intentionally unquoted (may hold multiple flags).
# shellcheck disable=SC2086
exec "${RUN[@]}" \
  --account="$ACCT" \
  --partition="$PART" \
  --nodes="${EXP0_NODES:-1}" \
  --cpus-per-task="$EXP0_CPUS_PER_TASK" \
  --mem="$MEM" \
  --output="$EXP_DIR/results/logs/%x_%j.out" \
  ${EXP0_SBATCH_EXTRA:-} \
  "$@"
