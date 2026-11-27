#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Slurm submission wrapper for exp2_scaling on Empire AI GB200 (beta).
#
# Enforces site rules:
#   - Always specifies --account (su_kchiu_molecular)
#   - Always allocates all 4 GPUs per node (--gpus-per-node=4 --mem=0)
#   - Automatically sets --segment=<nodes> (for N <= 18, keeps all nodes in
#     one NVL72 rack block so MNNVL GPUDirect Fabric handles can connect)
#
# Usage:
#   scripts/sbatch_submit.sh [--test] [--nodes=N] [--segment=S] [--time=HH:MM:SS] \
#       [--job-name=NAME] [extra sbatch flags...] <script.sbatch> [script args...]
#
# Set EXP2_DRY_RUN=1 to print the sbatch command without submitting.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
. "$EXP_DIR/cluster.env"

LOG_DIR="${EXP2_RESULTS_DIR:-$EXP_DIR/results}/logs"
[ "${EXP2_DRY_RUN:-0}" = "1" ] || mkdir -p "$LOG_DIR"

USE_TEST=0
NODES=1
SEGMENT=""
PASS_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --test)
            USE_TEST=1
            shift
            ;;
        --nodes=*)
            NODES="${1#--nodes=}"
            shift
            ;;
        --nodes|-N)
            NODES="$2"
            shift 2
            ;;
        --segment=*)
            SEGMENT="${1#--segment=}"
            shift
            ;;
        --segment)
            SEGMENT="$2"
            shift 2
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$USE_TEST" = "1" ]; then
    PART="$EXP2_TEST_PARTITION"
    QOS="$EXP2_TEST_QOS"
else
    PART="$EXP2_PARTITION"
    QOS="$EXP2_QOS"
fi

# Default segment to node count so multi-node jobs land in one NVL72 block (<=18 nodes)
if [ -z "$SEGMENT" ]; then
    SEGMENT="$NODES"
fi

export EXP2_SCRIPTS_DIR="$SCRIPT_DIR"

CMD=(sbatch)
[ "${EXP2_DRY_RUN:-0}" = "1" ] && CMD=(echo "+ sbatch")

# shellcheck disable=SC2086
exec "${CMD[@]}" \
    --account="$EXP2_ACCOUNT" \
    --partition="$PART" \
    --qos="$QOS" \
    --nodes="$NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node="$EXP2_GPUS_PER_NODE" \
    --cpus-per-task="$EXP2_CPUS_PER_TASK" \
    --mem="$EXP2_MEM" \
    --segment="$SEGMENT" \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    ${EXP2_SBATCH_EXTRA:-} \
    "${PASS_ARGS[@]}"
