#!/bin/bash
# ---------------------------------------------------------------------------
# UMA kernel-configuration triage sweep (RESTART.md section 6).
#
# Question this answers: with activation checkpointing OFF (a fixed decision --
# exp0 is a compute-kernel benchmark, memory is the secondary axis), which of
# `merge_mole` / `compile` actually works, and which one deadlocks FairChem's
# Ray graph-parallel workers?
#
# It is a 2x2 factorial in those two flags plus one memory-safe control, run as
# five INDEPENDENT subprocesses inside ONE allocation. Independent matters: a
# wedged Ray actor or a hung NCCL collective poisons its process, so each cell
# must be reaped on its own (run_cell.sbatch wraps every cell in `timeout` and
# writes a status=oom stub if it never returns).
#
#   sub-config     tf32  ackpt  merge_mole  compile   isolates
#   k0_tf32ac        1     1        0          0      control (ackpt ON)
#   k1_noac          1     0        0          0      baseline for the 2x2
#   k2_mole          1     0        1          0      merge_mole   <-- also
#                                                     unlocks umas_fast_gpu
#   k3_compile       1     0        0          1      compile
#   k4_turbo         1     0        1          1      both (== `turbo`)
#
# Read the result as: whichever of k2 / k3 hangs is the culprit; if both run and
# only k4 hangs, it is the interaction, not either flag alone.
#
# Submit (per engine) via the cluster-agnostic wrapper:
#
#   scripts/sbatch_exp0.sh --gpus-per-node=2 --job-name=tune-fc --time=01:40:00 \
#       scripts/tune_kernels.sbatch fairchem-uma exp_datasets/baselines/ch4_32k_splited.xyz 2
#
# Cost note (Delta, `PriorityFlags=MAX_TRES` -> billing = max(cpus*62.5,
# mem/16G, gpus*1000)): submit against gpuA100x4-preempt (GRES/gpu weight 500 =
# 0.5x) rather than gpuA100x4 (1000). gpuA40x4-preempt is 250 (0.25x) and has
# 48 GB/GPU, but A40 nodes have no NVLink -- for a question that is *about* an
# NCCL/Ray collective hang, prefer the A100 topology the real matrix uses.
# NEVER use gpuA100x4-interactive for this: weight 2000 = 2x.
#
#   EXP0_GPU_PARTITION=gpuA100x4-preempt scripts/sbatch_exp0.sh ...
# ---------------------------------------------------------------------------
#SBATCH --job-name=exp0-tune
#SBATCH --time=01:40:00

set -uo pipefail

_S="${EXP0_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$_S/_lib.sh" ] || { echo "ERROR: scripts/_lib.sh not found in '$_S' -- submit via scripts/sbatch_exp0.sh"; exit 1; }

ENGINE="${1:?engine: distnnp-uma | fairchem-uma}"
XYZ="${2:?xyz path}"
NGPU="${3:-2}"

# Keep the sweep out of the main matrix dir: these cells are deliberately on
# different kernel configurations and must never be tabulated alongside it.
export EXP0_RESULTS_DIR="${EXP0_RESULTS_DIR:-$(cd "$_S/.." && pwd)/results/tune}"
mkdir -p "$EXP0_RESULTS_DIR/logs"
export EXP0_SCRIPTS_DIR="$_S"

# Absolute, so the cd-into-spool-copy that Slurm does cannot break it.
case "$XYZ" in /*) ;; *) XYZ="$(cd "$_S/.." && pwd)/$XYZ" ;; esac
[ -f "$XYZ" ] || { echo "ERROR: input not found: $XYZ"; exit 1; }

echo "=================================================================="
echo " exp0 kernel triage: engine=$ENGINE  ngpu=$NGPU"
echo " input=$XYZ"
echo " results=$EXP0_RESULTS_DIR"
echo " partition=${SLURM_JOB_PARTITION:-?}  job=${SLURM_JOB_ID:-?}"
echo "=================================================================="

# tag | hard_timeout_s | warmup | timed | driver flags
#
# compile cells get a much larger budget (torch.compile's first trace on eSCN is
# minutes) and a longer warmup, so the recompile storm settles *before* the
# timed window rather than inside it. Timing still excludes warmup entirely.
CONFIGS=(
  "k0_tf32ac|600|5|5|--tf32 1 --activation-checkpointing 1 --merge-mole 0 --compile 0"
  "k1_noac|600|5|5|--tf32 1 --activation-checkpointing 0 --merge-mole 0 --compile 0"
  "k2_mole|900|5|5|--tf32 1 --activation-checkpointing 0 --merge-mole 1 --compile 0"
  "k3_compile|1500|12|8|--tf32 1 --activation-checkpointing 0 --merge-mole 0 --compile 1"
  "k4_turbo|1500|12|8|--inference-settings turbo"
)

# Only run the sub-configs named in EXP0_TUNE_ONLY (space-separated), if set --
# lets a follow-up job re-test just the cell that hung without paying for the
# four that already answered.
ONLY="${EXP0_TUNE_ONLY:-}"

rc_all=0
for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r tag tmo warm timed flags <<<"$spec"

    if [ -n "$ONLY" ] && ! grep -qw -- "$tag" <<<"$ONLY"; then
        echo; echo "--- skip $tag (not in EXP0_TUNE_ONLY) ---"
        continue
    fi

    echo
    echo "################################################################"
    echo "### $tag   timeout=${tmo}s  warmup=$warm timed=$timed"
    echo "###   flags: $flags"
    echo "################################################################"
    t0=$SECONDS

    # A separate process per cell. run_cell.sbatch owns the `timeout` wrapper,
    # the stub-JSON-on-hang trap and the module/venv bootstrap, so a cell that
    # deadlocks is reaped and recorded without taking the sweep down with it.
    CELL_TAG="$tag" HARD_TIMEOUT="$tmo" \
        bash "$_S/run_cell.sbatch" "$ENGINE" "$XYZ" "$NGPU" \
            --warmup-steps "$warm" --timed-steps "$timed" \
            --rebalance-interval 1000 \
            $flags
    rc=$?
    [ "$rc" -ne 0 ] && rc_all=$rc
    echo "### $tag done rc=$rc in $((SECONDS - t0))s"

    # A killed cell can leave Ray actors / NCCL shared memory behind that the
    # NEXT cell would then inherit. Clear both before moving on.
    pkill -u "$USER" -f 'ray::' 2>/dev/null
    pkill -u "$USER" -f 'raylet'  2>/dev/null
    pkill -u "$USER" -f 'run_fairchem_uma.py' 2>/dev/null
    pkill -u "$USER" -f 'run_distnnp_uma.py'  2>/dev/null
    sleep 10
    rm -rf /dev/shm/nccl-* 2>/dev/null
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
done

echo
echo "=================================================================="
echo " sweep done (last non-zero rc=$rc_all). Report:"
echo "   \$VENV/bin/python analysis/tune_report.py --results-dir results/tune"
echo "=================================================================="
exit 0
