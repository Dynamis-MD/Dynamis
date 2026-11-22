#!/bin/bash
# ---------------------------------------------------------------------------
# exp0 MULTI-NODE FEASIBILITY: DistNNP-UMA distributed MD on the 9GMO solvated
# complex (9gmo_density.xyz, 1,536,442 atoms) across MULTIPLE NODES.
#
# SCOPE: feasibility only. Prove the run stands up across nodes and produces
# a stable steps/s, then STOP. Not a scientific production run -- short step
# windows, <=15 min walltime, hard-timeout well inside that.
#
# It launches one torchrun per node under srun with a c10d rendezvous, and
# runs the DistNNP driver with the multi-node halo exchanger:
#
#     run_distnnp_uma.py --exchanger nccl   (torch.distributed all_to_all;
#     GPUDirect/CUDA-IPC has no cross-node peer and would hang the watchdog)
#
# Kernel config = RESTART.md's "k0": TF32 on + activation-checkpointing on +
# SFC-Morton partitioner -- the memory-safe reference config that reached
# ~32k atoms/GPU on 40GB A100s. SFC-Morton builds far faster than kdtree.
#
# ---------------------------------------------------------------------------
# SUBMIT -- you supply the site bits, the script has NO site #SBATCH lines:
#
#     cd experiments/exp0_end_to_end
#     sbatch -p <gpu-partition> [-A <account>] \
#            --gpus-per-node=4 \        # OR:  --gres=gpu:4   (whichever your Slurm takes)
#            scripts/hero_9gmo.sbatch
#
#   fewer / more nodes:   sbatch ... --nodes=12 scripts/hero_9gmo.sbatch
#   8 GPUs per node:      sbatch ... --gres=gpu:8 --nodes=8 EXP0_HERO_GPN=8 ...
#
# Config comes from experiments/exp0_end_to_end/scripts/cluster.local.env
# (modules, venv path, NCCL fabric interface) -- set that up first.
#
# Per-run env overrides:
#     EXP0_HERO_GPN=4            GPUs per node the RUN uses (torchrun ranks/node
#                                + the atoms/GPU math). Must match your GPU
#                                request. Default 4; auto-checked vs nvidia-smi.
#     EXP0_HERO_NODES=16         node count if you don't pass --nodes
#     EXP0_HERO_XYZ=exp_datasets/protein_solvated/9gmo_density.xyz
#     EXP0_HERO_TAG=feas_9gmo_64gpu
#     EXP0_HERO_PARTITION=sfc_morton    (sfc_hilbert / kdtree / simple)
#     EXP0_HERO_TF32=1 / EXP0_HERO_ACKPT=1
#     EXP0_HERO_INFERENCE=default
#     EXP0_HERO_EXTRA="--warmup-steps 3 --timed-steps 10"
#     EXP0_HERO_HARD_TIMEOUT=720        (seconds; keep < the walltime in seconds)
#
# ---------------------------------------------------------------------------
# SIZING (TF32 + ackpt-on UMA-s, 40GB A100 -- ~32k atoms/GPU ceiling). 1.54M atoms:
#     atoms/GPU   GPUs   nodes@4/node
#     ~24k        64     16     <- default (headroom)
#     ~32k        48     12     <- at the ceiling
#     ~48k        32      8     <- expect status=oom (still a data point)
# On 80GB+ GPUs the ceiling roughly doubles. status=oom here is DATA, not a
# failure -- the driver writes the record either way.
# ---------------------------------------------------------------------------
#SBATCH --job-name=exp0-feas-9gmo
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/hero_9gmo_%j.out
#SBATCH --error=results/logs/hero_9gmo_%j.err

set -uo pipefail

# --- locate the exp0 _lib.sh -------------------------------------------------
# Under `sbatch`, ${BASH_SOURCE[0]} is a spool copy, not scripts/. Resolve via,
# in order: `scontrol` (the real submitted path -- most reliable under Slurm);
# an explicit EXP0_SCRIPTS_DIR; the submit dir and its parents; the BASH_SOURCE
# dir and cwd (covers `bash scripts/hero_9gmo.sbatch`).
_find_lib() {
    local rel="experiments/exp0_end_to_end/scripts/_lib.sh"
    local sd="" c
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        sd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -n1)"
        [ -n "$sd" ] && sd="$(dirname "$sd")"
    fi
    for c in \
        "${sd:+$sd/_lib.sh}" \
        "${EXP0_SCRIPTS_DIR:-}/_lib.sh" \
        "${SLURM_SUBMIT_DIR:-}/$rel" \
        "${SLURM_SUBMIT_DIR:-}/../$rel" \
        "${SLURM_SUBMIT_DIR:-}/../../$rel" \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)/_lib.sh" \
        "$(pwd)/scripts/_lib.sh" "$(pwd)/$rel"
    do
        [ -n "$c" ] && [ -f "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}
_LIB="$(_find_lib)" || {
    echo "ERROR: cannot find experiments/exp0_end_to_end/scripts/_lib.sh."
    echo "  Run:  sbatch --chdir=<repo>/experiments/exp0_end_to_end ... scripts/hero_9gmo.sbatch"
    echo "  or:   EXP0_SCRIPTS_DIR=<repo>/experiments/exp0_end_to_end/scripts sbatch ..."
    exit 1
}
# shellcheck source=/dev/null
source "$_LIB"

: "${SLURM_JOB_PARTITION:?submit with  -p <gpu-partition>  (add -A <account> if your cluster needs one)}"

exp0_check_venv
exp0_load_modules
exp0_export_runtime

XYZ="${EXP0_HERO_XYZ:-$EXP_DIR/exp_datasets/protein_solvated/9gmo_density.xyz}"
[ -f "$XYZ" ] || { echo "ERROR: system not found: $XYZ  -- run scripts/setup_datasets.sh first"; exit 1; }

RESULTS="$RESULTS_DIR"
mkdir -p "$RESULTS/logs"

NNODES="${SLURM_NNODES:-${EXP0_HERO_NODES:?set --nodes or EXP0_HERO_NODES}}"

# GPUs per node the RUN uses. Trust EXP0_HERO_GPN, else what Slurm recorded, else 4.
GPN="${EXP0_HERO_GPN:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}}"
GPN="${GPN##*:}"                                  # strip a "gpu:" / "type:" prefix
# this script runs on the first allocated node, which has GPUs -- no srun needed
_seen="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "${_seen:-0}" -gt 0 ] && [ "$_seen" != "$GPN" ]; then
    echo "WARNING: using GPN=$GPN but this node shows $_seen GPUs. If GPN is wrong,"
    echo "         set EXP0_HERO_GPN and match your --gpus-per-node / --gres request."
fi
WORLD=$(( NNODES * GPN ))

NUM_ATOMS="$(head -n1 "$XYZ" | tr -d '[:space:]')"
TAG="${EXP0_HERO_TAG:-feas_9gmo_${WORLD}gpu}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${EXP0_HERO_PORT:-29518}"
RDZV="--rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=${SLURM_JOB_ID}"

export NCCL_DEBUG="${EXP0_NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INFERENCE="${EXP0_HERO_INFERENCE:-default}"
PARTITION="${EXP0_HERO_PARTITION:-sfc_morton}"
TF32="${EXP0_HERO_TF32:-1}"
ACKPT="${EXP0_HERO_ACKPT:-1}"

# Feasibility, not science: 3 warmup + 10 timed is enough for a stable steps/s.
# build-timeout bumped from 420s: cold model load + a 1.5M-atom partition is
# slower than the synthetic matrix cells it was tuned for. EXP0_HERO_EXTRA is
# appended last so it wins on any repeated flag.
read -r -a EXTRA <<< "${EXP0_HERO_EXTRA:-}"
DEFAULT_STEPS=(--warmup-steps 3 --timed-steps 10 --rebalance-interval 50
               --build-timeout-s 540
               --inference-settings "$INFERENCE" --tf32 "$TF32"
               --activation-checkpointing "$ACKPT"
               --partition-algorithm "$PARTITION")

echo "=========================================================================="
echo " exp0 multi-node feasibility run"
echo "   system      : $(basename "$XYZ")  ($NUM_ATOMS atoms)"
echo "   nodes       : $NNODES x $GPN GPU = $WORLD ranks   (~$(( NUM_ATOMS / WORLD )) atoms/GPU)"
echo "   nodelist    : $SLURM_JOB_NODELIST"
echo "   master      : $MASTER_ADDR:$MASTER_PORT"
echo "   partition   : $SLURM_JOB_PARTITION   account: ${SLURM_JOB_ACCOUNT:-none}"
echo "   venv        : $VENV"
echo "   kernel cfg  : $INFERENCE + tf32=$TF32 + ackpt=$ACKPT   partitioner: $PARTITION"
echo "   exchanger   : nccl (torch.distributed all_to_all)"
echo "   result      : $RESULTS/distnnp-uma_${NUM_ATOMS}atoms_${WORLD}gpu_${TAG}.json"
echo "=========================================================================="
"$PY" --version
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'

# One torchrun per node; each spawns $GPN local ranks.
# Hard timeout stays inside the walltime: a dead rank -> ~600s NCCL collective
# timeout, and `timeout` must reap the tree before Slurm SIGKILLs the step
# (which would leave no result JSON).
HARD_TIMEOUT="${EXP0_HERO_HARD_TIMEOUT:-720}"
RC=0
timeout --signal=TERM --kill-after=60 "$HARD_TIMEOUT" \
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpu-bind=none \
    "$TORCHRUN" --nnodes="$NNODES" --nproc-per-node="$GPN" $RDZV \
        "$EXP_DIR/drivers/run_distnnp_uma.py" \
        --input "$XYZ" --engine distnnp-uma \
        --exchanger nccl \
        --output-dir "$RESULTS" --tag "$TAG" \
        "${DEFAULT_STEPS[@]}" "${EXTRA[@]}" || RC=$?

echo "=== feasibility run finished (rc=$RC) ==="
RESULT_JSON="$RESULTS/distnnp-uma_${NUM_ATOMS}atoms_${WORLD}gpu_${TAG}.json"
if [ -f "$RESULT_JSON" ]; then
    echo "--- $RESULT_JSON ---"
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d.get(k) for k in ('status','world_size','nodes','num_atoms','steps_per_sec','ns_per_day','peak_mem_mb_max','mean_temperature_k','exchanger','error')}, indent=2))" "$RESULT_JSON"
else
    echo "WARNING: no result JSON at $RESULT_JSON (rc=$RC) -- check the .err log"
fi
exit "$RC"
