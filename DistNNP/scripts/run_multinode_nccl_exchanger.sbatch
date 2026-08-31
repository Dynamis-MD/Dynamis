#!/bin/bash
# ---------------------------------------------------------------------------
# Multi-node verification for PythonNCCLExchanger (feature exchange over
# torch.distributed all_to_all instead of CUDA-IPC / GPUDirect).
#
# GPUDirectExchanger can only halo-exchange between GPUs on the SAME node
# (cudaIpcOpenMemHandle has no cross-node peer), so any >1-node run hangs in
# the NCCL watchdog. PythonNCCLExchanger should make >1-node runs work.
#
#   Stage 0  (2 node) cross-node transport smoke -- HARD GATE       (~1 min)
#   Stage 1  (1 node) exchanger fwd/bwd + dynamic topology (pytest)
#   Stage 2  (2 node) distributed UMA MD, --exchanger nccl
#   Stage 3  (1 node, 1 GPU) single-GPU reference run (no parity compare)
#
# ---------------------------------------------------------------------------
# SUBMIT -- no site #SBATCH lines; you supply partition, (account), GPUs:
#
#     cd <repo>/experiments/exp0_end_to_end        # for scripts/cluster.local.env
#     sbatch -p <gpu-partition> [-A <account>] \
#            --gpus-per-node=2 \                    # OR: --gres=gpu:2
#            ../../DistNNP/scripts/run_multinode_nccl_exchanger.sbatch
#
# Modules / venv path / NCCL fabric interface all come from
# experiments/exp0_end_to_end/scripts/cluster.local.env (same file the rest of
# exp0 uses). Set that up first.  Per-run env overrides:
#     MNV_GPN=2            GPUs per node the run uses (must match your request)
#     MNV_MASTER_PORT      rendezvous port (default 29517)
# ---------------------------------------------------------------------------
#SBATCH --job-name=multinode_nccl_exchanger
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/multinode_nccl_exchanger_%j.out
#SBATCH --error=logs/multinode_nccl_exchanger_%j.err

set -u

: "${SLURM_JOB_PARTITION:?submit with  -p <gpu-partition>  (add -A <account> if your cluster needs one)}"

# --- locate the exp0 _lib.sh (see hero_9gmo.sbatch for the rationale) -------
_find_lib() {
    local rel="experiments/exp0_end_to_end/scripts/_lib.sh"
    local sd="" c
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        sd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -n1)"
        # this script is <repo>/DistNNP/scripts/<name>; _lib.sh is two dirs over
        [ -n "$sd" ] && sd="$(cd "$(dirname "$sd")/../.." 2>/dev/null && pwd)/experiments/exp0_end_to_end/scripts"
    fi
    for c in \
        "${sd:+$sd/_lib.sh}" \
        "${EXP0_SCRIPTS_DIR:-}/_lib.sh" \
        "${SLURM_SUBMIT_DIR:-}/$rel" \
        "${SLURM_SUBMIT_DIR:-}/../$rel" \
        "${SLURM_SUBMIT_DIR:-}/../../$rel" \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)/$rel" \
        "$(pwd)/$rel"
    do
        [ -n "$c" ] && [ -f "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}
_LIB="$(_find_lib)" || {
    echo "ERROR: cannot find experiments/exp0_end_to_end/scripts/_lib.sh."
    echo "  Submit from <repo>/experiments/exp0_end_to_end, or set EXP0_SCRIPTS_DIR."
    exit 1
}
# shellcheck source=/dev/null
source "$_LIB"

exp0_check_venv
exp0_load_modules
exp0_export_runtime          # sets NCCL_SOCKET_IFNAME from EXP0_NCCL_SOCKET_IFNAME
export HF_HUB_OFFLINE="${EXP0_HF_HUB_OFFLINE:-1}"

REPO="$MONO"                  # <repo>/DistNNP
[ -d "$REPO/distnnp_model" ] || { echo "ERROR: \$MONO=$REPO has no distnnp_model/"; exit 1; }
cd "$REPO"
mkdir -p logs scripts/_mnv_runs
WORKDIR=$(mktemp -d "$REPO/scripts/_mnv_runs/${SLURM_JOB_ID}_XXXX")   # shared FS (required)

GPN="${MNV_GPN:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}}"; GPN="${GPN##*:}"
NN="${SLURM_NNODES:?}"

echo "=== nodes:    $SLURM_JOB_NODELIST  (${NN} x ${GPN} GPU)"
echo "=== workdir:  $WORKDIR"
echo "=== venv:     $VENV"
echo "=== python:   $("$PY" --version 2>&1)  torch $("$PY" -c 'import torch;print(torch.__version__)' 2>&1)"
srun --ntasks="$NN" --ntasks-per-node=1 bash -c 'echo "  $(hostname): $(nvidia-smi -L 2>/dev/null | wc -l) GPU"'

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT="${MNV_MASTER_PORT:-29517}"
export NCCL_DEBUG="${EXP0_NCCL_DEBUG:-WARN}"
RDZV="--rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=${SLURM_JOB_ID}"

run_dist () {   # $@ = script + args, launched NN node / (NN*GPN) rank under torchrun
    srun --ntasks="$NN" --ntasks-per-node=1 \
        "$TORCHRUN" --nnodes="$NN" --nproc-per-node="$GPN" $RDZV "$@"
}

echo
echo "############ Stage 0: cross-node transport smoke (HARD GATE) ############"
run_dist distnnp_model/tests/mn_transport_smoke.py
STAGE0=$?
echo "Stage 0 exit: $STAGE0"
[ $STAGE0 -ne 0 ] && { echo "Stage 0 failed -- cross-node transport broken. Stopping."; exit 1; }

echo
echo "############ Stage 1: exchanger pytest (this node, ${GPN} GPU) ############"
"$PY" -m pytest -x -q distnnp_model/tests/test_python_nccl_exchanger.py
echo "Stage 1 exit: $?"

echo
echo "############ Stage 2: distributed MD, --exchanger nccl (${NN} node) ############"
run_dist distnnp_model/tests/run_distributed_trajectory.py --path "$WORKDIR" --exchanger nccl
STAGE2=$?
echo "Stage 2 exit: $STAGE2"

echo
echo "############ Stage 3: single-GPU reference run (1 node / 1 rank) ############"
# world_size=1 baseline for a reference throughput number and to prove the
# single-process path stands up. No trajectory parity compare -- exp0 is
# throughput-only. Own workdir so it does not clobber Stage 2's dump.
REFDIR="${WORKDIR}_ref"
mkdir -p "$REFDIR"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=1 \
        distnnp_model/tests/run_distributed_trajectory.py --path "$REFDIR" --exchanger nccl
echo "Stage 3 exit: $?  (artifacts in $REFDIR)"

echo
echo "=== done. artifacts in $WORKDIR ==="
