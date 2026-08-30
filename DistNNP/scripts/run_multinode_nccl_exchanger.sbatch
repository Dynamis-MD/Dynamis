#!/bin/bash
# ---------------------------------------------------------------------------
# Multi-node verification for PythonNCCLExchanger (feature exchange over
# torch.distributed all_to_all instead of CUDA-IPC / GPUDirect).
#
# GPUDirectExchanger can only halo-exchange between GPUs on the SAME node
# (cudaIpcOpenMemHandle has no cross-node peer), so any >1-node run hangs in
# the NCCL watchdog. PythonNCCLExchanger should make >1-node runs work.
#
#   Stage 0  (2 node, 4 rank) cross-node transport smoke        (~1 min)
#   Stage 1  (1 node, 2 GPU)  exchanger fwd/bwd + dynamic topology (pytest)
#   Stage 2  (2 node, 4 rank) distributed UMA MD, --exchanger nccl
#   Stage 3  (1 node, 1 GPU)  single-GPU reference run (throughput baseline;
#                             no trajectory parity compare)
#
# Uses .venv-uma (torch 2.8 + fairchem): the only env with Simulator +
# PythonNCCLExchanger + UMAModel + pytest all working.
#
#   sbatch scripts/run_multinode_nccl_exchanger.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=multinode_nccl_exchanger
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/multinode_nccl_exchanger_%j.out
#SBATCH --error=logs/multinode_nccl_exchanger_%j.err
#
# partition + account come from the submit line (they differ per cluster):
#   sbatch -p <gpu-partition> -A <account> scripts/run_multinode_nccl_exchanger.sbatch
#
# Per-cluster overrides (env, all have Delta defaults):
#   MNV_MODULES            module load args      (default: Delta's python+cuda+aws-ofi-nccl)
#   MNV_VENV               path to .venv-uma     (default: <repo>/.venv-uma)
#   MNV_NCCL_SOCKET_IFNAME high-speed fabric if  (default: hsn; set "" to auto-detect)
#   MNV_MASTER_PORT        rendezvous port       (default: 29517)

set -u

: "${SLURM_JOB_PARTITION:?submit with  -p <gpu-partition> -A <account>}"

_MODS="${MNV_MODULES:-python/3.13.5-gcc13.3.1 cudatoolkit/26.5_12.9-unsupported gcc-native/14 aws-ofi-nccl}"
if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $_MODS 2>/dev/null || module load ${_MODS%% *}
fi

# Repo root (the DistNNP/ monorepo dir that holds distnnp_model/): prefer an
# explicit MNV_REPO, else the sbatch submit dir, else the Delta path.
REPO="${MNV_REPO:-${SLURM_SUBMIT_DIR:-}}"
[ -d "$REPO/distnnp_model" ] || REPO=/u/szaman5/DistNNP/DistNNP
[ -d "$REPO/distnnp_model" ] || { echo "ERROR: can't find the repo -- run 'sbatch' from the DistNNP/ dir or set MNV_REPO"; exit 1; }
VENV="${MNV_VENV:-$REPO/.venv-uma}"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"
[ -x "$PY" ] || { echo "ERROR: venv python not found at $PY -- set MNV_VENV"; exit 1; }

cd "$REPO"
mkdir -p logs scripts/_mnv_runs
WORKDIR=$(mktemp -d "$REPO/scripts/_mnv_runs/${SLURM_JOB_ID}_XXXX")   # shared FS (required)

echo "=== nodes:    $SLURM_JOB_NODELIST"
echo "=== workdir:  $WORKDIR"
echo "=== python:   $($PY --version)  torch $($PY -c 'import torch;print(torch.__version__)')"
nvidia-smi -L

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT="${MNV_MASTER_PORT:-29517}"
export NCCL_DEBUG=WARN
_IFNAME="${MNV_NCCL_SOCKET_IFNAME-hsn}"   # single-dash: honour an explicit empty value
[ -n "$_IFNAME" ] && export NCCL_SOCKET_IFNAME="$_IFNAME"
export HF_HUB_OFFLINE=1
RDZV="--rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=${SLURM_JOB_ID}"

run_dist () {   # $@ = script + args, launched 2 node / 4 rank under torchrun
    srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
        "$TORCHRUN" --nnodes="$SLURM_NNODES" --nproc-per-node=2 $RDZV "$@"
}

echo
echo "############ Stage 0: cross-node transport smoke (2 node / 4 rank) ############"
run_dist distnnp_model/tests/mn_transport_smoke.py
STAGE0=$?
echo "Stage 0 exit: $STAGE0"
[ $STAGE0 -ne 0 ] && { echo "Stage 0 failed -- cross-node transport broken. Stopping."; exit 1; }

echo
echo "############ Stage 1: exchanger pytest (this node, 2 GPU) ############"
"$PY" -m pytest -x -q distnnp_model/tests/test_python_nccl_exchanger.py
echo "Stage 1 exit: $?"

echo
echo "############ Stage 2: distributed MD, --exchanger nccl (2 node / 4 rank) ############"
run_dist distnnp_model/tests/run_distributed_trajectory.py --path "$WORKDIR" --exchanger nccl
STAGE2=$?
echo "Stage 2 exit: $STAGE2"

echo
echo "############ Stage 3: single-GPU reference run (1 node / 1 rank) ############"
# A world_size=1 baseline for the same system + integrator, for a reference
# throughput number and to prove the single-process path stands up. No
# trajectory parity compare -- exp0 is throughput-only (correctness lives in
# distnnp_model/tests/). Own workdir so it does not clobber Stage 2's dump.
REFDIR="${WORKDIR}_ref"
mkdir -p "$REFDIR"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=1 \
        distnnp_model/tests/run_distributed_trajectory.py --path "$REFDIR" --exchanger nccl
echo "Stage 3 exit: $?  (artifacts in $REFDIR)"

echo
echo "=== done. artifacts in $WORKDIR ==="
