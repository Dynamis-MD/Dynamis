#!/bin/bash
# ---------------------------------------------------------------------------
# exp0 MULTI-NODE FEASIBILITY: DistNNP-UMA distributed MD on the 9GMO solvated
# complex (9gmo_density.xyz, 1,536,442 atoms) across MULTIPLE NODES.
#
# SCOPE: feasibility only. Prove the run stands up across nodes and produces
# a stable steps/s, then STOP. Not a scientific production run -- short step
# windows, <=15 min walltime, hard-timeout well inside that.
#
# This is the first exp0 cell that leaves a single node, so it does NOT go
# through scripts/run_cell.sbatch (torchrun --standalone, one node only).
# It launches one torchrun per node under srun with a c10d rendezvous, and
# runs the DistNNP driver with the multi-node halo exchanger:
#
#     run_distnnp_uma.py --exchanger nccl   (torch.distributed all_to_all;
#     GPUDirect/CUDA-IPC has no cross-node peer and would hang the watchdog)
#
# PREREQUISITE: run the 2-node de-risk ladder first --
#     sbatch -p gpuA100x4 -A bcpt-delta-gpu \
#         ../../DistNNP/scripts/run_multinode_nccl_exchanger.sbatch
# (Stage 0 cross-node transport smoke + Stage 2 small distributed MD.)
#
# ---------------------------------------------------------------------------
# Submit (the #SBATCH block below already sets nodes/gpus/time; -p and -A are
# the only flags you must add):
#
#     cd experiments/exp0_end_to_end
#     sbatch -p gpuA100x4 -A bcpt-delta-gpu scripts/hero_9gmo.sbatch
#
#   smaller/larger:  sbatch --nodes=12 ... scripts/hero_9gmo.sbatch
#
# Kernel config = RESTART.md's "k0": TF32 on + activation-checkpointing on +
# SFC-Morton partitioner. This is the memory-safe reference config that reached
# ~32k atoms/GPU in the archived matrix (checkpointing is what buys the
# footprint; the Triton fast backend needs it OFF and does not fit at exp0
# sizes on 40GB A100s anyway). SFC-Morton is materially faster to build than
# kdtree / hilbert and the 9GMO box is close enough to cubic for it.
#
# Override via env:
#     EXP0_HERO_XYZ=exp_datasets/protein_solvated/9gmo_density.xyz  (default)
#     EXP0_HERO_TAG=feas_9gmo_64gpu                                 (result stem suffix)
#     EXP0_HERO_PARTITION=sfc_morton     (sfc_hilbert / kdtree / simple)
#     EXP0_HERO_TF32=1                   (1 = TF32 GEMMs; 0 = full FP32)
#     EXP0_HERO_ACKPT=1                  (1 = activation checkpointing on -- keep on to fit)
#     EXP0_HERO_INFERENCE=default        (base preset the tf32/ackpt overrides sit on top of)
#     EXP0_HERO_EXTRA="--warmup-steps 3 --timed-steps 10"           (default; keep it short)
#     EXP0_HERO_HARD_TIMEOUT=720         (seconds; must stay < the 900 s walltime)
#
# ---------------------------------------------------------------------------
# SIZING (TF32 + ackpt-on UMA-s, A100-40GB). The archived matrix reached
# ~32k atoms/GPU in this config. 9GMO is 1.54M atoms:
#
#     atoms/GPU     GPUs   nodes (A100x4)
#     ~24k          64     16     <- default here (headroom over the ~32k ceiling)
#     ~32k          48     12     <- at the ceiling, higher OOM risk
#     ~48k          32      8     <- expect status=oom (still a data point)
#
# status=oom on this cell is DATA (the multi-node OOM ceiling), not a failure
# -- the driver writes the record either way.
# ---------------------------------------------------------------------------
#SBATCH --job-name=exp0-feas-9gmo
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/hero_9gmo_%j.out
#SBATCH --error=results/logs/hero_9gmo_%j.err

set -uo pipefail

_S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_S/_lib.sh"

# partition + account must come from the submit line (Slurm ignores #SBATCH
# after the first command). See the header: sbatch -p gpuA100x4 -A bcpt-delta-gpu
: "${SLURM_JOB_PARTITION:?submit with  -p gpuA100x4 -A bcpt-delta-gpu}"

exp0_check_venv
exp0_load_modules
exp0_export_runtime

XYZ="${EXP0_HERO_XYZ:-$EXP_DIR/exp_datasets/protein_solvated/9gmo_density.xyz}"
[ -f "$XYZ" ] || { echo "ERROR: system not found: $XYZ"; echo "  run scripts/setup_datasets.sh (Tier 4) first"; exit 1; }

RESULTS="$RESULTS_DIR"
mkdir -p "$RESULTS/logs"

NNODES="${SLURM_NNODES:?}"
GPN="${SLURM_GPUS_PER_NODE:-4}"; GPN="${GPN##*:}"
WORLD=$(( NNODES * GPN ))
NUM_ATOMS="$(head -n1 "$XYZ" | tr -d '[:space:]')"
TAG="${EXP0_HERO_TAG:-feas_9gmo_${WORLD}gpu}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${EXP0_HERO_PORT:-29518}"
RDZV="--rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=${SLURM_JOB_ID}"

# NCCL over the high-speed fabric + libfabric/OFI plugin (loaded via EXP0_MODULES
# = aws-ofi-nccl). exp0_export_runtime already set NCCL_SOCKET_IFNAME=hsn.
export NCCL_DEBUG="${EXP0_NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INFERENCE="${EXP0_HERO_INFERENCE:-default}"
PARTITION="${EXP0_HERO_PARTITION:-sfc_morton}"
TF32="${EXP0_HERO_TF32:-1}"
ACKPT="${EXP0_HERO_ACKPT:-1}"

# Short step windows -- feasibility, not science: 3 warmup + 10 timed is enough
# for a stable steps/s. build-timeout bumped from the 420 s default: cold model
# load + a partition of 1.5M atoms across ~64 ranks legitimately takes longer
# than the synthetic matrix cells it was tuned for. Caller EXP0_HERO_EXTRA is
# appended last, so it wins on any repeated flag.
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
echo "   partition   : $SLURM_JOB_PARTITION   account: ${SLURM_JOB_ACCOUNT:-?}"
echo "   venv        : $VENV"
echo "   kernel cfg  : $INFERENCE + tf32=$TF32 + ackpt=$ACKPT   partitioner: $PARTITION"
echo "   exchanger   : nccl (torch.distributed all_to_all)"
echo "   result      : $RESULTS/distnnp-uma_${NUM_ATOMS}atoms_${WORLD}gpu_${TAG}.json"
echo "=========================================================================="
"$PY" --version
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c 'echo "  $(hostname): $(nvidia-smi -L | wc -l) GPU"'

# One torchrun per node; each spawns $GPN local ranks.
# Hard timeout stays well inside the 900 s walltime: a dead rank -> 600 s NCCL
# collective timeout, and `timeout` must reap the tree before Slurm SIGKILLs
# the step (which would leave no result JSON).
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
