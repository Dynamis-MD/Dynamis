#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:10:00
#SBATCH --job-name=distmace-thin-trajectory
#SBATCH --output=/u/jsaravananindira/DistNNP/DistNNP/results/thin_mace_trajectory_%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/u/jsaravananindira/DistNNP/DistNNP}
RESULT_DIR=${RESULT_DIR:-${PROJECT_ROOT}/results/distmace_thin_trajectory_${SLURM_JOB_ID:-local}}
MODEL_FAMILY=${MODEL_FAMILY:-mp}
MODEL_NAME=${MODEL_NAME:-medium-mpa-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

if [[ "${NPROC_PER_NODE}" != "2" ]]; then
  echo "The standardized trajectory acceptance workflow requires exactly two ranks." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/distnnp/src:${PROJECT_ROOT}/distnnp_model/src:${PROJECT_ROOT}/distnnp_state/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MASTER_PORT=${MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 40000)))}

torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  --nproc-per-node=2 \
  distnnp_model/tests/mace/run_distributed_trajectory.py \
  --path "${RESULT_DIR}" --model-family "${MODEL_FAMILY}" --model "${MODEL_NAME}"

python distnnp_model/tests/mace/test_md.py \
  --path "${RESULT_DIR}" --model-family "${MODEL_FAMILY}" --model "${MODEL_NAME}" \
  --reference-mode monolithic

echo "Thin trajectory validation completed: ${RESULT_DIR}"
