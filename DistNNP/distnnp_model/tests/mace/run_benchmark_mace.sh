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
#SBATCH --job-name=distmace-thin-static
#SBATCH --output=/u/jsaravananindira/DistNNP/DistNNP/results/thin_mace_benchmark_%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/u/jsaravananindira/DistNNP/DistNNP}
RESULT_DIR=${RESULT_DIR:-${PROJECT_ROOT}/results/distmace_thin_static_${SLURM_JOB_ID:-local}}
MODEL_FAMILY=${MODEL_FAMILY:-mp}
MODEL_NAME=${MODEL_NAME:-medium-mpa-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
ENABLE_BENCHMARK=${ENABLE_BENCHMARK:-0}
XYZ_PATH=${XYZ_PATH:-}

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/distnnp/src:${PROJECT_ROOT}/distnnp_model/src:${PROJECT_ROOT}/distnnp_state/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MASTER_PORT=${MASTER_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 40000)))}

REFERENCE_ARTIFACT="${RESULT_DIR}/standalone_reference.pt"
REFERENCE_ARGS=(--model-family "${MODEL_FAMILY}" --model "${MODEL_NAME}" --output "${REFERENCE_ARTIFACT}")
DRIVER_ARGS=(run-distributed --model-family "${MODEL_FAMILY}" --model "${MODEL_NAME}" --reference-artifact "${REFERENCE_ARTIFACT}" --strict-parity --output-json "${RESULT_DIR}/thin_summary.json")

if [[ -n "${XYZ_PATH}" ]]; then
  REFERENCE_ARGS+=(--xyz-path "${XYZ_PATH}")
  DRIVER_ARGS+=(--xyz-path "${XYZ_PATH}")
fi
if [[ "${ENABLE_BENCHMARK}" == "1" ]]; then
  REFERENCE_ARGS+=(--benchmark --benchmark-warmup 5 --benchmark-iterations 20 --benchmark-samples 7)
  DRIVER_ARGS+=(--benchmark --benchmark-warmup 5 --benchmark-iterations 20 --benchmark-samples 7)
fi

python distnnp_model/tests/mace/mace_standalone_reference.py "${REFERENCE_ARGS[@]}"
torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  distnnp_model/tests/mace/test_distmace_thin_adapter.py "${DRIVER_ARGS[@]}"

echo "Thin static validation completed: ${RESULT_DIR}"
