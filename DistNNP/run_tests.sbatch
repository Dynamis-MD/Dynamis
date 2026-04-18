#!/bin/bash
#SBATCH --job-name=distnnp-tests
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j.log

# ---------------------------------------------------------------------------
# run_tests.sbatch — SLURM launcher for the DistNNP distributed test suite.
#
# Usage:
#   sbatch run_tests.sbatch                        # full suite (pytest.ini testpaths)
#   sbatch run_tests.sbatch distnnp/tests/         # distnnp integration tests only
#   sbatch run_tests.sbatch distnnp/tests/ -m "not perf"  # skip performance tests
#
# Any arguments after the script name are forwarded verbatim to pytest.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Timestamped log (SLURM --output captures early boot; real detail goes here)
# ---------------------------------------------------------------------------
mkdir -p logs
LOG="logs/test_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "===== DistNNP Test Run ====="
echo "Job     : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Date    : $(date -u)"
echo "==========================="

# ---------------------------------------------------------------------------
# CUDA_HOME — required for C++ extensions. Try common cluster paths.
# Override by setting CUDA_HOME before calling sbatch.
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
    for candidate in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 /opt/cuda; do
        if [[ -d "${candidate}" ]]; then
            export CUDA_HOME="${candidate}"
            break
        fi
    done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "CUDA_HOME: ${CUDA_HOME}"
else
    echo "WARNING: CUDA_HOME not found; C++ extensions may fail to load"
fi

# ---------------------------------------------------------------------------
# NCCL — intra-node NVLink + inter-node InfiniBand (leave IB enabled)
# ---------------------------------------------------------------------------
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Rendezvous: pick the first node as master, choose a free port.
# srun launches one torchrun per node; torchrun spawns nproc_per_node
# processes on that node. Together they form a world of nnodes*nproc ranks.
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

echo "Master  : ${MASTER_ADDR}:${MASTER_PORT}"
echo "World   : ${SLURM_NNODES} nodes × ${SLURM_NTASKS_PER_NODE} GPUs = $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) ranks"

# Generate reference fixtures if they don't already exist.
if ! ls distnnp/tests/fixtures/lj_reference_rank*.pt 1>/dev/null 2>&1; then
    echo "Reference fixtures not found, generating..."
    srun bash -c "
        torchrun \
            --nnodes=${SLURM_NNODES} \
            --nproc_per_node=${SLURM_NTASKS_PER_NODE} \
            --node_rank=\${SLURM_NODEID} \
            --rdzv_id=${SLURM_JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            -m distnnp.tests.fixtures.generate_reference
    "
fi

# srun runs this block on every node in the allocation in parallel.
# Each node gets its own NODE_RANK from $SLURM_NODEID.
srun bash -c "
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=${SLURM_NTASKS_PER_NODE} \
        --node_rank=\${SLURM_NODEID} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        -m pytest \
        -v \
        --tb=short \
        $(printf '%q ' "$@")
"

EXIT_CODE=$?
echo ""
echo "===== Finished (exit ${EXIT_CODE}) ====="
exit "${EXIT_CODE}"
