#!/bin/bash
#SBATCH --job-name=distnnp-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

# ---------------------------------------------------------------------------
# run_tests.sbatch — SLURM launcher for the DistNNP distributed test suite.
#
# Usage:
#   sbatch --account=<your_allocation> run_tests.sbatch                        # default: distnnp/tests/
#   sbatch --account=<your_allocation> run_tests.sbatch -m "not perf"          # skip performance tests
#
# Defaults to distnnp/tests/ so that sub-package mp.spawn tests are not collected under torchrun.
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
# Python — resolve the interpreter that torchrun uses, then ensure all
# requirements (pytest, editable packages) are installed under it.
# This makes the script portable across systems with different envs.
# ---------------------------------------------------------------------------
TORCHRUN=$(command -v torchrun)
if [[ -z "${TORCHRUN}" ]]; then
    echo "ERROR: torchrun not found in PATH"
    exit 1
fi

# Extract the interpreter from torchrun's shebang line.
PYTHON=$(head -1 "${TORCHRUN}" | sed 's/^#!//' | awk '{print $1}')
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    PYTHON=$(command -v python3 || command -v python)
fi
echo "Python  : ${PYTHON} ($(${PYTHON} --version 2>&1))"

# Set CMAKE_PREFIX_PATH so scikit-build-core can find PyTorch's cmake config.
# PyTorch ships TorchConfig.cmake under torch.utils.cmake_prefix_path.
TORCH_CMAKE_PREFIX=$("${PYTHON}" -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || true)
if [[ -n "${TORCH_CMAKE_PREFIX}" ]]; then
    export CMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
else
    echo "WARNING: could not detect torch cmake prefix; C++ builds may fail"
fi

# Limit parallel compile jobs to avoid OOM during C++ builds in the job.
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

# Verify that requirements are pre-installed.
echo "Verifying environment..."
PACKAGES=("torch" "pytest" "distnnp_state" "distnnp_model" "distnnp_integrator" "distnnp")
for pkg in "${PACKAGES[@]}"; do
    if ! "${PYTHON}" -c "import ${pkg}" &>/dev/null; then
        echo "ERROR: Package '${pkg}' not found. Please run 'pip install -r requirements-dev.txt' manually."
        exit 1
    fi
done
echo "Environment OK."

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
echo "World   : ${SLURM_NNODES} nodes × ${SLURM_GPUS_PER_NODE} GPUs = $((SLURM_NNODES * SLURM_GPUS_PER_NODE)) ranks"

# Generate reference fixtures if they don't already exist.
FIXTURE_MISSING=false
for integrator in euler verlet langevin; do
    for r in 0 1; do
        [[ ! -f "distnnp/tests/fixtures/lj_reference_${integrator}_rank${r}.pt" ]] && FIXTURE_MISSING=true
    done
done

if $FIXTURE_MISSING; then
    echo "Reference fixtures not found, generating..."
    torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc_per_node=${SLURM_GPUS_PER_NODE} \
        --node_rank=${SLURM_NODEID} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        distnnp/tests/fixtures/generate_reference.py
    # Verify files were actually created (torchrun may exit non-zero even on success)
    for integrator in euler verlet langevin; do
        for r in 0 1; do
            if [[ ! -f "distnnp/tests/fixtures/lj_reference_${integrator}_rank${r}.pt" ]]; then
                echo "ERROR: fixture generation failed — missing lj_reference_${integrator}_rank${r}.pt"
                exit 1
            fi
        done
    done
fi

# srun runs this block on every node in the allocation in parallel.
# Each node gets its own NODE_RANK from $SLURM_NODEID.
# Default to distnnp/tests/ to avoid collecting mp.spawn sub-package tests under torchrun.
PYTEST_ARGS=("${@:-distnnp/tests/}")

torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --node_rank=${SLURM_NODEID} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    -m pytest \
    -v \
    --tb=short \
    $(printf '%q ' "${PYTEST_ARGS[@]}")

EXIT_CODE=$?
echo ""
echo "===== Finished (exit ${EXIT_CODE}) ====="
exit "${EXIT_CODE}"