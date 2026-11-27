# shellcheck shell=bash
# Common bootstrap for exp2_scaling scripts (model-agnostic).

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
REPO_ROOT="${EXP2_REPO_ROOT:-$(cd "$EXP_DIR/../.." && pwd)}"

# shellcheck source=/dev/null
. "$EXP_DIR/cluster.env"

# Default dataset root points to the shared Zenodo directory populated by:
#   bash experiments/exp_datasets/scripts/download_zenodo.sh <dataset>
EXP2_DATASETS_ROOT="${EXP2_DATASETS_ROOT:-$REPO_ROOT/experiments/exp_datasets/zenodo}"
DATA_DIR="${EXP2_DATA_DIR:-$EXP2_DATASETS_ROOT/ch4_o2}"
RESULTS_DIR="${EXP2_RESULTS_DIR:-$EXP_DIR/results}"
VENV="${EXP2_VENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"

exp2_export_cuda() {
    if [ -d "$EXP2_CUDA_HOME" ]; then
        export CUDA_HOME="$EXP2_CUDA_HOME"
        export CUDAToolkit_ROOT="$EXP2_CUDA_HOME"
        case ":$PATH:" in
            *":$CUDA_HOME/bin:"*) ;;
            *) export PATH="$CUDA_HOME/bin:$PATH" ;;
        esac
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    export TORCH_CUDA_ARCH_LIST="$EXP2_TORCH_CUDA_ARCH_LIST"
}

exp2_export_runtime() {
    exp2_export_cuda
    [ -n "${EXP2_NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME="$EXP2_NCCL_SOCKET_IFNAME"
    [ -n "${EXP2_GLOO_SOCKET_IFNAME:-}" ] && export GLOO_SOCKET_IFNAME="$EXP2_GLOO_SOCKET_IFNAME"
    [ -n "${EXP2_NCCL_IB_HCA:-}" ] && export NCCL_IB_HCA="$EXP2_NCCL_IB_HCA"
    export NCCL_MNNVL_ENABLE="${EXP2_NCCL_MNNVL_ENABLE:-0}"
    export NCCL_DEBUG="${EXP2_NCCL_DEBUG:-WARN}"
    export OMP_NUM_THREADS="${EXP2_OMP_NUM_THREADS:-8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

exp2_check_venv() {
    if [ ! -x "$PY" ] || [ ! -x "$TORCHRUN" ]; then
        echo "ERROR: Dynamis venv not found at $VENV" >&2
        echo "       Set EXP2_VENV in cluster.local.env or run your model's setup script." >&2
        exit 1
    fi
}

exp2_require_dataset() {
    local dataset_name="$1"
    local target_dir="${2:-$EXP2_DATASETS_ROOT/$dataset_name}"
    if [ ! -d "$target_dir" ] || [ -z "$(find "$target_dir" -maxdepth 2 -name '*.xyz' -print -quit 2>/dev/null)" ]; then
        echo "ERROR: dataset '$dataset_name' not found in $target_dir" >&2
        echo "       Download it once with:" >&2
        echo "         bash $REPO_ROOT/experiments/exp_datasets/scripts/download_zenodo.sh $dataset_name" >&2
        exit 1
    fi
}
