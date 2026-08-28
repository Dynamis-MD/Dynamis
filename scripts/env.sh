# Source this at the start of every session (and inside sbatch scripts)
# before building or running DistNNP on this machine (aarch64 / GB200).
#
#   source scripts/env.sh
#
# Override CUDA_HOME / DISTNNP_VENV before sourcing to point elsewhere.

_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_repo_root="$(cd "$_env_dir/.." && pwd)"

export CUDA_HOME="${CUDA_HOME:-$HOME/cuda/13.0.3}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# GB200 is Blackwell (sm_100); also keep Ampere/Hopper for portability.
# Torch's CMake ignores CMAKE_CUDA_ARCHITECTURES and drives nvcc -gencode from
# TORCH_CUDA_ARCH_LIST instead -- and its autodetect default still includes
# sm_50, which CUDA 13 rejects ("Unsupported gpu architecture 'compute_50'").
# Setting this pins the arch list for every sub-package's CUDA compile.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 9.0 10.0}"

DISTNNP_VENV="${DISTNNP_VENV:-$_repo_root/DistNNP/.venv}"
if [ -f "$DISTNNP_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$DISTNNP_VENV/bin/activate"
    # requirements-dev.txt computes CMAKE_PREFIX_PATH by importing torch;
    # make sure CMake can find Torch's config package during rebuilds too.
    _torch_cmake="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake"))' 2>/dev/null || true)"
    [ -n "$_torch_cmake" ] && export CMAKE_PREFIX_PATH="$_torch_cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    unset _torch_cmake
fi

unset _env_dir _repo_root
