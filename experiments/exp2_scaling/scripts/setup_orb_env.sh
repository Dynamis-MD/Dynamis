#!/usr/bin/env bash
# Build the Dynamis + ORB Python 3.12 uv venv at <repo>/.venv on GB200 (aarch64, CUDA 13.0)
# and pre-warm ORB v3 OMAT checkpoints into ~/.cache/cached_path so compute nodes
# run completely offline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_lib.sh"

exp2_export_cuda
export CACHED_PATH_CACHE_ROOT="${EXP2_CACHED_PATH_CACHE_ROOT:-$HOME/.cache/cached_path}"

if ! command -v "$EXP2_UV" >/dev/null 2>&1; then
    echo "ERROR: '$EXP2_UV' not found on PATH" >&2
    exit 1
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc (set EXP2_CUDA_HOME in cluster.local.env)" >&2
    exit 1
fi

echo "=== setup_orb_env: Dynamis + ORB on GB200 ==="
echo "  REPO_ROOT : $REPO_ROOT"
echo "  VENV      : $VENV"
echo "  CUDA_HOME : $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -n1))"
echo "  ARCH_LIST : $TORCH_CUDA_ARCH_LIST"

# 1. Create Python 3.12 venv (orb-models==0.7.0 pins dm-tree==0.1.8, needs py3.12)
if [ ! -f "$VENV/bin/activate" ]; then
    echo ">> Creating uv venv (Python $EXP2_PYTHON): $VENV"
    "$EXP2_UV" venv --python "$EXP2_PYTHON" "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# 2. Install torch+cu130 first (C++/CUDA sub-packages link against libtorch)
if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.split('+')[0]=='$EXP2_TORCH_VERSION' else 1)" 2>/dev/null; then
    echo ">> Installing torch==$EXP2_TORCH_VERSION from $EXP2_TORCH_INDEX"
    "$EXP2_UV" pip install "torch==$EXP2_TORCH_VERSION" --index-url "$EXP2_TORCH_INDEX"
fi

# 3. Ensure CUDA math headers/libs from torch wheel are visible in CUDA_HOME if needed
TORCH_NVIDIA="$(python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2>/dev/null || true)"
if [ -d "$TORCH_NVIDIA/include" ]; then
    for f in "$TORCH_NVIDIA"/include/*; do
        t="$CUDA_HOME/include/$(basename "$f")"
        [ -e "$t" ] || ln -s "$f" "$t" 2>/dev/null || true
    done
    for f in "$TORCH_NVIDIA"/lib/lib{cublas,cublasLt,cusparse,cusolver,cusolverMg,cufft,cufftw,curand}.so*; do
        [ -e "$f" ] || continue
        t="$CUDA_HOME/lib64/$(basename "$f")"
        [ -e "$t" ] || ln -s "$f" "$t" 2>/dev/null || true
    done
fi

# 4. Build + test tools (post-analysis & plotting live in Dynamis-experiments-log)
echo ">> Installing build/test dependencies"
"$EXP2_UV" pip install cmake ninja scikit-build-core pybind11 pytest scipy

# Expose Torch's CMake config
TORCH_CMAKE="$(python -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"share","cmake"))')"
export CMAKE_PREFIX_PATH="$TORCH_CMAKE${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# 5. Pin torch so orb-models doesn't overwrite the cu130 wheel with CPU/cu12x PyPI torch
CONSTRAINT="$(mktemp)"
trap 'rm -f "$CONSTRAINT"' EXIT
python -c 'import torch; print(f"torch=={torch.__version__}")' > "$CONSTRAINT"
echo ">> Pinning $(cat "$CONSTRAINT")"

# 6. Install orb-models + editable Dynamis sub-packages
# NOTE: dynamis_state MUST be installed before dynamis_integrator and dynamis_model
# because their CMakeLists locate installed dynamis_state headers.
echo ">> Installing orb-models==0.7.0 and Dynamis sub-packages (editable, --no-build-isolation)"
rm -rf "$REPO_ROOT/dynamis_state/skbuild" "$REPO_ROOT/dynamis_model/skbuild" \
       "$REPO_ROOT/dynamis_integrator/build" "$REPO_ROOT/dynamis/skbuild"

"$EXP2_UV" pip install --constraint "$CONSTRAINT" "orb-models==0.7.0"
(
    cd "$REPO_ROOT"
    "$EXP2_UV" pip install --no-build-isolation --constraint "$CONSTRAINT" -e ./dynamis_state
    "$EXP2_UV" pip install --no-build-isolation --constraint "$CONSTRAINT" -e ./dynamis_integrator -e "./dynamis_model[orb]" -e ./dynamis
)

# 7. Pre-warm ORB checkpoints into CACHED_PATH_CACHE_ROOT so compute nodes need no internet
exp2_export_runtime
mkdir -p "$CACHED_PATH_CACHE_ROOT"
echo ">> Pre-warming ORB checkpoints into $CACHED_PATH_CACHE_ROOT"
python - <<'PY'
from orb_models.forcefield import pretrained
for name in (
    "orb_v3_direct_inf_omat",
    "orb_v3_direct_20_omat",
    "orb_v3_conservative_inf_omat",
    "orb_v3_conservative_20_omat",
):
    print(f"  caching {name} ...", flush=True)
    fn = getattr(pretrained, name)
    model, adapter = fn(device="cpu", precision="float32-high", compile=False)
    print(f"    OK: radius={adapter.radius}, max_num_neighbors={adapter.max_num_neighbors}")
PY

# 8. Verify imports
echo ">> Verifying Dynamis + ORB imports..."
python - <<'PY'
import torch
import dynamis, dynamis_state, dynamis_integrator, dynamis_model
from dynamis_model.nnp.orb import ORBModel
from dynamis_model.comm.exchanger import GPUDirectExchanger, PythonNCCLExchanger
import dynamis_state.dynamis_state_impl as _impl
assert hasattr(_impl, "radius_graph_pbc_select"), "missing radius_graph_pbc_select"
assert hasattr(_impl, "extract_halo_info"), "missing extract_halo_info"
print(f"torch={torch.__version__} cuda={torch.version.cuda} — ALL IMPORTS OK")
PY
echo "=== setup_orb_env complete ==="
