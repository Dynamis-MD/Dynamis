#!/usr/bin/env bash
#
# End-to-end DistNNP install for this machine (aarch64 / GB200, driver CUDA 13,
# no system CUDA toolkit, no conda).
#
#   1. minimal CUDA 13.0 toolkit from NVIDIA redist tarballs  -> $CUDA_HOME
#   2. uv venv (Python 3.12)                                  -> DistNNP/.venv
#   3. Torch 2.13.0 + cu130 (aarch64 wheel)
#   4. cmake + ninja (build tools, not on PATH otherwise)
#   5. all four sub-packages in editable mode via requirements-dev.txt
#
# Torch is 2.13.0 (not the latest 2.13.x-only concern): it's the lowest torch
# with a cu130 aarch64 wheel that fairchem-core (distnnp_model[all]) can also
# satisfy -- fairchem-core >=2.22 needs torch >=2.13,<2.14, and every 2.x below
# that needs torch <2.9 (no cu130 build exists). torch 2.9-2.12 force
# fairchem-core down to the unusable 1.3.0.
#
# Re-runnable: each step no-ops if already satisfied.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INNER="$REPO_ROOT/DistNNP"                       # dir with requirements-dev.txt
export DISTNNP_VENV="${DISTNNP_VENV:-$INNER/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

# --- 1. CUDA toolkit -------------------------------------------------------
"$SCRIPT_DIR/install_cuda_redist.sh"

# --- 2. venv -------------------------------------------------------------
if [ ! -f "$DISTNNP_VENV/bin/activate" ]; then
    echo ">> Creating uv venv (Python 3.12): $DISTNNP_VENV"
    uv venv --python 3.12 "$DISTNNP_VENV"
fi

# CUDA_HOME / PATH / venv activation / CMAKE_ARGS
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

# --- 3. Torch (must exist before the sub-package build: their CMake does
#        find_package(Torch), and requirements-dev.txt imports torch) -----
if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.split('+')[0]=='$TORCH_VERSION' else 1)" 2>/dev/null; then
    echo ">> Installing torch==$TORCH_VERSION from $TORCH_INDEX"
    uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi

# --- 3b. Torch's CUDA math libraries ------------------------------------
# Torch's ATen CUDA headers #include <cublas_v2.h>, <cusparse.h>, <cusolverDn.h>,
# <cufft.h>, <curand.h> -- none of which are in the minimal toolkit. The exact
# matching versions ship inside the torch cu130 wheel under nvidia/cu13/; link
# them into $CUDA_HOME so the toolkit is a complete, standard layout.
TORCH_NVIDIA="$(python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2>/dev/null || true)"
if [ -d "$TORCH_NVIDIA/include" ]; then
    echo ">> Linking torch's CUDA math libs into $CUDA_HOME"
    for f in "$TORCH_NVIDIA"/include/*; do
        t="$CUDA_HOME/include/$(basename "$f")"
        [ -e "$t" ] || ln -s "$f" "$t"
    done
    for f in "$TORCH_NVIDIA"/lib/lib{cublas,cublasLt,cusparse,cusolver,cusolverMg,cufft,cufftw,curand}.so*; do
        [ -e "$f" ] || continue
        t="$CUDA_HOME/lib64/$(basename "$f")"
        [ -e "$t" ] || ln -s "$f" "$t"
    done
fi

# --- 4. build tools ------------------------------------------------------
python -c 'import shutil,sys; sys.exit(0 if shutil.which("cmake") and shutil.which("ninja") else 1)' \
    || uv pip install cmake ninja

# re-source so CMAKE_PREFIX_PATH now includes Torch's cmake dir
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

# --- 5. sub-packages ---------------------------------------------------
# Pin the already-installed cu130 torch so the resolver doesn't try to pull
# a plain-PyPI torch over it while resolving requirements-dev.txt.
constraint="$(mktemp)"
trap 'rm -f "$constraint"' EXIT
python - >"$constraint" <<'PY'
import torch
print(f"torch=={torch.__version__}")
PY
echo ">> Constraint: $(cat "$constraint")"

# scikit-build-core keeps a persistent CMake build dir per sub-package and does
# NOT reconfigure when only the installed torch changes -- a stale dir silently
# reinstalls a .so built against the old torch ABI (undefined-symbol at import).
# Wipe them so each full setup run is a clean build. (For day-to-day incremental
# rebuilds after a C++/CUDA edit, use `uv pip install -e ./distnnp_state` etc.)
echo ">> Clearing stale sub-package build dirs"
rm -rf "$INNER"/distnnp_state/skbuild "$INNER"/distnnp_model/skbuild \
       "$INNER"/distnnp_integrator/build "$INNER"/distnnp/skbuild

echo ">> Building + installing sub-packages (editable). First run pulls"
echo "   fairchem-core and its deps -- this is the slow step (10-20 min)."
( cd "$INNER" && uv pip install -r requirements-dev.txt --constraint "$constraint" --reinstall-package distnnp-state --reinstall-package distnnp-model --reinstall-package distnnp-integrator --reinstall-package distnnp )

# --- verify ----------------------------------------------------------
echo
echo ">> Verifying imports..."
python - <<'PY'
import distnnp_state, distnnp_integrator, distnnp_model, distnnp
import torch
print("torch            :", torch.__version__, "| CUDA", torch.version.cuda)
print("CUDA available   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0         :", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
print("distnnp packages : OK")
PY

echo
echo ">> Install complete. New shells: 'source scripts/env.sh' before use."
