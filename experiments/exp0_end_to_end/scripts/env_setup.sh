#!/bin/bash
# ---------------------------------------------------------------------------
# Build the `.venv-uma` environment for the exp0 end-to-end benchmark.
#
#   scripts/sbatch_exp0.sh --cpu scripts/env_setup.sh
#
# Runs on the CPU partition -- nvcc cross-compiles the CUDA kernels for
# EXP0_CUDA_ARCH without a GPU present (~15-25 min). Only the driver smoke
# needs a GPU afterwards.
#
# Mirrors the repo's proven build (HOW_TO_RUN.md): uv venv on the configured
# python, torch pinned + cu128, then `requirements-dev.txt` which builds all
# 4 distnnp extensions AND pulls fairchem-core via `distnnp_model[all]`.
# Deviations for this benchmark:
#   * torch EXP0_TORCH_VERSION (2.8.0) so fairchem-core resolves to a 2.1x
#     release that supports it;
#   * `ray` added explicitly (graph-parallel FairChem baseline);
#   * CUDA arch pinned to EXP0_CUDA_ARCH (Volta dropped, see CMakeLists edits).
#
# Requires `uv` (set EXP0_UV if not on PATH). HF auth: accept the facebook/UMA
# license and `huggingface-cli login` once; step 7 pre-warms the checkpoint.
#SBATCH --job-name=exp0-env-setup
#SBATCH --time=01:00:00

set -euo pipefail

# Slurm runs this from a spool copy, so ${BASH_SOURCE[0]} is not scripts/.
# scripts/sbatch_exp0.sh exports EXP0_SCRIPTS_DIR; fall back to $0's dir when
# the script is run directly (non-Slurm testing).
_S="${EXP0_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$_S/_lib.sh" ] || { echo "ERROR: scripts/_lib.sh not found in '$_S' -- submit via scripts/sbatch_exp0.sh"; exit 1; }
source "$_S/_lib.sh"

UV="${EXP0_UV:-uv}"
TORCH_VERSION="${EXP0_TORCH_VERSION:-2.8.0}"
CUDA_ARCH="${EXP0_CUDA_ARCH:-80;90}"

mkdir -p "$RESULTS_DIR/logs"

echo "=== [1/8] modules ==="
exp0_load_modules
command -v "$UV" >/dev/null || { echo "uv not found ('$UV'); set EXP0_UV"; exit 1; }

echo "=== [2/8] fresh venv: $VENV ==="
rm -rf "$VENV"
"$UV" venv --python "$(command -v python3)" "$VENV"
PYBIN="$VENV/bin/python"
PIP=("$UV" pip install --python "$PYBIN")

echo "=== [3/8] torch $TORCH_VERSION + cu128 + build backend ==="
"${PIP[@]}" "torch==${TORCH_VERSION}" torchvision --index-url https://download.pytorch.org/whl/cu128
# Build deps live IN the target venv so step 6 can use --no-build-isolation:
# that forces CMake's find_package(Torch) to resolve THIS venv's torch instead
# of a sibling .venv (ABI clash -> "sym_node has no attribute 'DynamicInt'").
"${PIP[@]}" ninja "scikit-build-core>=0.5" cmake pybind11 setuptools wheel
"$PYBIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== [4/8] constraints ==="
CONSTRAINTS="$(mktemp)"
cat > "$CONSTRAINTS" <<EOF
torch==${TORCH_VERSION}+cu128
fairchem-core<2.22
EOF
cat "$CONSTRAINTS"

echo "=== [5/8] HF auth sanity (checkpoint pre-warm is step 7) ==="
"$PYBIN" - <<'PY' || echo "  (huggingface_hub not yet present; fine)"
import os
print("HF token present:", os.path.exists(os.path.expanduser("~/.cache/huggingface/token")))
PY

echo "=== [6/8] requirements-dev.txt (+ ray) -> builds 4 extensions, pulls fairchem-core ==="
grep -H CUDA_ARCHITECTURES "$MONO/distnnp_state/CMakeLists.txt" "$MONO/distnnp_model/CMakeLists.txt" || true

# Nuke stale per-package build trees: a leftover CMakeCache.txt from an earlier
# build against a different venv pins Torch_DIR and silently wins.
for pkg in distnnp_state distnnp_model distnnp_integrator distnnp; do
    rm -rf "$MONO/$pkg/skbuild" "$MONO/$pkg/build" "$MONO/$pkg/_skbuild"
done
rm -rf "${SCIKIT_BUILD_CORE_CACHE:-$HOME/.cache/scikit-build-core}"
"$UV" cache clean distnnp distnnp-state distnnp-model distnnp-integrator 2>/dev/null || true

TORCH_CMAKE="$("$PYBIN" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake"))')"
export CMAKE_PREFIX_PATH="$TORCH_CMAKE:${CMAKE_PREFIX_PATH:-}"
export Torch_DIR="$TORCH_CMAKE/Torch"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DCMAKE_PREFIX_PATH=$TORCH_CMAKE"
export CUDA_HOME="${CUDA_HOME:-${CUDATOOLKIT_HOME:-}}"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-16}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
cd "$MONO"
"$UV" pip install --python "$PYBIN" --no-build-isolation --no-cache \
    --reinstall-package distnnp --reinstall-package distnnp-state \
    --reinstall-package distnnp-model --reinstall-package distnnp-integrator \
    -r requirements-dev.txt --constraint "$CONSTRAINTS"
"${PIP[@]}" --constraint "$CONSTRAINTS" ray
rm -f "$CONSTRAINTS"

echo "--- verify every distnnp extension links THIS venv's torch ---"
BAD=0
for so in "$VENV"/lib/python*/site-packages/distnnp_*/*_impl*.so; do
    [ -e "$so" ] || continue
    if ldd "$so" | grep -qE '/\.venv/lib|/\.venv/'; then
        echo "  WRONG libtorch: $so"; ldd "$so" | grep -i 'libtorch\|libc10'; BAD=1
    else
        echo "  ok: $(basename "$so")"
    fi
done
[ "$BAD" -eq 0 ] || { echo "extension(s) linked the wrong libtorch -- aborting"; exit 1; }

echo "=== [7/8] import smoke-check + checkpoint pre-warm ==="
"$PYBIN" -c "import distnnp_state, distnnp_integrator, distnnp_model, distnnp; from distnnp_model.nnp import UMAModel; print('distnnp ok')"
"$PYBIN" -c "import fairchem.core, ray; from fairchem.core.calculate.pretrained_mlip import get_predict_unit; print('fairchem', fairchem.core.__version__, '| ray', ray.__version__)"
# Pre-warm uma-s-1p2 into fairchem's OWN cache (~/.cache/fairchem) so benchmark
# jobs run HF_HUB_OFFLINE=1 (no download stampede on compute nodes).
"$PYBIN" - <<'PY'
from distnnp_model.nnp import UMAModel
UMAModel("uma-s-1p2", dataset="omol", device="cpu")
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_turbo
s = inference_settings_turbo(); s.merge_mole = True; s.compile = False
get_predict_unit("uma-s-1p2", device="cpu", inference_settings=s, workers=1)
print("uma-s-1p2 fully warmed into ~/.cache/fairchem")
PY

echo "=== [8/8] freeze resolved versions ==="
NO_COLOR=1 "$UV" pip freeze --python "$PYBIN" 2>/dev/null \
    | sed $'s/\x1b\\[[0-9;]*m//g' \
    | grep -Ei '^(fairchem-core|torch|torchvision|ray|e3nn|numpy|ase|numba|triton|scipy|huggingface-hub)==' \
    | tee "$EXP_DIR/resolved_versions.txt" || true

echo
echo "DONE. venv: $VENV"
echo "Next -- driver smoke:  scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke.sbatch"
