# shellcheck shell=bash
# ---------------------------------------------------------------------------
# Common bootstrap for every exp0 script. Source it right after the #SBATCH
# block (or at the top of a plain shell script):
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#
# Provides:
#   REPO_ROOT  MONO  EXP_DIR  SCRIPTS_DIR  RESULTS_DIR   -- resolved from this
#              file's location, no hardcoded /u/... paths
#   VENV  PY  TORCHRUN                                    -- the exp0 venv
#   all EXP0_* vars                                       -- from cluster.env
#   exp0_load_modules      -- honour EXP0_MODULE_PURGE / _UNLOAD / EXP0_MODULES
#   exp0_export_runtime    -- NCCL_SOCKET_IFNAME, OMP_NUM_THREADS, HF_HUB_OFFLINE
# ---------------------------------------------------------------------------

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"

# shellcheck source=cluster.env
. "$SCRIPTS_DIR/cluster.env"

REPO_ROOT="${EXP0_REPO_ROOT:-$(cd "$EXP_DIR/../.." && pwd)}"
MONO="$REPO_ROOT/DistNNP"
RESULTS_DIR="$EXP_DIR/results"
VENV="${EXP0_VENV:-$MONO/.venv-uma}"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"

exp0_load_modules() {
    if ! command -v module >/dev/null 2>&1; then
        echo "[_lib] no 'module' command -- assuming python/toolchain already on PATH"
        return 0
    fi
    [ "${EXP0_MODULE_PURGE:-0}" = "1" ] && { echo "[_lib] module purge"; module purge 2>/dev/null || true; }
    for m in ${EXP0_MODULE_UNLOAD:-}; do
        module unload "$m" 2>/dev/null || true
    done
    if [ -n "${EXP0_MODULES:-}" ]; then
        echo "[_lib] module load ${EXP0_MODULES}"
        # shellcheck disable=SC2086
        module load ${EXP0_MODULES}
    fi
    module list 2>&1 || true
}

exp0_export_runtime() {
    if [ -n "${EXP0_NCCL_SOCKET_IFNAME:-}" ]; then
        export NCCL_SOCKET_IFNAME="${EXP0_NCCL_SOCKET_IFNAME}"
    fi
    export OMP_NUM_THREADS="${EXP0_OMP_NUM_THREADS:-8}"
    export HF_HUB_OFFLINE="${EXP0_HF_HUB_OFFLINE:-1}"
    [ -n "${EXP0_HF_HOME:-}" ] && export HF_HOME="${EXP0_HF_HOME}"
}

exp0_check_venv() {
    [ -x "$PY" ] || {
        echo "ERROR: venv python not found at $PY"
        echo "       build it first:  scripts/sbatch_exp0.sh --cpu scripts/env_setup.sh"
        echo "       (or set EXP0_VENV / cluster.local.env to point at an existing one)"
        exit 1
    }
}
