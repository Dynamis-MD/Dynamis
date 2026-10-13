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
# Overridable so a sweep (e.g. tune_kernels.sbatch) can keep its cells out of
# the main matrix dir -- mixing kernel configurations in one dir makes steps/s
# incomparable (summarize.py warns about it).
RESULTS_DIR="${EXP0_RESULTS_DIR:-$EXP_DIR/results}"
VENV="${EXP0_VENV:-$MONO/.venv-uma}"
PY="$VENV/bin/python"
TORCHRUN="$VENV/bin/torchrun"

# --- LAMMPS + MACE baseline (Workstream B) --------------------------------
# Separate env + a from-source LAMMPS build; see cluster.env for the defaults.
LAMMPS_VENV="${EXP0_LAMMPS_VENV:-$MONO/.venv-lammps-mace}"
LAMMPS_PY="$LAMMPS_VENV/bin/python"
LMP_BIN="${EXP0_LAMMPS_BIN:-$LAMMPS_VENV/bin/lmp}"

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
        if ! module load ${EXP0_MODULES} 2>&1; then
            echo "[_lib] WARNING: that 'module load' failed. EXP0_MODULES holds the DELTA"
            echo "[_lib]   module names -- on another cluster, set EXP0_MODULES in"
            echo "[_lib]   scripts/cluster.local.env to your cluster's python / CUDA /"
            echo "[_lib]   compiler modules, or to \"\" if the toolchain is already on PATH."
            echo "[_lib]   Continuing anyway -- steps that truly need a module will say so."
        fi
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

# --- LAMMPS module / venv helpers (mirror the UMA ones) -------------------
exp0_load_lammps_modules() {
    if ! command -v module >/dev/null 2>&1; then
        echo "[_lib] no 'module' command -- assuming LAMMPS toolchain already on PATH"
        return 0
    fi
    [ "${EXP0_MODULE_PURGE:-0}" = "1" ] && { echo "[_lib] module purge"; module purge 2>/dev/null || true; }
    for m in ${EXP0_MODULE_UNLOAD:-}; do
        module unload "$m" 2>/dev/null || true
    done
    if [ -n "${EXP0_LAMMPS_MODULES:-}" ]; then
        echo "[_lib] module load ${EXP0_LAMMPS_MODULES}"
        # shellcheck disable=SC2086
        module load ${EXP0_LAMMPS_MODULES}
    fi
    module list 2>&1 || true
    # liblammps.so + the venv's libtorch must be findable at runtime.
    local _lmp_prefix _torch_lib
    _lmp_prefix="$(dirname "$(dirname "$LMP_BIN")")"
    _torch_lib="$(ls -d "$LAMMPS_VENV"/lib/python*/site-packages/torch/lib 2>/dev/null | head -1)"
    export LD_LIBRARY_PATH="$_lmp_prefix/lib64:$_lmp_prefix/lib:${_torch_lib}:${LD_LIBRARY_PATH:-}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    # PYTHONHOME/PYTHONPATH for the interpreter EMBEDDED in `lmp` are set per
    # subprocess by drivers/run_lammps.py (setting them here would perturb the
    # venv python that runs run_lammps.py itself).
}

exp0_check_lammps() {
    [ -x "$LMP_BIN" ] || {
        echo "ERROR: lmp binary not found / not executable at $LMP_BIN"
        echo "       build it first:  scripts/sbatch_exp0.sh --cpu scripts/build_lammps.sh"
        echo "       (set EXP0_LAMMPS_BIN / cluster.local.env to point at an existing one)"
        exit 1
    }
    [ -x "$LAMMPS_PY" ] || {
        echo "ERROR: LAMMPS venv python not found at $LAMMPS_PY"
        echo "       build it first:  scripts/build_lammps_env.sh"
        exit 1
    }
}

# Copy the LAMMPS venv + install tree to NODE-LOCAL scratch and repoint
# LAMMPS_VENV / LAMMPS_PY / LMP_BIN at it. The env lives on Lustre, and a cold
# read of torch's ~2 GB of .so files on a busy/degraded Lustre client can stall
# a job for many minutes; a one-time local copy makes every later python /
# `lmp` invocation instant. Set EXP0_LAMMPS_NO_STAGE=1 to skip.
exp0_stage_lammps() {
    [ "${EXP0_LAMMPS_NO_STAGE:-0}" = "1" ] && { echo "[_lib] venv staging disabled"; return 0; }
    local dst="${SLURM_TMPDIR:-/tmp}/exp0-lammps.${SLURM_JOB_ID:-$$}"
    local root; root="$(dirname "$LAMMPS_VENV")"
    local venv_tar="$root/venv-lammps-mace.tar" inst_tar="$root/lammps-install.tar"
    local src_prefix; src_prefix="$(dirname "$(dirname "$LMP_BIN")")"
    local t0=$SECONDS
    mkdir -p "$dst"
    # Prefer a single-file tarball (streams fast; many-small-file reads on the
    # network fs are pathological). Fall back to cp -a.
    if [ -f "$venv_tar" ]; then
        echo "[_lib] staging via $venv_tar -> $dst"
        tar -xf "$venv_tar" -C "$dst" && LAMMPS_VENV="$dst/$(basename "$LAMMPS_VENV")" || {
            echo "[_lib] venv untar failed -- using network path"; return 0; }
    else
        echo "[_lib] staging via cp -a $LAMMPS_VENV -> $dst/venv"
        cp -a "$LAMMPS_VENV" "$dst/venv" && LAMMPS_VENV="$dst/venv" || {
            echo "[_lib] venv copy failed -- using network path"; return 0; }
    fi
    if [ -f "$inst_tar" ]; then
        tar -xf "$inst_tar" -C "$dst" && src_prefix="$dst/$(basename "$src_prefix")" || \
            echo "[_lib] install untar failed -- keeping network lmp"
    else
        cp -a "$src_prefix" "$dst/install" && src_prefix="$dst/install" || \
            echo "[_lib] install copy failed -- keeping network lmp"
    fi
    LAMMPS_PY="$LAMMPS_VENV/bin/python"
    LMP_BIN="$src_prefix/bin/lmp"
    export LD_LIBRARY_PATH="$src_prefix/lib64:$src_prefix/lib:$(ls -d "$LAMMPS_VENV"/lib/python*/site-packages/torch/lib 2>/dev/null | head -1):${LD_LIBRARY_PATH:-}"
    echo "[_lib] staged in $((SECONDS - t0))s: LMP_BIN=$LMP_BIN  LAMMPS_PY=$LAMMPS_PY"
}
