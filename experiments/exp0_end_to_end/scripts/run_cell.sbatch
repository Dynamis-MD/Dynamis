#!/bin/bash
# Run ONE benchmark cell: (engine, xyz, ngpu).
#
# Submit via the cluster-agnostic wrapper (injects account/partition/mem/output
# from scripts/cluster.env):
#
#   scripts/sbatch_exp0.sh --gpus-per-node=<N> --job-name=<name> --time=00:20:00 \
#       scripts/run_cell.sbatch <engine> <xyz> <N> [extra driver args]
#
#   engine : distnnp-uma | fairchem-uma
#
# On any failure (crash, timeout, or no result JSON produced) the trap writes
# a stub JSON with status oom/error so the matrix never has a hole. A Ray
# worker OOM typically hangs the next broadcast, hence the hard timeout.
#SBATCH --job-name=exp0-cell
#SBATCH --time=00:25:00

set -uo pipefail

# Slurm runs this from a spool copy, so ${BASH_SOURCE[0]} is not scripts/.
# scripts/sbatch_exp0.sh exports EXP0_SCRIPTS_DIR; fall back to $0's dir when
# the script is run directly (non-Slurm testing).
_S="${EXP0_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$_S/_lib.sh" ] || { echo "ERROR: scripts/_lib.sh not found in '$_S' -- submit via scripts/sbatch_exp0.sh"; exit 1; }
source "$_S/_lib.sh"

ENGINE="${1:?engine}"; XYZ="${2:?xyz path}"; NGPU="${3:?ngpu}"; shift 3
EXTRA_ARGS=("$@")

RESULTS="$RESULTS_DIR"
mkdir -p "$RESULTS/logs"

# The lammps-* engines use a separate venv + module set + `lmp` binary.
case "$ENGINE" in
  lammps-*)
    exp0_check_lammps
    exp0_load_lammps_modules
    exp0_stage_lammps
    ;;
  *)
    exp0_check_venv
    exp0_load_modules
    ;;
esac
exp0_export_runtime

NUM_ATOMS="$(head -n1 "$XYZ" | tr -d '[:space:]')"
TAG="${CELL_TAG:-}"
STEM="${ENGINE}_${NUM_ATOMS}atoms_${NGPU}gpu"
[ -n "$TAG" ] && STEM="${STEM}_${TAG}"
RESULT_JSON="$RESULTS/${STEM}.json"
rm -f "$RESULT_JSON"

write_stub() {
    local status="$1" code="$2"
    [ -f "$RESULT_JSON" ] && return 0
    local stub_py="$PY"; [ -x "$stub_py" ] || stub_py="$LAMMPS_PY"; [ -x "$stub_py" ] || stub_py="$(command -v python3)"
    "$stub_py" - "$RESULT_JSON" "$ENGINE" "$NUM_ATOMS" "$NGPU" "$status" "$code" "$XYZ" "$TAG" "${CELL_MODEL:-uma-s-1p2}" <<'PY'
import json, sys, datetime, subprocess
path, engine, natoms, ngpu, status, code, xyz, tag, model = sys.argv[1:10]
try:
    commit = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
except Exception:
    commit = "unknown"
task = "none" if engine.startswith("lammps-") else "omol"
rec = {
    "engine": engine, "model": model, "task": task, "integrator": "langevin",
    "num_atoms": int(natoms), "world_size": int(ngpu), "gpus_per_node": int(ngpu), "nodes": 1,
    "status": status, "exit_code": int(code), "input_file": xyz,
    "tag": tag or None, "steps_per_sec": None, "ns_per_day": None,
    "peak_mem_mb_per_rank": None, "peak_mem_mb_max": None,
    "error": f"cell wrapper caught {status} (exit {code}) with no result JSON",
    "git_commit": commit,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z"),
}
with open(path, "w") as f:
    json.dump(rec, f, indent=2, sort_keys=True)
print(f"wrote stub {path}: {status}")
PY
}

# timeout / signal -> stub as oom (Ray-worker OOM manifests as a hang)
trap 'rc=$?; echo "trap: caught signal / non-zero ($rc)"; write_stub oom "$rc"; exit $rc' TERM INT

# Hard external kill: the in-driver SIGALRM watchdog cannot interrupt a thread
# stuck in a blocking NCCL/CUDA C call (a dead Ray worker -> 600 s collective
# timeout). `timeout` reaps the whole process tree well before that.
HARD_TIMEOUT="${HARD_TIMEOUT:-600}"

echo "=== cell: engine=$ENGINE xyz=$(basename "$XYZ") natoms=$NUM_ATOMS ngpu=$NGPU extra=${EXTRA_ARGS[*]:-} hard_timeout=${HARD_TIMEOUT}s ==="
case "$ENGINE" in
  lammps-*) echo "    lammps_venv=$LAMMPS_VENV  lmp=$LMP_BIN" ;;
  *)        echo "    venv=$VENV" ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RC=0
case "$ENGINE" in
  distnnp-uma)
    timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT" \
        "$TORCHRUN" --standalone --nproc_per_node="$NGPU" \
        "$EXP_DIR/drivers/run_distnnp_uma.py" \
        --input "$XYZ" --engine distnnp-uma \
        --output-dir "$RESULTS" ${TAG:+--tag "$TAG"} "${EXTRA_ARGS[@]}" || RC=$?
    ;;
  fairchem-uma)
    timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT" \
        "$PY" "$EXP_DIR/drivers/run_fairchem_uma.py" \
        --input "$XYZ" --engine fairchem-uma --workers "$NGPU" \
        --output-dir "$RESULTS" ${TAG:+--tag "$TAG"} "${EXTRA_ARGS[@]}" || RC=$?
    ;;
  lammps-mace)
    # run_lammps.py is a single python process that spawns `lmp` / `mpirun`; it needs
    # the hard timeout as an env var (it cannot be wrapped in `timeout` cleanly
    # because that would orphan the srun step).
    EXP0_LAMMPS_HARD_TIMEOUT="$HARD_TIMEOUT" \
        timeout --signal=TERM --kill-after=60 "$((HARD_TIMEOUT + 120))" \
        "$LAMMPS_PY" "$EXP_DIR/drivers/run_lammps.py" \
        --input "$XYZ" --engine lammps-mace --world-size "$NGPU" \
        --lammps-bin "$LMP_BIN" \
        --mace-model "${EXP0_MACE_MODEL:?set EXP0_MACE_MODEL in cluster.env}" \
        --elements "${EXP0_MACE_ELEMENTS:-C H O}" \
        --output-dir "$RESULTS" ${TAG:+--tag "$TAG"} "${EXTRA_ARGS[@]}" || RC=$?
    ;;
  *)
    echo "unknown engine: $ENGINE"; exit 2 ;;
esac

if [ ! -f "$RESULT_JSON" ]; then
    echo "no result JSON at $RESULT_JSON (driver rc=$RC) -> writing stub"
    # 124 = timeout expired, 137 = timeout's SIGKILL: treat as OOM/hang, not a bug
    if [ "$RC" = "124" ] || [ "$RC" = "137" ]; then
        write_stub oom "$RC"
    elif [ "$RC" -ne 0 ]; then
        write_stub error "$RC"
    else
        write_stub error 0
    fi
fi

echo "=== cell done (rc=$RC) -> $RESULT_JSON ==="
exit "$RC"
