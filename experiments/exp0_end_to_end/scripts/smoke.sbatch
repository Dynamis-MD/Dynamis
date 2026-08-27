#!/bin/bash
# <=3 min smoke test: one tiny cell per engine on 2 GPUs, then assert each
# wrote a schema-valid results/*.json with status "ok".
#
#   scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/smoke.sbatch
#SBATCH --job-name=exp0-smoke
#SBATCH --time=00:20:00

set -uo pipefail

# Slurm runs this from a spool copy, so ${BASH_SOURCE[0]} is not scripts/.
# scripts/sbatch_exp0.sh exports EXP0_SCRIPTS_DIR; fall back to $0's dir when
# the script is run directly (non-Slurm testing).
_S="${EXP0_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$_S/_lib.sh" ] || { echo "ERROR: scripts/_lib.sh not found in '$_S' -- submit via scripts/sbatch_exp0.sh"; exit 1; }
source "$_S/_lib.sh"
exp0_check_venv
exp0_load_modules
exp0_export_runtime

RESULTS="$RESULTS_DIR/smoke"
XYZ="$EXP_DIR/exp_datasets/baselines/ch4_1k_splited.xyz"
mkdir -p "$RESULTS/logs"

COMMON=(--warmup-steps 3 --timed-steps 10)

echo "=== smoke: fairchem-uma (workers 2) ==="
"$PY" "$EXP_DIR/drivers/run_fairchem_uma.py" --input "$XYZ" --engine fairchem-uma \
    --workers 2 --output-dir "$RESULTS" --tag smoke "${COMMON[@]}" || true

echo "=== smoke: distnnp-uma (2 ranks) ==="
"$TORCHRUN" --standalone --nproc_per_node=2 "$EXP_DIR/drivers/run_distnnp_uma.py" \
    --input "$XYZ" --engine distnnp-uma --output-dir "$RESULTS" --tag smoke "${COMMON[@]}" || true

echo "=== validate ==="
"$PY" - "$RESULTS" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.json")))
assert files, f"no result JSON in {d}"
required = {"engine","num_atoms","world_size","status","steps_per_sec"}
bad = 0
for f in files:
    r = json.load(open(f))
    missing = required - set(r)
    ok = (not missing) and r["status"] == "ok" and (r.get("steps_per_sec") or 0) > 0 \
         and r["steps_per_sec"] < 1e4
    print(f"  {os.path.basename(f)}: status={r['status']} "
          f"steps/s={r.get('steps_per_sec')} missing={missing or '-'} -> {'OK' if ok else 'FAIL'}")
    bad += (not ok)
sys.exit(1 if bad else 0)
PY
RC=$?
echo "=== smoke rc=$RC ==="
exit $RC
