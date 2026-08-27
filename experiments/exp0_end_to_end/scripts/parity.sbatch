#!/bin/bash
# Correctness gate for the venv build: the distributed (2-rank) UMA MD
# trajectory must match the single-process baseline within the repo's tolerance
# schedule. This is the real check that the installed fairchem-core's internals
# still line up with distnnp_model/nnp/uma.py. Then a quick driver smoke on
# both engines.
#
#   scripts/sbatch_exp0.sh --gpus-per-node=2 scripts/parity.sbatch
#SBATCH --job-name=exp0-parity
#SBATCH --time=00:30:00

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

WORK="${EXP0_SCRATCH:-/tmp}/uma_check_${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK" "$RESULTS_DIR/logs"

cd "$MONO/distnnp_model/tests"

echo "=== [1/3] distributed 2-rank trajectory dump ==="
"$TORCHRUN" --standalone --nproc_per_node=2 run_distributed_trajectory.py --path "$WORK"
RC1=$?
echo "rc=$RC1"

echo "=== [2/3] single-process baseline + compare (must print PASSED) ==="
"$PY" test_uma_md.py --path "$WORK"
RC2=$?
echo "rc=$RC2"

echo "=== [3/3] driver smoke: both engines, tiny cell, 2 GPU ==="
XYZ="$EXP_DIR/exp_datasets/baselines/ch4_1k_splited.xyz"
SMOKE_OUT="$RESULTS_DIR/smoke"
mkdir -p "$SMOKE_OUT"
"$PY" "$EXP_DIR/drivers/run_fairchem_uma.py" --input "$XYZ" --engine fairchem-uma \
    --workers 2 --output-dir "$SMOKE_OUT" --tag parity --warmup-steps 3 --timed-steps 10 || true
"$TORCHRUN" --standalone --nproc_per_node=2 "$EXP_DIR/drivers/run_distnnp_uma.py" \
    --input "$XYZ" --engine distnnp-uma --output-dir "$SMOKE_OUT" --tag parity \
    --warmup-steps 3 --timed-steps 10 || true

echo "=== summary ==="
echo "trajectory dump rc=$RC1 ; parity compare rc=$RC2"
ls -la "$SMOKE_OUT"/*.json 2>/dev/null
[ "$RC2" -eq 0 ] && echo "PARITY GATE: PASS" || echo "PARITY GATE: FAIL"
rm -rf "$WORK"
exit "$RC2"
