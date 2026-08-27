#!/bin/bash
# Thin wrapper over the unmodified CH4:O2 generator. numpy-only, bit-repro,
# runs in a few seconds on a login node (no GPU, no Slurm). Does NOT modify
# 03_generate_ch4.py.
#
#   bash scripts/gen_datasets.sh [SIZE_K ...]
#
# Default sizes cover smoke (1,2), strong (32), fill (64,128) and the OOM
# ceiling (192,256). Output: exp_datasets/baselines/ch4_<k>k_splited.xyz
#
# Positions are checksum-verified for sizes present in checksums.txt; sizes
# added later (not in the pinned set) are generated and reported but skipped
# by the verify step (they are still bit-reproducible from the same seed).
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
GEN="$EXP_DIR/exp_datasets/scripts/03_generate_ch4.py"

# Prefer the exp0 venv's python (has numpy); fall back to system python3.
if [ -f "$SCRIPTS_DIR/cluster.env" ]; then . "$SCRIPTS_DIR/cluster.env"; fi
REPO_ROOT="${EXP0_REPO_ROOT:-$(cd "$EXP_DIR/../.." && pwd)}"
VENV="${EXP0_VENV:-$REPO_ROOT/DistNNP/.venv-uma}"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"

SIZES=("$@")
[ ${#SIZES[@]} -eq 0 ] && SIZES=(1 2 32 64 128 192 256)

echo "Generating sizes: ${SIZES[*]}  (python: $PY)"
"$PY" "$GEN" --sizes "${SIZES[@]}" --variants splited --out-dir "$EXP_DIR/exp_datasets/baselines"

echo "Verifying positions against checksums.txt (pinned sizes only)..."
cd "$EXP_DIR/exp_datasets"
verified=0 skipped=""
for k in "${SIZES[@]}"; do
  line="$(grep -E "  baselines/ch4_${k}k_splited\.xyz$" scripts/checksums.txt || true)"
  if [ -n "$line" ]; then
    echo "$line" | sha256sum -c -
    verified=$((verified+1))
  else
    skipped="$skipped $k"
  fi
done
echo "verified $verified pinned size(s);${skipped:+ not in checksums.txt (ok, still reproducible):$skipped}"
