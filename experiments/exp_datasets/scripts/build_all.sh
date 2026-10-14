#!/usr/bin/env bash
# build_all.sh — Regenerate all exp_datasets from scratch.
#
# Run from the repository root:
#   bash experiments/exp_datasets/scripts/build_all.sh
#
# Each step is independent; comment out steps you want to skip.
# Pass --overwrite to force-regenerate existing files:
#   bash experiments/exp_datasets/scripts/build_all.sh --overwrite

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERWRITE="${1:-}"

_py() { python "$SCRIPT_DIR/$@"; }
_sh() { bash "$SCRIPT_DIR/$@"; }

echo "======================================================"
echo "  DistNNP exp_datasets full rebuild"
echo "  Scripts: $SCRIPT_DIR"
echo "======================================================"
echo ""

# ── Step 1: Download PDB files ─────────────────────────────────────────
echo ">>> Step 1: Download PDB files"
_sh 01_download_pdbs.sh
echo ""

# ── Step 2: Convert PDB → vacuum XYZ ──────────────────────────────────
echo ">>> Step 2: Convert vacuum proteins"
_py 02_convert_vacuum.py ${OVERWRITE:+--overwrite}
echo ""

# ── Step 3: Generate CH4 baselines ────────────────────────────────────
echo ">>> Step 3: Generate CH4/O2 baselines"
_py 03_generate_ch4.py ${OVERWRITE:+--overwrite}
echo ""

# ── Step 4: Generate solvated proteins ────────────────────────────────
echo ">>> Step 4: Generate solvated proteins (grid water)"
echo "    NOTE: 9gmo takes ~10 min (1.5M atoms)"
_py 04_generate_solvated.py ${OVERWRITE:+--overwrite}
echo ""

# ── Step 5: Generate OpenMM equilibrated systems ──────────────────────
echo ">>> Step 5: Generate OpenMM equilibrated systems (DHFR, ApoA1, DPPC)"
_py 05_generate_openmm.py ${OVERWRITE:+--overwrite}
echo ""

# ── Step 6: Generate electrochemical systems ──────────────────────────
echo ">>> Step 6: Generate Pt slab + water"
_py 06_generate_electrochemical.py ${OVERWRITE:+--overwrite}
echo ""

# ── Step 7: Retrieve HIV-1 capsid (3J3Q) + shell subsets ──────────────
echo ">>> Step 7: Retrieve HIV-1 capsid (3J3Q, ~2.4 M atoms)"
echo "    NOTE: downloads ~40 MB from RCSB and parses ~2.4 M atoms (~1 min)"
_py 07_retrieve_hiv.py --subsets ${OVERWRITE:+--overwrite}
echo ""

echo "======================================================"
echo "  Done. Dataset summary:"
echo "======================================================"

# Print file count and total size per subdirectory
for d in baselines protein_vacuum protein_solvated membrane electrochemical capsid pdb; do
    dir="$SCRIPT_DIR/../$d"
    if [[ -d "$dir" ]]; then
        n=$(find "$dir" -name "*.xyz" -o -name "*.pdb" 2>/dev/null | wc -l)
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "  %-25s %3d files  %s\n" "$d/" "$n" "$sz"
    fi
done
