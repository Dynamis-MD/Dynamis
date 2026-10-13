#!/usr/bin/env bash
# 01_download_pdbs.sh — Download raw PDB files from RCSB and OpenMM.
#
# Outputs to ../pdb/ (relative to this script's location).
# Skips files that already exist.
#
# Usage:
#   bash scripts/01_download_pdbs.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../pdb"
mkdir -p "$OUT_DIR"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  SKIP (exists): $(basename "$dest")"
    else
        echo "  Downloading $(basename "$dest") ..."
        wget -q -O "$dest" "$url" || { echo "  ERROR: $url"; rm -f "$dest"; exit 1; }
        echo "  OK: $(basename "$dest")"
    fi
}

# Non-fatal variant for optional / upstream-volatile sources (e.g. files in a
# third-party GitHub repo that may move). A failure warns and continues instead
# of aborting the whole setup.
download_optional() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  SKIP (exists): $(basename "$dest")"
    elif wget -q -O "$dest" "$url"; then
        echo "  OK: $(basename "$dest")"
    else
        rm -f "$dest"
        echo "  WARN: could not fetch $(basename "$dest") from $url"
        echo "        (optional — its processed form ships in the Tier 3 archive)"
    fi
}

echo "=== RCSB PDB structures ==="

# Lysozyme — small compact protein, ~1.9k atoms without H
download "https://files.rcsb.org/download/1AKI.pdb" "$OUT_DIR/1AKI.pdb"

# Titin I-band fragment — rod-shaped, ~4.4k atoms without H
download "https://files.rcsb.org/download/2RIK.pdb" "$OUT_DIR/2RIK.pdb"

# Large ribosomal complex — ~229k atoms without H
# NOTE: 9GMO is available only as mmCIF (PDB format too large)
download "https://files.rcsb.org/download/9GMO.cif" "$OUT_DIR/9GMO.cif"

# GroEL chaperonin ring — 7-chain symmetric complex, ~57k atoms
download "https://files.rcsb.org/download/1GRL.pdb" "$OUT_DIR/1GRL.pdb"

# Insect flight muscle filament — DistMLIP benchmark, extremely sparse
download "https://files.rcsb.org/download/2W49.pdb" "$OUT_DIR/2W49.pdb"

# GPCR multi-chain — DistMLIP benchmark, multiple separated chains
download "https://files.rcsb.org/download/6P9X.pdb" "$OUT_DIR/6P9X.pdb"

echo ""
echo "=== OpenMM examples ==="

# ApoA1 lipoprotein — protein + POPC membrane + water, pre-equilibrated by OpenMM.
# Optional: only needed to regenerate membrane/apoa1 from scratch via 05; the
# processed apoa1_92k_openmm.xyz already ships in the Tier 3 archive.
download_optional \
    "https://raw.githubusercontent.com/openmm/openmm/master/examples/benchmarks/apoa1.pdb" \
    "$OUT_DIR/apoa1.pdb"

echo ""
echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.pdb "$OUT_DIR"/*.cif 2>/dev/null || true
