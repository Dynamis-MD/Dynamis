#!/usr/bin/env bash
# setup_data.sh — One command to obtain every experiment dataset.
#
# Fastest, most reproducible path (the default):
#   Tier 1  CH4/O2 baselines   -> GENERATED locally (03_generate_ch4.py), no deps
#                                 beyond numpy, bit-reproducible everywhere.
#   Tier 2  raw PDB structures -> DOWNLOADED from RCSB.
#   Tier 3  protein/membrane   -> DOWNLOADED as one checksummed archive
#                                 (avoids re-running pdbfixer/OpenMM, which are
#                                  not reproducible across versions).
#
# Usage:
#   bash setup_data.sh                 # generate CH4 + download pdb + archive
#   bash setup_data.sh --generate-only # only the CH4/O2 baselines (numpy only)
#   bash setup_data.sh --no-archive    # CH4 + pdb, skip the hosted archive
#   bash setup_data.sh --from-scratch  # rebuild EVERYTHING with build_all.sh
#                                        (needs rdkit/pdbfixer/openmm; version-
#                                         dependent output — not recommended)
#
# The hosted archive needs a source; configure it before running Tier 3:
#   export DISTNNP_ARCHIVE_URL=<url>     or     export DISTNNP_ARCHIVE_LOCAL=<path>
# (see download_datasets.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKSUMS="$SCRIPT_DIR/checksums.txt"

MODE="default"
case "${1:-}" in
    --generate-only) MODE="generate" ;;
    --no-archive)    MODE="noarchive" ;;
    --from-scratch)  MODE="scratch" ;;
    "" )             ;;
    * ) echo "Unknown option: $1"; exit 2 ;;
esac

echo "======================================================"
echo "  DistNNP experiment data setup  (mode: $MODE)"
echo "======================================================"

if [[ "$MODE" == "scratch" ]]; then
    echo ">>> Rebuilding all datasets from scratch (build_all.sh)"
    bash "$SCRIPT_DIR/build_all.sh" "${2:-}"
    echo "Done (from scratch)."
    exit 0
fi

# ── Tier 1: generate CH4/O2 baselines (numpy only) ────────────────────────
echo ""
echo ">>> Tier 1: generating CH4/O2 baselines"
python "$SCRIPT_DIR/03_generate_ch4.py"

echo "  Verifying Tier 1 checksums ..."
if ( cd "$DATASETS_DIR" \
     && grep -E '^[0-9a-f]{64}  baselines/' "$CHECKSUMS" | sha256sum -c - >/dev/null ) ; then
    echo "  OK: CH4/O2 baselines reproduced byte-for-byte"
else
    echo "  WARN: CH4 checksums differ from the pinned manifest."
    echo "        Positions should be bit-exact; momenta may differ across libm."
fi

if [[ "$MODE" == "generate" ]]; then
    echo ""; echo "Done (Tier 1 only)."
    exit 0
fi

# ── Tier 2 + 3: downloads ─────────────────────────────────────────────────
echo ""
if [[ "$MODE" == "noarchive" ]]; then
    bash "$SCRIPT_DIR/download_datasets.sh" --pdb-only
else
    bash "$SCRIPT_DIR/download_datasets.sh"
fi

# ── summary ───────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Dataset summary"
echo "======================================================"
for d in baselines pdb protein_vacuum protein_solvated membrane electrochemical; do
    dir="$DATASETS_DIR/$d"
    if [[ -d "$dir" ]]; then
        n=$(find "$dir" \( -name '*.xyz' -o -name '*.pdb' -o -name '*.cif' \) 2>/dev/null | wc -l)
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "  %-22s %3d files  %s\n" "$d/" "$n" "$sz"
    fi
done
echo ""
echo "Note: Tier 4 *_density.xyz (large solvated systems) are external user data"
echo "      (../data/ symlinks) — regenerate with 04/06 or provide separately."
echo "Done."
