#!/usr/bin/env bash
# download_datasets.sh — Fetch the datasets that are NOT generated locally.
#
# Two tiers are handled here:
#   Tier 2  raw PDB structures  -> delegates to 01_download_pdbs.sh (RCSB)
#   Tier 3  version-dependent protein/membrane xyz -> one hosted tar.xz archive,
#           verified against scripts/checksums.txt, then extracted in place.
#
# The CH4/O2 baselines (Tier 1) are NOT downloaded — generate them with
#   python 03_generate_ch4.py            (or use setup_data.sh for everything)
#
# Usage:
#   bash download_datasets.sh                 # pdb + hosted archive
#   bash download_datasets.sh --pdb-only
#   bash download_datasets.sh --archive-only
#
# The archive is distributed as a GitHub Release asset (NOT committed to git —
# it stays out of the repo's history). By default this script fetches it from
# the release below; create it once with:
#   gh release create datasets-v1 distnnp_exp_datasets_v1.tar.xz \
#       --title "DistNNP experiment datasets v1" --notes "protein/membrane XYZ"
# Override the source via any of:
#   export DISTNNP_ARCHIVE_REPO="owner/repo"      # default: szaman19/DistNNP
#   export DISTNNP_ARCHIVE_TAG="datasets-v1"      # release tag
#   export DISTNNP_ARCHIVE_URL="https://.../distnnp_exp_datasets_v1.tar.xz"  # full URL
#   export DISTNNP_ARCHIVE_LOCAL="/path/to/distnnp_exp_datasets_v1.tar.xz"   # local copy

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # experiments/exp_datasets/
CHECKSUMS="$SCRIPT_DIR/checksums.txt"

ARCHIVE_NAME="distnnp_exp_datasets_v1.tar.xz"
# Default source: a GitHub Release asset (see header for how to create it).
ARCHIVE_REPO="${DISTNNP_ARCHIVE_REPO:-szaman19/DistNNP}"
ARCHIVE_TAG="${DISTNNP_ARCHIVE_TAG:-datasets-v1}"
ARCHIVE_URL="${DISTNNP_ARCHIVE_URL:-https://github.com/${ARCHIVE_REPO}/releases/download/${ARCHIVE_TAG}/${ARCHIVE_NAME}}"
ARCHIVE_LOCAL="${DISTNNP_ARCHIVE_LOCAL:-}"

DO_PDB=1
DO_ARCHIVE=1
case "${1:-}" in
    --pdb-only)     DO_ARCHIVE=0 ;;
    --archive-only) DO_PDB=0 ;;
    "" )            ;;
    * ) echo "Unknown option: $1"; exit 2 ;;
esac

# Verify a subset of checksums.txt selected by a path-prefix regex.
# Runs from DATASETS_DIR so the relative paths match. Non-fatal for Tier 2.
verify_subset() {
    local prefix_regex="$1" label="$2" fatal="$3"
    local lines
    lines="$(grep -E "^[0-9a-f]{64}  ${prefix_regex}" "$CHECKSUMS" || true)"
    if [[ -z "$lines" ]]; then
        echo "  (no checksums for ${label})"; return 0
    fi
    echo "  Verifying ${label} ..."
    if ( cd "$DATASETS_DIR" && echo "$lines" | sha256sum -c - ) ; then
        echo "  OK: ${label} checksums match"
    else
        if [[ "$fatal" == "1" ]]; then
            echo "  ERROR: ${label} checksum mismatch"; exit 1
        else
            echo "  WARN: ${label} checksum mismatch (informational tier — upstream may have changed)"
        fi
    fi
}

# ── Tier 2: raw PDB structures ────────────────────────────────────────────
if [[ "$DO_PDB" == "1" ]]; then
    echo ">>> Tier 2: PDB structures (RCSB)"
    bash "$SCRIPT_DIR/01_download_pdbs.sh"
    verify_subset 'pdb/' "PDB files" 0
    echo ""
fi

# ── Tier 3: hosted version-dependent archive ──────────────────────────────
if [[ "$DO_ARCHIVE" == "1" ]]; then
    echo ">>> Tier 3: hosted protein/membrane archive"

    # Resolve the archive source. `cleanup=1` marks a temp we downloaded and may
    # delete; a user-provided local archive is NEVER deleted.
    if [[ -n "$ARCHIVE_LOCAL" ]]; then
        echo "  Using local archive: $ARCHIVE_LOCAL"
        src="$ARCHIVE_LOCAL"; cleanup=0
    else
        echo "  Downloading $ARCHIVE_URL ..."
        src="$DATASETS_DIR/$ARCHIVE_NAME"; cleanup=1
        if ! wget -q -O "$src" "$ARCHIVE_URL"; then
            rm -f "$src"
            echo "  ERROR: download failed."
            echo "  Expected a published GitHub Release '${ARCHIVE_TAG}' on ${ARCHIVE_REPO}"
            echo "  with asset '${ARCHIVE_NAME}' attached. Create it with:"
            echo "    gh release create ${ARCHIVE_TAG} ${ARCHIVE_NAME} --title '...' --notes '...'"
            echo "  Or override: DISTNNP_ARCHIVE_URL=<url> / DISTNNP_ARCHIVE_LOCAL=<path>"
            exit 1
        fi
    fi

    # Verify the archive itself before trusting it.
    echo "  Verifying archive checksum ..."
    expected="$(grep "  ${ARCHIVE_NAME}\$" "$CHECKSUMS" | awk '{print $1}')"
    actual="$(sha256sum "$src" | awk '{print $1}')"
    if [[ "$expected" != "$actual" ]]; then
        echo "  ERROR: archive checksum mismatch"
        echo "    expected $expected"; echo "    actual   $actual"
        [[ "$cleanup" == "1" ]] && rm -f "$src"
        exit 1
    fi
    echo "  OK: archive checksum matches"

    echo "  Extracting into $DATASETS_DIR ..."
    tar -C "$DATASETS_DIR" -xf "$src"
    [[ "$cleanup" == "1" ]] && rm -f "$src"

    # Verify extracted per-file checksums (fatal).
    verify_subset '(protein_vacuum/|protein_solvated/|membrane/)' "extracted files" 1
    echo ""
fi

echo "Done."
