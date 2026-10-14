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
# Archive source resolution (first hit that verifies wins):
#   1. DYNAMIS_ARCHIVE_LOCAL      — explicit local copy (never downloaded/deleted)
#   2. shared-dir cache           — DYNAMIS_ARCHIVE_SHARED_DIR/<archive> (see below)
#   3. gh release download        — authenticated (needed for a PRIVATE repo)
#   4. wget from DYNAMIS_ARCHIVE_URL / GitHub Release URL (public only)
# On a shared cluster (e.g. Delta) the archive lives in a group-readable dir so
# the ~23 MB download happens ONCE for the whole team; only a miss or a stale
# copy falls through to the network. A fresh download is cached back there.
#
# The archive is distributed as a GitHub Release asset (NOT committed to git —
# it stays out of the repo's history). Create it once with:
#   gh release create datasets-v1 dynamis_exp_datasets_v1.tar.xz \
#       --title "Dynamis experiment datasets v1" --notes "protein/membrane/capsid XYZ"
# Override the source via any of:
#   export DYNAMIS_ARCHIVE_REPO="owner/repo"      # default: Dynamis-MD/Dynamis
#   export DYNAMIS_ARCHIVE_TAG="datasets-v1"      # release tag
#   export DYNAMIS_ARCHIVE_URL="https://.../dynamis_exp_datasets_v1.tar.xz"  # full URL
#   export DYNAMIS_ARCHIVE_LOCAL="/path/to/dynamis_exp_datasets_v1.tar.xz"   # local copy
#   export DYNAMIS_ARCHIVE_SHARED_DIR="/projects/bcpt/public/dynamis_datasets" # shared cache (default)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # experiments/exp_datasets/
CHECKSUMS="$SCRIPT_DIR/checksums.txt"

ARCHIVE_NAME="dynamis_exp_datasets_v1.tar.xz"
# Default source: a GitHub Release asset (see header for how to create it).
ARCHIVE_REPO="${DYNAMIS_ARCHIVE_REPO:-Dynamis-MD/Dynamis}"
ARCHIVE_TAG="${DYNAMIS_ARCHIVE_TAG:-datasets-v1}"
ARCHIVE_URL="${DYNAMIS_ARCHIVE_URL:-https://github.com/${ARCHIVE_REPO}/releases/download/${ARCHIVE_TAG}/${ARCHIVE_NAME}}"
ARCHIVE_LOCAL="${DYNAMIS_ARCHIVE_LOCAL:-}"
# Shared group-readable cache (checked before any download; "" disables).
ARCHIVE_SHARED_DIR="${DYNAMIS_ARCHIVE_SHARED_DIR:-/projects/bcpt/public/dynamis_datasets}"

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
    echo ">>> Tier 3: hosted protein/membrane/capsid archive"

    expected="$(grep "  ${ARCHIVE_NAME}\$" "$CHECKSUMS" | awk '{print $1}')"
    archive_ok() {  # 0 iff $1 exists and its sha256 matches the pinned archive hash
        [[ -f "$1" ]] || return 1
        [[ "$(sha256sum "$1" | awk '{print $1}')" == "$expected" ]]
    }
    shared_copy=""
    [[ -n "$ARCHIVE_SHARED_DIR" ]] && shared_copy="$ARCHIVE_SHARED_DIR/$ARCHIVE_NAME"

    # Resolve the archive source. `cleanup=1` marks a temp we downloaded and may
    # delete; a local/shared copy is NEVER deleted.
    src=""; cleanup=0
    if [[ -n "$ARCHIVE_LOCAL" ]]; then
        echo "  Using local archive: $ARCHIVE_LOCAL"
        src="$ARCHIVE_LOCAL"; cleanup=0
    elif [[ -n "$shared_copy" && -f "$shared_copy" ]] && archive_ok "$shared_copy"; then
        # Primary path on a shared cluster: verified cache, no network.
        echo "  Using shared cache: $shared_copy"
        src="$shared_copy"; cleanup=0
    else
        [[ -n "$shared_copy" && -f "$shared_copy" ]] && \
            echo "  (shared cache present but checksum stale — re-downloading)"
        # Download to a TEMP file, never to "$DATASETS_DIR/$ARCHIVE_NAME": a
        # locally-built archive may already live there, and cleanup on failure
        # (rm -f "$src") would destroy it.
        src="$(mktemp "${TMPDIR:-/tmp}/dynamis_archive.XXXXXX")"; cleanup=1
        got=0
        # Preferred: authenticated download via gh. Required for a PRIVATE repo —
        # release assets inherit repo visibility, so an unauthenticated wget of a
        # private asset 404s. Skipped when the user pinned an explicit public URL
        # via DYNAMIS_ARCHIVE_URL.
        if [[ -z "${DYNAMIS_ARCHIVE_URL:-}" ]] && command -v gh >/dev/null 2>&1; then
            echo "  Downloading via gh: ${ARCHIVE_REPO} ${ARCHIVE_TAG}/${ARCHIVE_NAME} ..."
            if gh release download "$ARCHIVE_TAG" -R "$ARCHIVE_REPO" \
                   -p "$ARCHIVE_NAME" -O "$src" --clobber 2>/dev/null; then
                got=1
            else
                echo "  (gh download unavailable — falling back to direct URL)"
            fi
        fi
        # Fallback: unauthenticated direct URL. Works only if the asset is PUBLIC,
        # or DYNAMIS_ARCHIVE_URL points at a public host. The interface is kept for
        # when the archive is eventually published publicly — nothing is uploaded here.
        if [[ "$got" == "0" ]]; then
            echo "  Downloading $ARCHIVE_URL ..."
            if wget -q -O "$src" "$ARCHIVE_URL"; then
                got=1
            fi
        fi
        if [[ "$got" == "0" ]]; then
            rm -f "$src"
            echo "  ERROR: could not obtain '${ARCHIVE_NAME}'."
            echo "  ${ARCHIVE_REPO} is private, so its release asset needs an authenticated"
            echo "  download. Do one of:"
            echo "    - gh auth login   (with read access to ${ARCHIVE_REPO}), then re-run"
            echo "    - publish it:  gh release create ${ARCHIVE_TAG} ${ARCHIVE_NAME} -R ${ARCHIVE_REPO} --title '...' --notes '...'"
            echo "    - override:    DYNAMIS_ARCHIVE_LOCAL=<path>  or  DYNAMIS_ARCHIVE_URL=<public-url>"
            echo "    - shared cache: put ${ARCHIVE_NAME} in ${ARCHIVE_SHARED_DIR:-<disabled>}"
            exit 1
        fi
        # Best-effort: seed the shared cache so teammates skip the download.
        # Atomic (tmp + mv); never fatal if the dir is read-only for us.
        if [[ -n "$shared_copy" ]] && archive_ok "$src"; then
            if mkdir -p "$ARCHIVE_SHARED_DIR" 2>/dev/null \
               && cp "$src" "${shared_copy}.tmp.$$" 2>/dev/null \
               && mv "${shared_copy}.tmp.$$" "$shared_copy" 2>/dev/null; then
                chmod 0664 "$shared_copy" 2>/dev/null || true
                echo "  Cached archive into shared dir: $shared_copy"
            else
                rm -f "${shared_copy}.tmp.$$" 2>/dev/null || true
            fi
        fi
    fi

    # Authoritative verify before trusting it (also catches a bad LOCAL override).
    echo "  Verifying archive checksum ..."
    if ! archive_ok "$src"; then
        echo "  ERROR: archive checksum mismatch"
        echo "    expected $expected"
        echo "    actual   $(sha256sum "$src" | awk '{print $1}')"
        [[ "$cleanup" == "1" ]] && rm -f "$src"
        exit 1
    fi
    echo "  OK: archive checksum matches"

    echo "  Extracting into $DATASETS_DIR ..."
    tar -C "$DATASETS_DIR" -xf "$src"
    [[ "$cleanup" == "1" ]] && rm -f "$src"

    # Verify extracted per-file checksums (fatal).
    verify_subset '(protein_vacuum/|protein_solvated/|membrane/|capsid/)' "extracted files" 1
    echo ""
fi

echo "Done."
