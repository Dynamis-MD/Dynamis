#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_zenodo.sh — Download, MD5-verify, and extract benchmark datasets
# from Zenodo record 24487955 ("Dynamis Experiments").
#
# All archives are extracted under experiments/exp_datasets/zenodo/<name>/
# (overridable via --dest or DYNAMIS_ZENODO_DIR) so every model's scaling and
# end-to-end benchmarks (ORB, UMA, MACE, ...) share a single verified copy.
#
# Usage:
#   bash download_zenodo.sh --list                     # show available archives
#   bash download_zenodo.sh ch4_o2 li3po4              # fetch specific archives
#   bash download_zenodo.sh --scaling                  # shorthand for ch4_o2 + li3po4
#   bash download_zenodo.sh --all                      # fetch all 8 archives (~736 MB)
#   bash download_zenodo.sh --dest /path/to/dir li3po4 # custom target directory
#   bash download_zenodo.sh --verify-only ch4_o2       # verify existing zip(s) only
#
# Shared cache (optional, avoids repeated downloads on multi-user clusters):
#   export DYNAMIS_ZENODO_CACHE_DIR=/shared/cache/dynamis_zenodo
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ZENODO_RECORD="${DYNAMIS_ZENODO_RECORD:-24487955}"
ZENODO_BASE_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files"
DEST_DIR="${DYNAMIS_ZENODO_DIR:-$DATASETS_DIR/zenodo}"
CACHE_DIR="${DYNAMIS_ZENODO_CACHE_DIR:-}"

# Pinned archive manifest (Zenodo record 24487955):
#   name|md5|approx_size|description
ARCHIVE_MANIFEST=(
    "ch4_o2|6f7ada6b8e541378600f8f4b0eaa4efb|445 MB|CH4:O2 (1:2) split scaling ladder (~8k to ~10M atoms)"
    "li3po4|c1fbf4761eec67db00bbef2b2d1de089|42 MB|Crystalline Li3PO4 (mp-13725) supercells (32k to 1.01M atoms)"
    "2w49|5103749a89457801d80481f225b269e9|28 MB|PDB 2W49 protein (solvated, vacuum, strained vacuum)"
    "apoa1|4be86e341b0764112cf07657affb32b6|1.0 MB|ApoA1 in vacuum (OpenMM-prepared)"
    "CH4_450|ca17b46b8fc5ccafdeba5bf1456e9797|17 KB|Methane box (~450 atoms, 298 K)"
    "CH4_11250|8a6e80d4813078d3dd2f9f2ea5cfbbac|402 KB|Methane box (~11.25k atoms, 298 K)"
    "CH4_1125000|58d06749ab0b79c3ffd09574bfdf92fb|42 MB|Methane box (~1.125M atoms, 298 K)"
    "RP3_1320|8fcf6925a71380aac6e73f4745a9e04d|178 MB|RP-3 jet fuel MD trajectory (md_RP3_jetfuel_3000.xyz)"
)

ALL_NAMES=()
declare -A ARCHIVE_MD5
declare -A ARCHIVE_SIZE
declare -A ARCHIVE_DESC
for entry in "${ARCHIVE_MANIFEST[@]}"; do
    IFS='|' read -r _name _md5 _sz _desc <<< "$entry"
    ALL_NAMES+=("$_name")
    ARCHIVE_MD5["$_name"]="$_md5"
    ARCHIVE_SIZE["$_name"]="$_sz"
    ARCHIVE_DESC["$_name"]="$_desc"
done

usage() {
    sed -n '3,21p' "$0"
}

list_archives() {
    echo "Zenodo record ${ZENODO_RECORD} (https://zenodo.org/records/${ZENODO_RECORD}):"
    printf "  %-14s %-9s %-34s %s\n" "ARCHIVE" "SIZE" "MD5" "DESCRIPTION"
    for name in "${ALL_NAMES[@]}"; do
        printf "  %-14s %-9s %-34s %s\n" \
            "$name" "${ARCHIVE_SIZE[$name]}" "${ARCHIVE_MD5[$name]}" "${ARCHIVE_DESC[$name]}"
    done
}

md5_of() {
    local file="$1"
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$file" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$file"
    else
        echo "ERROR: neither md5sum nor md5 is installed" >&2
        exit 1
    fi
}

fetch_url() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --waitretry=5 -O "$out" "$url"
    else
        echo "ERROR: neither curl nor wget is installed" >&2
        exit 1
    fi
}

TARGETS=()
VERIFY_ONLY=0
NO_EXTRACT=0
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --list|-l)
            list_archives
            exit 0
            ;;
        --all)
            TARGETS=("${ALL_NAMES[@]}")
            shift
            ;;
        --scaling)
            TARGETS+=("ch4_o2" "li3po4")
            shift
            ;;
        --dest)
            DEST_DIR="$2"
            shift 2
            ;;
        --dest=*)
            DEST_DIR="${1#--dest=}"
            shift
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --cache-dir=*)
            CACHE_DIR="${1#--cache-dir=}"
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        --no-extract)
            NO_EXTRACT=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            # Allow passing either "li3po4" or "li3po4.zip"
            TARGETS+=("${1%.zip}")
            shift
            ;;
    esac
done

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "ERROR: specify one or more archive names (e.g. ch4_o2 li3po4), --scaling, or --all." >&2
    echo "Run '$0 --list' to see available Zenodo archives." >&2
    exit 2
fi

# Deduplicate targets while preserving order and validating names
UNIQUE_TARGETS=()
declare -A _SEEN
for name in "${TARGETS[@]}"; do
    if [ -z "${ARCHIVE_MD5[$name]+x}" ]; then
        echo "ERROR: unknown Zenodo archive '$name'." >&2
        list_archives >&2
        exit 2
    fi
    if [ -z "${_SEEN[$name]+x}" ]; then
        _SEEN["$name"]=1
        UNIQUE_TARGETS+=("$name")
    fi
done

mkdir -p "$DEST_DIR"

verify_zip() {
    local path="$1" expected="$2"
    [ -f "$path" ] && [ "$(md5_of "$path")" = "$expected" ]
}

process_archive() {
    local name="$1"
    local expected="${ARCHIVE_MD5[$name]}"
    local size_hint="${ARCHIVE_SIZE[$name]}"
    local zip_name="${name}.zip"
    local zip_path="$DEST_DIR/$zip_name"
    local extract_dir="$DEST_DIR/$name"
    local url="${ZENODO_BASE_URL}/${zip_name}/content"

    echo ">>> [$name] (${size_hint}) — ${ARCHIVE_DESC[$name]}"

    if [ "$VERIFY_ONLY" = "1" ]; then
        if verify_zip "$zip_path" "$expected"; then
            echo "  OK: $zip_path matches MD5 $expected"
            return 0
        else
            echo "  FAIL: $zip_path missing or MD5 mismatch (expected $expected)" >&2
            return 1
        fi
    fi

    # 1. Obtain verified zip (existing local -> shared cache -> Zenodo download)
    if [ "$FORCE" = "0" ] && verify_zip "$zip_path" "$expected"; then
        echo "  Using existing verified archive: $zip_path"
    elif [ -n "$CACHE_DIR" ] && verify_zip "$CACHE_DIR/$zip_name" "$expected"; then
        echo "  Using shared cache: $CACHE_DIR/$zip_name"
        cp "$CACHE_DIR/$zip_name" "${zip_path}.tmp.$$"
        mv "${zip_path}.tmp.$$" "$zip_path"
    else
        echo "  Downloading $url (~${size_hint})..."
        local tmp_zip="${zip_path}.tmp.$$"
        trap 'rm -f "$tmp_zip"' RETURN
        fetch_url "$url" "$tmp_zip"
        local actual
        actual="$(md5_of "$tmp_zip")"
        if [ "$actual" != "$expected" ]; then
            echo "  ERROR: MD5 mismatch for $zip_name" >&2
            echo "    expected: $expected" >&2
            echo "    actual  : $actual" >&2
            rm -f "$tmp_zip"
            exit 1
        fi
        mv "$tmp_zip" "$zip_path"
        trap - RETURN
        echo "  MD5 verified: $expected"

        # Best-effort atomic seed into shared cache if configured
        if [ -n "$CACHE_DIR" ]; then
            local cached="$CACHE_DIR/$zip_name"
            if mkdir -p "$CACHE_DIR" 2>/dev/null \
               && cp "$zip_path" "${cached}.tmp.$$" 2>/dev/null \
               && mv "${cached}.tmp.$$" "$cached" 2>/dev/null; then
                chmod 0664 "$cached" 2>/dev/null || true
                echo "  Cached into shared dir: $cached"
            else
                rm -f "${cached}.tmp.$$" 2>/dev/null || true
            fi
        fi
    fi

    if [ "$NO_EXTRACT" = "1" ]; then
        return 0
    fi

    # 2. Extract into DEST_DIR (archives contain top-level <name>/ directory)
    if ! command -v unzip >/dev/null 2>&1; then
        echo "  ERROR: 'unzip' is required to extract $zip_path" >&2
        exit 1
    fi

    local existing_xyz=0
    if [ -d "$extract_dir" ]; then
        existing_xyz="$(find "$extract_dir" -maxdepth 2 -name '*.xyz' 2>/dev/null | wc -l | tr -d '[:space:]')"
    fi

    if [ "$FORCE" = "0" ] && [ "$existing_xyz" -gt 0 ]; then
        echo "  Already extracted in $extract_dir (${existing_xyz} .xyz files; pass --force to re-extract)"
    else
        echo "  Extracting $zip_path -> $DEST_DIR/"
        unzip -o -q "$zip_path" -d "$DEST_DIR"
    fi

    # 3. Print inventory of extracted .xyz structures
    if [ -d "$extract_dir" ]; then
        local xyz_files
        xyz_files="$(find "$extract_dir" -maxdepth 2 -name '*.xyz' | sort -V)"
        if [ -n "$xyz_files" ]; then
            while IFS= read -r f; do
                local natoms sz
                natoms="$(head -n1 "$f" | tr -d '[:space:]')"
                sz="$(du -h "$f" | awk '{print $1}')"
                printf "    %-36s %10s atoms  (%s)\n" "$(basename "$f")" "$natoms" "$sz"
            done <<< "$xyz_files"
        fi
    fi
}

for target in "${UNIQUE_TARGETS[@]}"; do
    process_archive "$target"
    echo ""
done

echo "Done. Zenodo datasets directory: $DEST_DIR"
