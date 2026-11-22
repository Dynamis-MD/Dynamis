#!/bin/bash
# ---------------------------------------------------------------------------
# Obtain every exp0 dataset (synthetic + real-world) on any cluster.
#
#   scripts/sbatch_exp0.sh --cpu scripts/setup_datasets.sh      # as a Slurm job
#   bash scripts/setup_datasets.sh                              # or directly
#
# Account / partition / mem are injected by scripts/sbatch_exp0.sh from
# scripts/cluster.env (override in scripts/cluster.local.env or via EXP0_*).
# This job carries NO site-specific #SBATCH lines.
#
# Four tiers (select a subset with EXP0_DATA_TIERS, default "1 2 3 4"):
#   1  baselines/ch4_*_splited.xyz   GENERATED (numpy only) -- bit-reproducible
#   2  pdb/*.pdb  *.cif              DOWNLOADED from RCSB          [needs network]
#   3  protein_vacuum/ membrane/ protein_solvated/dhfr capsid/
#                                    the hosted checksummed tar.xz  [needs network
#                                    OR DISTNNP_ARCHIVE_LOCAL / _SHARED_DIR]
#   4  protein_solvated/*_density.xyz + electrochemical/*_density.xyz
#                                    GENERATED from Tier 2/3 inputs (numpy+scipy+ase)
#
# NETWORK NOTE: Tiers 2 and 3 fetch from the internet. If this runs on a compute
# node with no outbound network, either (a) run `EXP0_DATA_TIERS="2 3" bash
# scripts/setup_datasets.sh` once on a login node first, or (b) stage the archive
# and set DISTNNP_ARCHIVE_LOCAL=<path> (see scripts/cluster.env for the knobs).
# Tiers 1 and 4 are pure compute and always work on a batch node.
#
#SBATCH --job-name=exp0-setup-datasets
#SBATCH --time=00:40:00

set -euo pipefail

# Slurm runs this from a spool copy, so ${BASH_SOURCE[0]} is not scripts/.
# scripts/sbatch_exp0.sh exports EXP0_SCRIPTS_DIR; fall back to $0's dir when
# the script is run directly (login node, no Slurm).
_S="${EXP0_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$_S/_lib.sh" ] || { echo "ERROR: scripts/_lib.sh not found in '$_S'"; exit 1; }
source "$_S/_lib.sh"

DATA_DIR="$EXP_DIR/exp_datasets"
DS="$DATA_DIR/scripts"
TIERS="${EXP0_DATA_TIERS:-1 2 3 4}"

# CH4:O2 sizes (thousands of atoms). Default covers smoke (1,2) + the isotropic
# control for the real-systems comparison (32,64,128). Override with EXP0_DATA_CH4_SIZES.
CH4_SIZES="${EXP0_DATA_CH4_SIZES:-1 2 32 64 128}"
# Tier 4 solvated proteins. 9gmo first -- it is the headline 1.5M-atom system.
# "" disables. 9gmo takes ~10 min (1.5M atoms).
SOLVATED="${EXP0_DATA_SOLVATED:-9gmo 2rik 1aki}"
# Tier 4 Pt(111) + water. "" disables.
ELECTROCHEM="${EXP0_DATA_ELECTROCHEM:-12k 17k}"

exp0_load_modules

# Python for the numpy/scipy/ase generators (Tiers 1 + 4). Prefer the exp0 venv;
# fall back to any python3 on PATH that has numpy.
GEN_PY="$PY"
if [ ! -x "$GEN_PY" ]; then
    GEN_PY="$(command -v python3 || true)"
fi
[ -n "$GEN_PY" ] || { echo "ERROR: no python found for the generators"; exit 1; }
if ! "$GEN_PY" -c 'import numpy' 2>/dev/null; then
    echo "ERROR: '$GEN_PY' has no numpy. Build the venv (scripts/env_setup.sh)"
    echo "       or set EXP0_VENV / put a numpy-capable python3 on PATH."
    exit 1
fi
HAVE_SCIPY_ASE=1
"$GEN_PY" -c 'import scipy, ase' 2>/dev/null || HAVE_SCIPY_ASE=0

echo "======================================================"
echo "  exp0 dataset setup"
echo "  tiers      : $TIERS"
echo "  python     : $GEN_PY"
echo "  dest       : $DATA_DIR"
echo "======================================================"

has_tier() { case " $TIERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ── Tier 1: CH4/O2 baselines (generate) ─────────────────────────────────
if has_tier 1; then
    echo ""; echo ">>> Tier 1: CH4/O2 baselines  (sizes: $CH4_SIZES)"
    # shellcheck disable=SC2086
    "$GEN_PY" "$DS/03_generate_ch4.py" --sizes $CH4_SIZES --variants splited \
        --out-dir "$DATA_DIR/baselines"
    _want="ch4_($(echo "$CH4_SIZES" | tr ' ' '|'))k_"
    _lines="$(grep -E '^[0-9a-f]{64}  baselines/' "$DS/checksums.txt" | grep -E "$_want" || true)"
    if [ -n "$_lines" ]; then
        ( cd "$DATA_DIR" && echo "$_lines" | sha256sum -c - ) \
            || echo "  WARN: a pinned CH4 size failed its checksum"
    fi
    echo "  (sizes not in checksums.txt are still bit-reproducible from seed 42)"
fi

# ── Tier 2 + 3: downloads (RCSB PDBs + hosted archive) ──────────────────
if has_tier 2 && has_tier 3; then
    echo ""; echo ">>> Tier 2 + 3: RCSB PDBs + hosted protein/membrane/capsid archive"
    bash "$DS/download_datasets.sh"
elif has_tier 2; then
    echo ""; echo ">>> Tier 2: RCSB PDBs only"
    bash "$DS/download_datasets.sh" --pdb-only
elif has_tier 3; then
    echo ""; echo ">>> Tier 3: hosted archive only"
    bash "$DS/download_datasets.sh" --archive-only
fi

# ── Tier 4: solvated proteins + electrochemical (generate) ──────────────
if has_tier 4; then
    echo ""; echo ">>> Tier 4: generated solvated / electrochemical systems"
    if [ "$HAVE_SCIPY_ASE" != 1 ]; then
        echo "  SKIP: '$GEN_PY' is missing scipy and/or ase (needed by 04/06)."
        echo "        Build the venv or: pip install scipy ase"
    else
        if [ -n "$SOLVATED" ]; then
            # shellcheck disable=SC2086
            "$GEN_PY" "$DS/04_generate_solvated.py" --systems $SOLVATED
        fi
        if [ -n "$ELECTROCHEM" ]; then
            # shellcheck disable=SC2086
            "$GEN_PY" "$DS/06_generate_electrochemical.py" --systems $ELECTROCHEM
        fi
    fi
fi

# ── summary ────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Dataset inventory"
echo "======================================================"
for d in baselines pdb protein_vacuum protein_solvated membrane electrochemical capsid; do
    dir="$DATA_DIR/$d"
    [ -d "$dir" ] || continue
    n=$(find "$dir" \( -name '*.xyz' -o -name '*.pdb' -o -name '*.cif' \) 2>/dev/null | wc -l)
    sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
    printf "  %-18s %3d files  %s\n" "$d/" "$n" "$sz"
done
echo ""
echo "DONE."
