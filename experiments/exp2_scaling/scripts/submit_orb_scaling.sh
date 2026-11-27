#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Matrix enumerator & Slurm submitter for ORB exp2_scaling (1..32 GB200 GPUs).
#
# Supports both Zenodo scaling datasets (downloaded via
# experiments/exp_datasets/scripts/download_zenodo.sh):
#
#   --dataset ch4_o2  (default)
#     Orb-Direct or Orb-Conservative on CH4:O2 (strong 64k/256k/1.02M,
#     weak 64k/128k per GPU, 10M-atom hero run).
#
#   --dataset li3po4
#     Orb-v3 Direct k=20 (chunked BF16 + Blackwell fused kernels by default)
#     on crystalline Li3PO4 (weak32k -> 1.01M, strong1m, strong61k).
#
# Usage:
#   scripts/submit_orb_scaling.sh --dry-run --dataset ch4_o2 --study all --model both
#   scripts/submit_orb_scaling.sh --dry-run --dataset li3po4 --study all
#   scripts/submit_orb_scaling.sh --dataset ch4_o2 --study strong --model both --exchanger gpudirect
#   scripts/submit_orb_scaling.sh --dataset li3po4 --study all --nodes-only 1
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_lib.sh"

DATASET="ch4_o2"
STUDY="all"
FAMILY="direct"
MODEL_SEL=""
EXCHANGER="gpudirect"
WEAK_DENSITY="both"
CHUNK_SIZE=""
MIXED_PREC=""
FUSED=""
NODES_ONLY=""
DRY_RUN=0
WU=10
TD=50
HERO_TD=20

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)         DATASET="$2"; shift 2 ;;
        --study)           STUDY="$2"; shift 2 ;;
        --family)          FAMILY="$2"; shift 2 ;;
        --model)           MODEL_SEL="$2"; shift 2 ;;
        --exchanger)       EXCHANGER="$2"; shift 2 ;;
        --weak-density)    WEAK_DENSITY="$2"; shift 2 ;;
        --edge-chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --mixed-precision) MIXED_PREC="$2"; shift 2 ;;
        --use-fused)       FUSED="$2"; shift 2 ;;
        --nodes-only)      NODES_ONLY="$2"; shift 2 ;;
        --warmup)          WU="$2"; shift 2 ;;
        --timed)           TD="$2"; shift 2 ;;
        --hero-timed)      HERO_TD="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,23p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$DATASET" in
    ch4_o2|li3po4) ;;
    *) echo "Invalid --dataset: $DATASET (expected ch4_o2|li3po4)" >&2; exit 2 ;;
esac

case "$FAMILY" in
    direct|conservative) ;;
    *) echo "Invalid --family: $FAMILY (expected direct|conservative)" >&2; exit 2 ;;
esac

# Dataset-specific defaults
if [ "$DATASET" = "li3po4" ]; then
    MODEL_SEL="${MODEL_SEL:-20}"
    CHUNK_SIZE="${CHUNK_SIZE:-524288}"
    MIXED_PREC="${MIXED_PREC:-bf16}"
    FUSED="${FUSED:-true}"
    export EXP2_DATA_DIR="${EXP2_DATA_DIR:-$EXP2_DATASETS_ROOT/li3po4}"
    export EXP2_RESULTS_DIR="${EXP2_RESULTS_DIR:-$EXP_DIR/results_li3po4}"
    # li3po4_*.xyz carries momenta in amu*A/fs rather than ASE units; zero before
    # Simulator.setup() so Dynamis samples 300 K Maxwell-Boltzmann velocities natively.
    export EXP2_RESAMPLE_VELOCITIES="${EXP2_RESAMPLE_VELOCITIES:-1}"
else
    MODEL_SEL="${MODEL_SEL:-both}"
    CHUNK_SIZE="${CHUNK_SIZE:-0}"
    MIXED_PREC="${MIXED_PREC:-none}"
    FUSED="${FUSED:-false}"
    export EXP2_DATA_DIR="${EXP2_DATA_DIR:-$EXP2_DATASETS_ROOT/ch4_o2}"
    export EXP2_RESULTS_DIR="${EXP2_RESULTS_DIR:-$EXP_DIR/results}"
    export EXP2_RESAMPLE_VELOCITIES="${EXP2_RESAMPLE_VELOCITIES:-0}"
fi

export EXP2_EDGE_CHUNK_SIZE="$CHUNK_SIZE"
export EXP2_MIXED_PRECISION="$MIXED_PREC"
export EXP2_USE_FUSED_KERNELS="$FUSED"
export EXP2_DRY_RUN="$DRY_RUN"

if [ "$DRY_RUN" = "0" ]; then
    exp2_require_dataset "$DATASET" "$EXP2_DATA_DIR"
fi

case "$MODEL_SEL" in
    inf)  MODELS=("orb_v3_${FAMILY}_inf_omat") ;;
    20)   MODELS=("orb_v3_${FAMILY}_20_omat") ;;
    both) MODELS=("orb_v3_${FAMILY}_inf_omat" "orb_v3_${FAMILY}_20_omat") ;;
    orb_v3_*) MODELS=("$MODEL_SEL") ;;
    *)    echo "Invalid --model: $MODEL_SEL (expected inf|20|both|orb_v3_*)" >&2; exit 2 ;;
esac

CELLS_1N=()
CELLS_2N=()
CELLS_4N=()
CELLS_8N=()

add_1n() { CELLS_1N+=("$1"); }
add_2n() { CELLS_2N+=("$1"); }
add_4n() { CELLS_4N+=("$1"); }
add_8n() { CELLS_8N+=("$1"); }

for M in "${MODELS[@]}"; do
    if [ "$DATASET" = "li3po4" ]; then
        # --- Li3PO4 (Zenodo li3po4.zip: 32K -> 1.01M atoms) ---
        if [ "$STUDY" = "all" ] || [ "$STUDY" = "weak" ]; then
            add_1n "${M}:li3po4_32032.xyz:1:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            add_1n "${M}:li3po4_60928.xyz:2:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            add_1n "${M}:li3po4_126720.xyz:4:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            add_2n "${M}:li3po4_256256.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            add_4n "${M}:li3po4_501760.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            add_8n "${M}:li3po4_1010240.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
        fi
        if [ "$STUDY" = "all" ] || [ "$STUDY" = "strong" ] || [ "$STUDY" = "strong1m" ]; then
            for W in 1 2 4; do add_1n "${M}:li3po4_1010240.xyz:${W}:strong:${EXCHANGER}:strong1m:${WU}:${TD}"; done
            add_2n "${M}:li3po4_1010240.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
            add_4n "${M}:li3po4_1010240.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
            add_8n "${M}:li3po4_1010240.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
        fi
        if [ "$STUDY" = "all" ] || [ "$STUDY" = "strong" ] || [ "$STUDY" = "strong61k" ]; then
            for W in 1 2 4; do add_1n "${M}:li3po4_60928.xyz:${W}:strong:${EXCHANGER}:strong61k:${WU}:${TD}"; done
            add_2n "${M}:li3po4_60928.xyz:strong:${EXCHANGER}:strong61k:${WU}:${TD}"
            add_4n "${M}:li3po4_60928.xyz:strong:${EXCHANGER}:strong61k:${WU}:${TD}"
            add_8n "${M}:li3po4_60928.xyz:strong:${EXCHANGER}:strong61k:${WU}:${TD}"
        fi
    elif [ "$FAMILY" = "direct" ]; then
        # --- CH4:O2 Orb-Direct ---
        if [ "$STUDY" = "strong" ] || [ "$STUDY" = "all" ]; then
            for W in 1 2 4; do add_1n "${M}:ch4_63999_splited.xyz:${W}:strong:${EXCHANGER}:strong64k:${WU}:${TD}"; done
            add_2n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"
            add_4n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"
            add_8n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"

            for W in 1 2 4; do add_1n "${M}:ch4_255996_splited.xyz:${W}:strong:${EXCHANGER}:strong256k:${WU}:${TD}"; done
            add_2n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
            add_4n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
            add_8n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"

            add_1n "${M}:ch4_1024002_splited.xyz:4:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
            add_2n "${M}:ch4_1024002_splited.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
            add_4n "${M}:ch4_1024002_splited.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
            add_8n "${M}:ch4_1024002_splited.xyz:strong:${EXCHANGER}:strong1m:${WU}:${TD}"
        fi
        if [ "$STUDY" = "weak" ] || [ "$STUDY" = "all" ]; then
            if [ "$WEAK_DENSITY" = "64k" ] || [ "$WEAK_DENSITY" = "both" ]; then
                add_1n "${M}:ch4_63999_splited.xyz:1:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_1n "${M}:ch4_127998_splited.xyz:2:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_1n "${M}:ch4_255996_splited.xyz:4:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_2n "${M}:ch4_512001_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_4n "${M}:ch4_1024002_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_8n "${M}:ch4_2048004_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
            fi
            if [ "$WEAK_DENSITY" = "128k" ] || [ "$WEAK_DENSITY" = "both" ]; then
                add_1n "${M}:ch4_127998_splited.xyz:1:weak:${EXCHANGER}:weak128k:${WU}:${TD}"
                add_1n "${M}:ch4_255996_splited.xyz:2:weak:${EXCHANGER}:weak128k:${WU}:${TD}"
                add_1n "${M}:ch4_512001_splited.xyz:4:weak:${EXCHANGER}:weak128k:${WU}:${TD}"
                add_2n "${M}:ch4_1024002_splited.xyz:weak:${EXCHANGER}:weak128k:${WU}:${TD}"
                add_4n "${M}:ch4_2048004_splited.xyz:weak:${EXCHANGER}:weak128k:${WU}:${TD}"
                add_8n "${M}:ch4_4095999_splited.xyz:weak:${EXCHANGER}:weak128k:${WU}:${HERO_TD}"
            fi
        fi
        if [ "$STUDY" = "hero" ]; then
            add_8n "${M}:ch4_9999999_splited.xyz:hero:${EXCHANGER}:hero10m:${WU}:${HERO_TD}"
        fi
    else
        # --- CH4:O2 Orb-Conservative ---
        if [ "$STUDY" = "strong" ] || [ "$STUDY" = "all" ]; then
            for W in 1 2 4; do add_1n "${M}:ch4_32004_splited.xyz:${W}:strong:${EXCHANGER}:strong32k:${WU}:${TD}"; done
            add_2n "${M}:ch4_32004_splited.xyz:strong:${EXCHANGER}:strong32k:${WU}:${TD}"
            add_4n "${M}:ch4_32004_splited.xyz:strong:${EXCHANGER}:strong32k:${WU}:${TD}"
            add_8n "${M}:ch4_32004_splited.xyz:strong:${EXCHANGER}:strong32k:${WU}:${TD}"

            if [ "$M" = "orb_v3_conservative_inf_omat" ]; then
                for W in 1 2 4; do add_1n "${M}:ch4_50004_splited.xyz:${W}:strong:${EXCHANGER}:strong50k:${WU}:${TD}"; done
                add_2n "${M}:ch4_50004_splited.xyz:strong:${EXCHANGER}:strong50k:${WU}:${TD}"
                add_4n "${M}:ch4_50004_splited.xyz:strong:${EXCHANGER}:strong50k:${WU}:${TD}"
                add_8n "${M}:ch4_50004_splited.xyz:strong:${EXCHANGER}:strong50k:${WU}:${TD}"

                add_1n "${M}:ch4_127998_splited.xyz:4:strong:${EXCHANGER}:strong128k:${WU}:${TD}"
                add_2n "${M}:ch4_127998_splited.xyz:strong:${EXCHANGER}:strong128k:${WU}:${TD}"
                add_4n "${M}:ch4_127998_splited.xyz:strong:${EXCHANGER}:strong128k:${WU}:${TD}"
                add_8n "${M}:ch4_127998_splited.xyz:strong:${EXCHANGER}:strong128k:${WU}:${TD}"
            else
                for W in 1 2 4; do add_1n "${M}:ch4_63999_splited.xyz:${W}:strong:${EXCHANGER}:strong64k:${WU}:${TD}"; done
                add_2n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"
                add_4n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"
                add_8n "${M}:ch4_63999_splited.xyz:strong:${EXCHANGER}:strong64k:${WU}:${TD}"

                add_1n "${M}:ch4_255996_splited.xyz:4:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
                add_2n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
                add_4n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
                add_8n "${M}:ch4_255996_splited.xyz:strong:${EXCHANGER}:strong256k:${WU}:${TD}"
            fi
        fi
        if [ "$STUDY" = "weak" ] || [ "$STUDY" = "all" ]; then
            if [ "$WEAK_DENSITY" = "32k" ] || [ "$WEAK_DENSITY" = "both" ]; then
                add_1n "${M}:ch4_32004_splited.xyz:1:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
                add_1n "${M}:ch4_63999_splited.xyz:2:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
                add_1n "${M}:ch4_127998_splited.xyz:4:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
                add_2n "${M}:ch4_255996_splited.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
                add_4n "${M}:ch4_512001_splited.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
                add_8n "${M}:ch4_1024002_splited.xyz:weak:${EXCHANGER}:weak32k:${WU}:${TD}"
            fi
            if [ "$M" = "orb_v3_conservative_20_omat" ] && { [ "$WEAK_DENSITY" = "64k" ] || [ "$WEAK_DENSITY" = "both" ]; }; then
                add_1n "${M}:ch4_63999_splited.xyz:1:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_1n "${M}:ch4_127998_splited.xyz:2:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_1n "${M}:ch4_255996_splited.xyz:4:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_2n "${M}:ch4_512001_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_4n "${M}:ch4_1024002_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${TD}"
                add_8n "${M}:ch4_2048004_splited.xyz:weak:${EXCHANGER}:weak64k:${WU}:${HERO_TD}"
            fi
        fi
        if [ "$STUDY" = "hero" ]; then
            add_8n "${M}:ch4_2048004_splited.xyz:hero:${EXCHANGER}:hero2m:${WU}:${HERO_TD}"
        fi
    fi
done

submit_batch() {
    local nodes="$1" jobname="$2" walltime="$3" script="$4"
    shift 4
    [ $# -eq 0 ] && return 0
    if [ -n "$NODES_ONLY" ] && [ "$NODES_ONLY" != "$nodes" ]; then
        return 0
    fi
    echo ">>> Submitting $jobname ($nodes nodes, $# cells, time=$walltime)"
    "$SCRIPT_DIR/sbatch_submit.sh" \
        --nodes="$nodes" --segment="$nodes" \
        --job-name="$jobname" --time="$walltime" \
        "$script" "$@"
}

JOB_PREFIX="orb-${DATASET}-${FAMILY}-${STUDY}"
submit_batch 1 "${JOB_PREFIX}-1n" "01:30:00" "$SCRIPT_DIR/run_orb_single_node.sbatch" "${CELLS_1N[@]}"
submit_batch 2 "${JOB_PREFIX}-2n" "01:00:00" "$SCRIPT_DIR/run_orb_multi_node.sbatch"  "${CELLS_2N[@]}"
submit_batch 4 "${JOB_PREFIX}-4n" "01:00:00" "$SCRIPT_DIR/run_orb_multi_node.sbatch"  "${CELLS_4N[@]}"
submit_batch 8 "${JOB_PREFIX}-8n" "01:00:00" "$SCRIPT_DIR/run_orb_multi_node.sbatch"  "${CELLS_8N[@]}"
