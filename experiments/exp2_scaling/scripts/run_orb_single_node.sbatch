#!/bin/bash
# ---------------------------------------------------------------------------
# Run one or more single-node ORB scaling cells (W ∈ {1, 2, 4}) sequentially
# inside a single 4-GPU GB200 node allocation.
#
# Each cell spec is colon-delimited:
#   <model>:<xyz_basename>:<world_size>:<study>:<exchanger>:<tag>:<warmup>:<timed>
# ---------------------------------------------------------------------------
#SBATCH --job-name=orb-1node
#SBATCH --time=01:30:00

set -uo pipefail

_S="${EXP2_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "$_S/_lib.sh"

exp2_check_venv
exp2_export_runtime
export CACHED_PATH_CACHE_ROOT="${EXP2_CACHED_PATH_CACHE_ROOT:-$HOME/.cache/cached_path}"

PRECISION="${EXP2_ORB_PRECISION:-float32-high}"
EDGE_CHUNK_SIZE="${EXP2_EDGE_CHUNK_SIZE:-0}"
MIXED_PRECISION="${EXP2_MIXED_PRECISION:-none}"
USE_FUSED_KERNELS="${EXP2_USE_FUSED_KERNELS:-false}"
RESAMPLE_VEL_FLAG=()
[ "${EXP2_RESAMPLE_VELOCITIES:-0}" = "1" ] && RESAMPLE_VEL_FLAG=(--resample-velocities)
HARD_TIMEOUT="${EXP2_HARD_TIMEOUT:-900}"
mkdir -p "$RESULTS_DIR"

echo "=========================================================================="
echo " exp2_scaling ORB single-node batch ($# cells)"
echo "   node      : $(hostname)   job: ${SLURM_JOB_ID:-local}"
echo "   data      : $DATA_DIR"
echo "   precision : $PRECISION   mixed: $MIXED_PRECISION   chunk: $EDGE_CHUNK_SIZE   fused: $USE_FUSED_KERNELS"
echo "   timeout   : ${HARD_TIMEOUT}s   results: $RESULTS_DIR"
echo "=========================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

write_stub() {
    local model="$1" exch="$2" study="$3" natoms="$4" ngpu="$5" tag="$6" status="$7" code="$8" xyz="$9"
    "$PY" - "$EXP_DIR/drivers" "$RESULTS_DIR" "$model" "$exch" "$study" "$natoms" "$ngpu" "$tag" "$status" "$code" "$xyz" "$PRECISION" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from bench_common import write_stub_result
outdir, model, exch, study, natoms, ngpu, tag, status, code, xyz, prec = sys.argv[2:13]
p = write_stub_result(
    output_dir=outdir, model=model, exchanger=exch, study=study,
    num_atoms=int(natoms), world_size=int(ngpu), nodes=1, gpus_per_node=4,
    tag=tag or None, status=status, exit_code=int(code),
    dataset_file=xyz, precision=prec,
)
if p:
    print(f"wrote stub {p}: {status}")
PY
}

for spec in "$@"; do
    IFS=':' read -r MODEL XYZ_BASE NGPU STUDY EXCH TAG WU TD <<< "$spec"
    XYZ="$DATA_DIR/$XYZ_BASE"
    if [ ! -f "$XYZ" ]; then
        echo "WARNING: missing dataset $XYZ — skipping"
        continue
    fi
    NATOMS="$(head -n1 "$XYZ" | tr -d '[:space:]')"
    echo
    echo "######## $MODEL | $EXCH | $STUDY | ${NATOMS} atoms | W=${NGPU} | tag=${TAG} ########"
    RC=0
    timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT" \
        "$TORCHRUN" --standalone --nproc_per_node="$NGPU" \
        "$EXP_DIR/drivers/run_orb_scaling.py" \
        --input "$XYZ" --model "$MODEL" --orb-precision "$PRECISION" \
        --edge-chunk-size "$EDGE_CHUNK_SIZE" --mixed-precision "$MIXED_PRECISION" \
        --use-fused-kernels "$USE_FUSED_KERNELS" \
        --exchanger "$EXCH" --study "$STUDY" --tag "$TAG" \
        --warmup-steps "$WU" --timed-steps "$TD" \
        --cutoff-angstrom 6.0 --ghost-buffer-angstrom 0.5 \
        --rebalance-interval 1000 --scalable-halo "${RESAMPLE_VEL_FLAG[@]}" \
        --output-dir "$RESULTS_DIR" || RC=$?

    if [ "$RC" = "124" ] || [ "$RC" = "137" ]; then
        write_stub "$MODEL" "$EXCH" "$STUDY" "$NATOMS" "$NGPU" "$TAG" timeout "$RC" "$XYZ"
    elif [ "$RC" -ne 0 ]; then
        write_stub "$MODEL" "$EXCH" "$STUDY" "$NATOMS" "$NGPU" "$TAG" error "$RC" "$XYZ"
    fi
done

echo "=== ORB single-node batch finished ==="
