#!/bin/bash
# ---------------------------------------------------------------------------
# Run one or more multi-node ORB scaling cells (N ∈ {2, 4, 8} nodes => W ∈ {8, 16, 32})
# inside one multi-node GB200 allocation (--gpus-per-node=4 --segment=<N>).
#
# Each cell spec is colon-delimited:
#   <model>:<xyz_basename>:<study>:<exchanger>:<tag>:<warmup>:<timed>
# ---------------------------------------------------------------------------
#SBATCH --job-name=orb-multinode
#SBATCH --time=01:00:00

set -uo pipefail

_S="${EXP2_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "$_S/_lib.sh"

exp2_check_venv
exp2_export_runtime
export CACHED_PATH_CACHE_ROOT="${EXP2_CACHED_PATH_CACHE_ROOT:-$HOME/.cache/cached_path}"

NNODES="${SLURM_NNODES:?must run under Slurm with --nodes >= 2}"
GPN="${EXP2_GPUS_PER_NODE:-4}"
WORLD=$(( NNODES * GPN ))
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
BASE_PORT="${EXP2_MASTER_PORT:-29530}"
PRECISION="${EXP2_ORB_PRECISION:-float32-high}"
EDGE_CHUNK_SIZE="${EXP2_EDGE_CHUNK_SIZE:-0}"
MIXED_PRECISION="${EXP2_MIXED_PRECISION:-none}"
USE_FUSED_KERNELS="${EXP2_USE_FUSED_KERNELS:-false}"
RESAMPLE_VEL_FLAG=()
[ "${EXP2_RESAMPLE_VELOCITIES:-0}" = "1" ] && RESAMPLE_VEL_FLAG=(--resample-velocities)
HARD_TIMEOUT="${EXP2_HARD_TIMEOUT:-900}"
mkdir -p "$RESULTS_DIR"

echo "=========================================================================="
echo " exp2_scaling ORB multi-node batch ($# cells)"
echo "   nodes     : $NNODES x $GPN GPU = $WORLD ranks"
echo "   nodelist  : $SLURM_JOB_NODELIST"
echo "   master    : $MASTER_ADDR:$BASE_PORT   job: ${SLURM_JOB_ID:-local}"
echo "   data      : $DATA_DIR"
echo "   precision : $PRECISION   mixed: $MIXED_PRECISION   chunk: $EDGE_CHUNK_SIZE   fused: $USE_FUSED_KERNELS"
echo "   timeout   : ${HARD_TIMEOUT}s   results: $RESULTS_DIR"
echo "=========================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

write_stub() {
    local model="$1" exch="$2" study="$3" natoms="$4" tag="$5" status="$6" code="$7" xyz="$8"
    "$PY" - "$EXP_DIR/drivers" "$RESULTS_DIR" "$model" "$exch" "$study" "$natoms" "$WORLD" "$NNODES" "$GPN" "$tag" "$status" "$code" "$xyz" "$PRECISION" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from bench_common import write_stub_result
outdir, model, exch, study, natoms, world, nodes, gpn, tag, status, code, xyz, prec = sys.argv[2:15]
p = write_stub_result(
    output_dir=outdir, model=model, exchanger=exch, study=study,
    num_atoms=int(natoms), world_size=int(world), nodes=int(nodes),
    gpus_per_node=int(gpn), tag=tag or None, status=status, exit_code=int(code),
    dataset_file=xyz, precision=prec,
)
if p:
    print(f"wrote stub {p}: {status}")
PY
}

IDX=0
for spec in "$@"; do
    IFS=':' read -r MODEL XYZ_BASE STUDY EXCH TAG WU TD <<< "$spec"
    XYZ="$DATA_DIR/$XYZ_BASE"
    if [ ! -f "$XYZ" ]; then
        echo "WARNING: missing dataset $XYZ — skipping"
        continue
    fi
    NATOMS="$(head -n1 "$XYZ" | tr -d '[:space:]')"
    PORT=$(( BASE_PORT + IDX ))
    IDX=$(( IDX + 1 ))
    RDZV="--rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${PORT} --rdzv-id=${SLURM_JOB_ID}_${IDX}"

    echo
    echo "######## $MODEL | $EXCH | $STUDY | ${NATOMS} atoms | W=${WORLD} (${NNODES}n) | tag=${TAG} ########"
    RC=0
    unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
    timeout --signal=TERM --kill-after=60 "$HARD_TIMEOUT" \
        srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=none \
        "$TORCHRUN" --nnodes="$NNODES" --nproc-per-node="$GPN" $RDZV \
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
        write_stub "$MODEL" "$EXCH" "$STUDY" "$NATOMS" "$TAG" timeout "$RC" "$XYZ"
    elif [ "$RC" -ne 0 ]; then
        write_stub "$MODEL" "$EXCH" "$STUDY" "$NATOMS" "$TAG" error "$RC" "$XYZ"
    fi
done

echo "=== ORB multi-node batch ($NNODES nodes / $WORLD GPUs) finished ==="
