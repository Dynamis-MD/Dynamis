#!/bin/bash
# ---------------------------------------------------------------------------
# ORB preflight verification gates (submit via scripts/sbatch_submit.sh):
#
#   parity          (1 node, --test):  test_orb_model.py + test_max_neighbors.py
#   ceiling         (1 node):          1-GPU HBM ceiling for Orb-Conservative on CH4:O2
#   ceiling-direct  (1 node):          1-GPU HBM ceiling for Orb-Direct on CH4:O2
#   mnnvl           (2 nodes):         verify MNNVL GPUDirect Fabric vs NCCL across 2 nodes
#   li3po4-fused    (1 node, --test):  Li3PO4 32K & 1.01M (W=1,4) chunked BF16 fused gate
#
# Usage:
#   scripts/sbatch_submit.sh --test --time=00:15:00 --job-name=orb-parity scripts/probe_orb_gate.sbatch parity
#   scripts/sbatch_submit.sh --time=00:20:00 --job-name=orb-ceiling scripts/probe_orb_gate.sbatch ceiling-direct
#   scripts/sbatch_submit.sh --nodes=2 --segment=2 --time=00:15:00 --job-name=orb-mnnvl scripts/probe_orb_gate.sbatch mnnvl
#   scripts/sbatch_submit.sh --test --time=00:20:00 --job-name=li3po4-gate scripts/probe_orb_gate.sbatch li3po4-fused
# ---------------------------------------------------------------------------
#SBATCH --job-name=orb-probe
#SBATCH --time=00:20:00

set -uo pipefail

_S="${EXP2_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "$_S/_lib.sh"

exp2_check_venv
exp2_export_runtime
export CACHED_PATH_CACHE_ROOT="${EXP2_CACHED_PATH_CACHE_ROOT:-$HOME/.cache/cached_path}"

MODE="${1:-ceiling-direct}"
if [ "$MODE" = "li3po4-fused" ]; then
    DATA_DIR="${EXP2_DATA_DIR:-$EXP2_DATASETS_ROOT/li3po4}"
    RESULTS_DIR="${EXP2_RESULTS_DIR:-$EXP_DIR/results_li3po4}"
fi
PROBE_DIR="$RESULTS_DIR/probes"
mkdir -p "$PROBE_DIR"

echo "=========================================================================="
echo " exp2_scaling ORB probe gate: MODE=$MODE"
echo "   nodes    : ${SLURM_NNODES:-1}   nodelist: ${SLURM_JOB_NODELIST:-local}"
echo "   venv     : $VENV"
echo "   data_dir : $DATA_DIR"
echo "=========================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
  parity)
    echo "=== Running distributed ORB parity tests (W=2) ==="
    cd "$REPO_ROOT"
    "$TORCHRUN" --standalone --nproc_per_node=2 -m pytest -v -s dynamis_model/tests/test_orb_model.py
    echo "=== Running max_num_neighbors unit tests ==="
    "$PY" -m pytest -v dynamis_state/tests/test_max_neighbors.py
    ;;

  ceiling)
    exp2_require_dataset ch4_o2 "$DATA_DIR"
    echo "=== Probing 1-GPU HBM ceiling for Orb-Conservative (CH4:O2) ==="
    for spec in \
      "orb_v3_conservative_inf_omat:ch4_32004_splited.xyz" \
      "orb_v3_conservative_inf_omat:ch4_63999_splited.xyz" \
      "orb_v3_conservative_20_omat:ch4_32004_splited.xyz" \
      "orb_v3_conservative_20_omat:ch4_63999_splited.xyz" \
      "orb_v3_conservative_20_omat:ch4_127998_splited.xyz"; do
        IFS=':' read -r model sz <<< "$spec"
        echo "--- probe: $model $sz W=1 ---"
        timeout --signal=TERM --kill-after=30 300 \
            "$TORCHRUN" --standalone --nproc_per_node=1 \
            "$EXP_DIR/drivers/run_orb_scaling.py" \
            --input "$DATA_DIR/$sz" --model "$model" --orb-precision float32-high \
            --exchanger gpudirect --study baseline --tag ceiling \
            --warmup-steps 3 --timed-steps 10 --output-dir "$PROBE_DIR" || true
    done
    ;;

  ceiling-direct)
    exp2_require_dataset ch4_o2 "$DATA_DIR"
    echo "=== Probing 1-GPU HBM ceiling for Orb-Direct (CH4:O2) ==="
    for spec in \
      "orb_v3_direct_inf_omat:ch4_63999_splited.xyz" \
      "orb_v3_direct_inf_omat:ch4_127998_splited.xyz" \
      "orb_v3_direct_inf_omat:ch4_255996_splited.xyz" \
      "orb_v3_direct_inf_omat:ch4_512001_splited.xyz" \
      "orb_v3_direct_20_omat:ch4_63999_splited.xyz" \
      "orb_v3_direct_20_omat:ch4_127998_splited.xyz" \
      "orb_v3_direct_20_omat:ch4_255996_splited.xyz" \
      "orb_v3_direct_20_omat:ch4_512001_splited.xyz" \
      "orb_v3_direct_20_omat:ch4_1024002_splited.xyz"; do
        IFS=':' read -r model sz <<< "$spec"
        echo "--- probe: $model $sz W=1 ---"
        timeout --signal=TERM --kill-after=30 300 \
            "$TORCHRUN" --standalone --nproc_per_node=1 \
            "$EXP_DIR/drivers/run_orb_scaling.py" \
            --input "$DATA_DIR/$sz" --model "$model" --orb-precision float32-high \
            --exchanger gpudirect --study baseline --tag ceiling_direct \
            --warmup-steps 3 --timed-steps 10 --output-dir "$PROBE_DIR" || true
    done
    ;;

  mnnvl)
    exp2_require_dataset ch4_o2 "$DATA_DIR"
    NNODES="${SLURM_NNODES:-2}"
    GPN="${EXP2_GPUS_PER_NODE:-4}"
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    MASTER_PORT="${EXP2_MASTER_PORT:-29521}"
    XYZ="$DATA_DIR/ch4_63999_splited.xyz"

    echo "=== Checking CUDA Fabric capability + IMEX across $NNODES nodes ($SLURM_JOB_NODELIST) ==="
    unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
    srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=none "$PY" -c '
import socket, torch, dynamis_model
from dynamis_model import FeatureExchanger
print(socket.gethostname(), "gpus=", torch.cuda.device_count(),
      "supports_fabric=", FeatureExchanger.supports_fabric_handles(0), flush=True)
'

    for exch in gpudirect nccl; do
        echo "=== 2-node ORB test: exchanger=$exch model=orb_v3_direct_20_omat ==="
        if [ "$exch" = "nccl" ]; then PORT=$(( MASTER_PORT + 1 )); else PORT="$MASTER_PORT"; fi
        timeout --signal=TERM --kill-after=30 300 \
            srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=none \
            "$TORCHRUN" --nnodes="$NNODES" --nproc-per-node="$GPN" \
                --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${PORT}" \
                --rdzv-id="${SLURM_JOB_ID}_${exch}" \
                "$EXP_DIR/drivers/run_orb_scaling.py" \
                --input "$XYZ" --model orb_v3_direct_20_omat \
                --orb-precision float32-high --exchanger "$exch" \
                --study baseline --tag "mnnvl_${exch}" \
                --warmup-steps 5 --timed-steps 15 --output-dir "$PROBE_DIR" || echo "FAILED: $exch (rc=$?)"
    done
    ;;

  li3po4-fused)
    exp2_require_dataset li3po4 "$DATA_DIR"
    echo "=== Gate 1: test_orb_model.py ==="
    (cd "$REPO_ROOT" && "$PY" -m pytest -v dynamis_model/tests/test_orb_model.py)

    for spec in \
      "li3po4_32032.xyz:1:gate_li3po4_32k" \
      "li3po4_32032.xyz:4:gate_li3po4_32k" \
      "li3po4_1010240.xyz:1:gate_li3po4_1m" \
      "li3po4_1010240.xyz:4:gate_li3po4_1m"; do
        IFS=':' read -r xyz_base ngpu tag <<< "$spec"
        echo "=== Gate 2 probe: $xyz_base W=$ngpu ($tag) ==="
        "$TORCHRUN" --standalone --nproc_per_node="$ngpu" \
            "$EXP_DIR/drivers/run_orb_scaling.py" \
            --input "$DATA_DIR/$xyz_base" \
            --model orb_v3_direct_20_omat \
            --orb-precision float32-high \
            --edge-chunk-size 524288 \
            --mixed-precision bf16 \
            --use-fused-kernels true \
            --exchanger gpudirect \
            --study baseline --tag "$tag" \
            --warmup-steps 5 --timed-steps 10 \
            --cutoff-angstrom 6.0 --ghost-buffer-angstrom 0.5 \
            --rebalance-interval 1000 --scalable-halo --resample-velocities \
            --output-dir "$PROBE_DIR"
    done
    ;;

  *)
    echo "Unknown probe mode: $MODE (expected parity | ceiling | ceiling-direct | mnnvl | li3po4-fused)" >&2
    exit 2
    ;;
esac

echo
echo "=== Probe summary ==="
"$PY" - "$PROBE_DIR" <<'PY'
import json, pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).rglob("result.json")):
    d = json.loads(p.read_text())
    metrics = d.get("metrics") or {}
    fused = metrics.get("fused_kernels_active")
    print(f"  {d.get('run_id', p.parent.name):65s} status={str(d.get('status')):7s} "
          f"fabric={str(d.get('use_fabric')):5s} fused={str(fused):5s} "
          f"sps={str(d.get('steps_per_sec'))[:7]:7s} "
          f"peakMB={str(d.get('peak_mem_mb_max'))[:8]:8s}")
PY
