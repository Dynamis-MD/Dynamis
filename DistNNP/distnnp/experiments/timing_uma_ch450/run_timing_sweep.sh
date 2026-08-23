#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=distnnp-timing-sweep
#SBATCH --output=/u/rcramer/DistNNP/distnnp_timing_sweep_%j.out

# Runs the UMA/CH450 timing test at 1, 2, and 4 GPUs on a single node and
# writes each run's per-rank timing_rank{N}.jsonl files to its own results
# subdirectory. Produces 7 timing files total (1 + 2 + 4). Consolidate with
# build_report.py once this finishes.
#
# Assumes the DistNNP sub-packages are already built/installed in the active
# venv -- run_test.sh (repo root) handles that and then calls this script;
# submit this directly only once that's done.

set -euo pipefail

module load cudatoolkit/26.5_13.2  # cudatoolkit/25.3_12.8 no longer exists on Delta
module load gcc-native/14

if [ -z "${VIRTUAL_ENV:-}" ]; then
    source /u/rcramer/.venvs/distnnp-py313/bin/activate
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

for NGPU in 1 2 4; do
    OUT_DIR="results/timing/${NGPU}gpu"
    mkdir -p "$OUT_DIR"
    echo "=== Running ${NGPU}-GPU timing test -> ${OUT_DIR} ==="
    torchrun --standalone --nproc_per_node=${NGPU} main.py --output-dir "$OUT_DIR"
done

python3 build_report.py --results-dir results/timing --gpu-counts 1 2 4
