#!/usr/bin/env bash
#
# Run the DistNNP test suite on a GPU node. Submit with:
#
#   srun --partition=beta --account=su_kchiu_molecular --qos=test \
#        --gres=gpu:2 --time=00:30:00 scripts/run_gpu_tests.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$(cd "$SCRIPT_DIR/.." && pwd)/DistNNP"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

echo "== node $(hostname) =="
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, '| GPUs', torch.cuda.device_count(),
      '|', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo

rc=0

echo "########## sub-package tests (non-perf) ##########"
for pkg in distnnp_state distnnp_integrator distnnp_model; do
    echo "----- $pkg -----"
    ( cd "$INNER/$pkg" && python -m pytest -q --no-header -p no:cacheprovider -m "not perf" \
        --ignore=tests/test_radius_graph_pbc_cuda.py tests ) || rc=1
done

echo
echo "########## distnnp/ distributed tests ##########"
cd "$INNER/distnnp"
echo "-- generating reference fixtures --"
torchrun --standalone --nproc_per_node=1 tests/fixtures/generate_reference.py || rc=1
echo "-- pytest under torchrun (2 ranks) --"
torchrun --standalone --nproc_per_node=2 -m pytest tests/ -v -m "not perf" || rc=1

echo
echo "########## done (rc=$rc) ##########"
exit $rc
