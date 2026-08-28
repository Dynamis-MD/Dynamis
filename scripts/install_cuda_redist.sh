#!/usr/bin/env bash
#
# Install a minimal CUDA toolkit (nvcc + cudart + cccl) from NVIDIA's
# redistributable tarballs into a user-writable prefix. No root, no conda.
#
# This is all the DistNNP C++/CUDA extensions need to build: they only
# #include <cuda_runtime.h> / <cuda.h> and link against Torch + the driver
# stub (libcuda.so, already present in /usr/lib/aarch64-linux-gnu).
#
# The resulting tree is a standard CUDA layout (bin/ include/ lib64/ nvvm/),
# so CMake's find_package(CUDAToolkit) picks it up from CUDA_HOME with no
# shims.
#
# Usage:
#   scripts/install_cuda_redist.sh                 # -> $HOME/cuda/13.0.3
#   CUDA_VER=13.0.3 CUDA_PREFIX=/path scripts/install_cuda_redist.sh
#
set -euo pipefail

CUDA_VER="${CUDA_VER:-13.0.3}"
CUDA_PREFIX="${CUDA_PREFIX:-$HOME/cuda/$CUDA_VER}"
PLATFORM="${CUDA_PLATFORM:-linux-sbsa}"   # aarch64 server (Grace/GB200)
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
MANIFEST_URL="$BASE/redistrib_${CUDA_VER}.json"
# CUDA 13 split the old monolithic nvcc package: crt headers (crt/host_config.h)
# are in cuda_crt, and the NVVM backend (cicc + libnvvm) is in libnvvm.
# cuda_nvrtc is needed because Torch's CMake config links CUDA_nvrtc_LIBRARY.
COMPONENTS=(cuda_nvcc cuda_crt libnvvm cuda_nvrtc cuda_cudart cuda_cccl cuda_cuobjdump cuda_nvdisasm)

echo ">> CUDA $CUDA_VER ($PLATFORM)  ->  $CUDA_PREFIX"

_marker="$CUDA_PREFIX/.distnnp-components"
if [ -x "$CUDA_PREFIX/bin/nvcc" ] && [ -f "$_marker" ] \
   && [ "$(cat "$_marker" 2>/dev/null)" = "${COMPONENTS[*]}" ]; then
    echo ">> Toolkit already complete at $CUDA_PREFIX -- nothing to do."
    "$CUDA_PREFIX/bin/nvcc" --version | tail -1
    exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo ">> Fetching manifest: $MANIFEST_URL"
curl -fsSL "$MANIFEST_URL" -o "$work/manifest.json"

mkdir -p "$CUDA_PREFIX"

for comp in "${COMPONENTS[@]}"; do
    read -r rel sha < <(python3 - "$work/manifest.json" "$comp" "$PLATFORM" <<'PY'
import json, sys
manifest, comp, platform = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(manifest))[comp][platform]
print(d["relative_path"], d["sha256"])
PY
)
    url="$BASE/$rel"
    tarball="$work/$(basename "$rel")"
    echo ">> Downloading $comp"
    echo "   $url"
    curl -fsSL "$url" -o "$tarball"
    echo "$sha  $tarball" | sha256sum -c - >/dev/null
    tar -xf "$tarball" -C "$work"
    extracted="$work/$(basename "$rel" .tar.xz)"
    echo ">> Merging $comp -> $CUDA_PREFIX"
    # Drop metadata files, merge the rest (bin/ include/ lib/ nvvm/ ...).
    rm -f "$extracted"/LICENSE "$extracted"/*.json
    rsync -a "$extracted"/ "$CUDA_PREFIX"/
done

# NVIDIA sbsa tarballs use lib/; CMake's FindCUDAToolkit also probes lib64/.
if [ -d "$CUDA_PREFIX/lib" ] && [ ! -e "$CUDA_PREFIX/lib64" ]; then
    ln -s lib "$CUDA_PREFIX/lib64"
fi

echo "${COMPONENTS[*]}" > "$_marker"

echo
echo ">> Done. Toolkit installed at: $CUDA_PREFIX"
"$CUDA_PREFIX/bin/nvcc" --version | tail -2
echo
echo "   Add to your shell (or: source scripts/env.sh):"
echo "     export CUDA_HOME=$CUDA_PREFIX"
echo "     export PATH=\$CUDA_HOME/bin:\$PATH"
