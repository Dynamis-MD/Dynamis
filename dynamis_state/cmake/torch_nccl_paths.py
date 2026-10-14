import json
import os
import glob
import importlib.util
import sys


def get_module_path(module_name):
    # find_spec raises (rather than returning None) when a parent package is
    # missing, e.g. "nvidia.nccl" with no "nvidia" in an isolated build env.
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None
    if spec and spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


def find(torch_path=None):
    out = {"include": "", "lib": "", "static": False}

    # pip/uv build isolation hides the environment's torch from this
    # interpreter, so fall back to the torch install CMake already located;
    # the nvidia-nccl wheel sits next to it in the same site-packages.
    torch_path = get_module_path("torch") or torch_path
    nvidia_nccl_path = get_module_path("nvidia.nccl")
    if not nvidia_nccl_path and torch_path:
        candidate = os.path.join(os.path.dirname(torch_path), "nvidia", "nccl")
        if os.path.isdir(candidate):
            nvidia_nccl_path = candidate

    # 1. Find the include directory
    if nvidia_nccl_path:
        inc = os.path.join(nvidia_nccl_path, "include")
        if os.path.isfile(os.path.join(inc, "nccl.h")):
            out["include"] = inc

    # 2. Find the shared library
    search_paths = []
    if nvidia_nccl_path:
        search_paths.append(os.path.join(nvidia_nccl_path, "lib", "libnccl.so*"))

    if torch_path:
        search_paths.append(os.path.join(torch_path, "lib", "libnccl.so*"))

    for pat in search_paths:
        hits = sorted(glob.glob(pat))
        if hits:
            out["lib"] = hits[-1]  # prefer libnccl.so.2 over libnccl.so
            break

    if not out["lib"]:
        out["static"] = True

    print(json.dumps(out))


if __name__ == "__main__":
    find(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None)
