import json
import os
import glob
import importlib.util


def get_module_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec and spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


def find():
    out = {"include": "", "lib": "", "static": False}

    # 1. Find the include directory
    nvidia_nccl_path = get_module_path("nvidia.nccl")
    if nvidia_nccl_path:
        inc = os.path.join(nvidia_nccl_path, "include")
        if os.path.isfile(os.path.join(inc, "nccl.h")):
            out["include"] = inc

    # 2. Find the shared library
    search_paths = []
    if nvidia_nccl_path:
        search_paths.append(os.path.join(nvidia_nccl_path, "lib", "libnccl.so*"))

    torch_path = get_module_path("torch")
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
    find()
