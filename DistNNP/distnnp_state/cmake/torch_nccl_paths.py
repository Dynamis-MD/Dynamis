import json, os, sys, sysconfig, glob

def find():
    out = {"include": "", "lib": "", "static": False}
    sp = sysconfig.get_paths()["purelib"]

    inc = os.path.join(sp, "nvidia", "nccl", "include")
    if os.path.isfile(os.path.join(inc, "nccl.h")):
        out["include"] = inc

    for pat in (os.path.join(sp, "nvidia", "nccl", "lib", "libnccl.so*"),
                os.path.join(sp, "torch", "lib", "libnccl.so*")):
        hits = sorted(glob.glob(pat))
        if hits:
            out["lib"] = hits[-1]   # prefer libnccl.so.2 over libnccl.so
            break

    if not out["lib"]:
        out["static"] = True        # torch likely has NCCL statically linked
    print(json.dumps(out))

find()