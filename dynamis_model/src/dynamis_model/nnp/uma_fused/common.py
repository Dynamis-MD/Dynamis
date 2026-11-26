"""Shared constants and helpers of the fused UMA-s edge block (no kernels here).

* tile / layout constants (``MAX_EDGES``, ``MAX_SLOTS``, ``W_KB``, TMA cache hints) and the M-row conventions of
  ``reference.py``; ``set_regs`` (setmaxnreg helper of the CuTe kernels);
* ``KERNELS``: one compile cache for the CuTe kernels (K0r, K1, K2), keyed by kernel name + argument shapes;
* ``ilv_alloc``: the tile-blocked edge-interleaved activation buffers (Ep/128, ncols/v, 128, v);
* ``nvcc_build``: build + load the generated CUDA sources (gpu_pack, S1, S3) as ctypes libraries.
"""
import ctypes
import hashlib
import os
import platform
import subprocess
import tempfile

import torch

MAX_EDGES = 128   # edges per tile (one tcgen05 M=128 tile)
MAX_SLOTS = 7     # target nodes per tile: 7 targets x 9 rows <= 64 = N of K2's reduction MMA
W_KB = 32         # K per K2 weight stage (512 x 32 FP16 = 32 KB)
CP_EVICT_FIRST, CP_EVICT_LAST = 0x12F0000000000000, 0x14F0000000000000   # TMA L2 cache hints (CUTLASS CacheHintSm90)

# M-row j takes L-coefficient M_PERM[j] (fairchem triton kernels.py); L_FROM_M is the inverse permutation
M_PERM = (0, 2, 6, 3, 7, 1, 5, 8, 4)
L_FROM_M = tuple(M_PERM.index(i) for i in range(9))
# radial block (width 2C) used by M-row j
RAD_ROW = (0, 1, 2, 3, 4, 3, 4, 5, 5)
# gate index (l - 1) for M-rows 1..8 (GateActivation, m_prime=True)
GATE_IDX = (0, 1, 0, 1, 0, 1, 1, 1)

# compiled CuTe kernels, shared by all blocks (weights are kernel arguments): key = (name, shapes, constexprs)
KERNELS = {}


def set_regs(n, entry):
    """CuTe kernels: setmaxnreg from the entry budget ``entry`` to ``n`` (.dec must not exceed, .inc not go below the
    current count; ptxas ignores setmaxnreg unless the launch passes min_blocks_per_mp=1)"""
    import cutlass.cute as cute
    if n < entry:
        cute.arch.setmaxregister_decrease(n)
    elif n > entry:
        cute.arch.setmaxregister_increase(n)


def ilv_alloc(E, ncols, dtype, dev, vec_bytes=16):
    """tile-blocked edge-interleaved buffer (E/128, ncols/v, 128, v), v = vec_bytes / itemsize; uninitialised
    (every user overwrites it completely: zero-filling GBs costs ms)"""
    v = vec_bytes // torch.empty(0, dtype=dtype).element_size()
    return torch.empty((E + 127) // 128, ncols // v, 128, v, device=dev, dtype=dtype)


def cuda_arch() -> str:
    """nvcc arch suffix of the current device, e.g. '100a' (the 'a' variant: tcgen05 / setmaxnreg features)"""
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}a"


def _include_dirs():
    out = []
    conda = os.environ.get("CONDA_PREFIX", "")
    if conda:   # conda's CUDA headers live under targets/<arch>-linux/include
        tgt = {"aarch64": "sbsa-linux", "x86_64": "x86_64-linux"}.get(platform.machine(), "sbsa-linux")
        out.append(os.path.join(conda, "targets", tgt, "include"))
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        out.append(os.path.join(cuda_home, "include"))
    return [d for d in out if os.path.isdir(d)]


def nvcc_build(src: str, tag: str, flags=(), check=None) -> ctypes.CDLL:
    """Compile CUDA source ``src`` to a shared library (cached in $TMPDIR/uma_fused_<uid>/) and load it.

    The cache key covers the source, the nvcc command line (flags, arch, include dirs) and the nvcc version, so a flag
    or GPU change never loads a stale build. Concurrent first builds (several ranks on one node) write private temp
    files and publish with an atomic rename. ``check(path_to_so)`` may inspect the build (e.g. its SASS) and raise
    before it is published. On a compiler error the nvcc log is part of the exception."""
    cmd = ["nvcc", "-O3", "-std=c++17", f"-gencode=arch=compute_{cuda_arch()},code=sm_{cuda_arch()}", "-shared",
           "-Xcompiler", "-fPIC", *flags, *("-I" + d for d in _include_dirs())]
    try:
        ver = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    except OSError as e:
        raise RuntimeError("fused UMA path: nvcc not found (set CUDA_HOME / PATH)") from e
    hsh = hashlib.sha1("\0".join([src, *cmd, ver]).encode()).hexdigest()[:16]
    d = os.path.join(tempfile.gettempdir(), f"uma_fused_{os.getuid()}")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"{tag}_{hsh}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        cu = f"{tmp}.cu"
        with open(cu, "w") as f:
            f.write(src)
        res = subprocess.run(cmd + ["-o", tmp, cu], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed for {tag} ({cu}):\n{res.stderr[-4000:]}")
        try:
            if check is not None:
                check(tmp)
            os.replace(tmp, so)
        finally:
            for p in (tmp, cu):
                if os.path.exists(p):
                    os.remove(p)
    return ctypes.CDLL(so)
