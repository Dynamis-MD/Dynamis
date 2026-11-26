"""Fused UMA-s edge-block kernels (Blackwell sm_100 only). See docs/uma_fused/README.md.

Nothing here is imported by default. ``check_available()`` reports whether this GPU and environment can run the fused
kernels; callers fall back to the stock fairchem path (``UMAModel(fused=False)``) otherwise.
"""
import importlib.util
import shutil


def missing_requirements() -> list:
    """Human-readable list of what is missing to run the fused kernels on the current device (empty = available)."""
    import torch

    missing = []
    if not torch.cuda.is_available():
        missing.append("a CUDA device")
    elif torch.cuda.get_device_capability() != (10, 0):
        cap = "sm_%d%d" % torch.cuda.get_device_capability()
        missing.append(f"a Blackwell sm_100 GPU (B200 / GB200); this GPU is {cap}")
    if importlib.util.find_spec("cutlass") is None:
        missing.append("the CuTe DSL (pip install nvidia-cutlass-dsl)")
    if importlib.util.find_spec("triton") is None:
        missing.append("Triton")
    if shutil.which("nvcc") is None:
        missing.append("nvcc on PATH (set CUDA_HOME); S1/S3 and the tile packer are compiled at run time")
    return missing


def check_available() -> None:
    """Raise a RuntimeError naming every missing requirement; use UMAModel(fused=False) on such systems."""
    missing = missing_requirements()
    if missing:
        raise RuntimeError("fused UMA path is not available: needs " + "; ".join(missing)
                           + ". Use the stock fairchem path instead (UMAModel(fused=False)).")
