"""
Optional neural-network potential models for DistNNP.

Adapter imports are resolved lazily so MACE and UMA can be installed and used
independently.
"""

from importlib import import_module
from typing import Any

__all__ = ["MACEModel", "UMAModel"]


def __getattr__(name: str) -> Any:
    if name == "MACEModel":
        return import_module("distnnp_model.nnp.mace").MACEModel
    if name == "UMAModel":
        return import_module("distnnp_model.nnp.uma").UMAModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
