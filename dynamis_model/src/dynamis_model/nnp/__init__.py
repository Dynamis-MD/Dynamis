"""
Optional neural-network potential models for Dynamis.

Adapter imports are resolved lazily so MACE and UMA can be installed and used
independently.
"""

from importlib import import_module
from typing import Any

__all__ = ["MACEModel", "UMAModel", "ORBModel"]


def __getattr__(name: str) -> Any:
    if name == "MACEModel":
        return import_module("dynamis_model.nnp.mace").MACEModel
    if name == "UMAModel":
        return import_module("dynamis_model.nnp.uma").UMAModel
    if name == "ORBModel":
        return import_module("dynamis_model.nnp.orb").ORBModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
