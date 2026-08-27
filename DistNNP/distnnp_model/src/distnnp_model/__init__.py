"""Core exports for distnnp_model.

Keep this file limited to symbols needed by lower-level modules so the MACE
top-down adapter can import the communication layer without recursing back
through optional adapter packages.
"""

from distnnp_model.potential import DistNNPModel
import distnnp_model.distnnp_model_impl as distnnp_model_impl

# C++ feature exchanger (wraps IPC exchange over NVLink)
FeatureExchanger = distnnp_model_impl.FeatureExchanger

__all__ = [
    "DistNNPModel",
    "FeatureExchanger",
]