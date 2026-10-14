"""
Core exports for dynamis_model.
"""

from dynamis_model.potential import DynamisModel
import dynamis_model.dynamis_model_impl as dynamis_model_impl

# C++ feature exchanger (wraps IPC exchange over NVLink)
FeatureExchanger = dynamis_model_impl.FeatureExchanger

__all__ = [
    "DynamisModel",
    "FeatureExchanger",
]
