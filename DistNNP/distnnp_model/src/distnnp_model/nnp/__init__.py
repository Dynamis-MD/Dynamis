"""Focused exports for current MACE adaptation experiments.

This package initializer intentionally avoids importing UMA by default because
the current cluster environment used for MACE experimentation does not provide
its optional ``fairchem`` dependency. Keeping this file MACE-focused prevents
unrelated optional-model imports from breaking the distributed adaptation runs.
"""

try:
    from distnnp_model.nnp.MACE_Adaptation.single_gpu_adapter import (
        MACEPotentialModel,
        load_raw_mace_model,
    )
except Exception:  # pragma: no cover - optional dependency import
    MACEPotentialModel = None
    load_raw_mace_model = None

try:
    from distnnp_model.nnp.MACE_Adaptation.phase1_raw_exchange_adapter import (
        MACEPotentialModelWithRawExchange,
    )
except Exception:  # pragma: no cover - optional dependency import
    MACEPotentialModelWithRawExchange = None

try:
    from distnnp_model.nnp.MACE_Adaptation.phase2_owned_energy_adapter import (
        MACEPotentialModelWithOwnedEnergy,
    )
except Exception:  # pragma: no cover - optional dependency import
    MACEPotentialModelWithOwnedEnergy = None

try:
    from distnnp_model.nnp.MACE_Adaptation.phase3_owned_force_adapter import (
        MACEPotentialModelWithOwnedEnergyForces,
    )
except Exception:  # pragma: no cover - optional dependency import
    MACEPotentialModelWithOwnedEnergyForces = None

try:
    from distnnp_model.nnp.MACE_Adaptation.phase4_internal_feature_exchange_adapter import (
        MACEPotentialModelWithInternalFeatureExchange,
    )
except Exception:  # pragma: no cover - optional dependency import
    MACEPotentialModelWithInternalFeatureExchange = None

try:
    from distnnp_model.nnp.MACE_Adaptation.final_adapter import (
        DistMACEAdapter,
        FeatureExchangePolicy,
    )
except Exception:  # pragma: no cover - optional dependency import
    DistMACEAdapter = None
    FeatureExchangePolicy = None

try:
    from distnnp_model.nnp.MACE_Adaptation.mace_subclass_boundary import (
        DistMACEModelSubclassBoundary,
    )
except Exception:  # pragma: no cover - optional dependency import
    DistMACEModelSubclassBoundary = None

__all__ = [
    "MACEPotentialModel",
    "MACEPotentialModelWithRawExchange",
    "MACEPotentialModelWithOwnedEnergy",
    "MACEPotentialModelWithOwnedEnergyForces",
    "MACEPotentialModelWithInternalFeatureExchange",
    "DistMACEAdapter",
    "FeatureExchangePolicy",
    "DistMACEModelSubclassBoundary",
    "load_raw_mace_model",
]
