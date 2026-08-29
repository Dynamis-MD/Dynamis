"""NNP exports for the current MACE top-down adapter workflow.

This subpackage intentionally avoids UMA and other optional model imports so
MACE-only runs do not require fairchem-core or legacy adapter files.
"""

from distnnp_model.nnp.MACE_Adaptation.distmace_top_down_adapter import (
    COMPUTE_DTYPE,
    DistMACETopDownAdapter,
    canonicalize_nonperiodic_geometry,
    load_raw_mace_model,
    validate_partition_info,
)
from distnnp_model.nnp.MACE_Adaptation.distmace_thin_adapter import (
    DistMACEThinAdapter,
)

__all__ = [
    "COMPUTE_DTYPE",
    "DistMACEThinAdapter",
    "DistMACETopDownAdapter",
    "canonicalize_nonperiodic_geometry",
    "load_raw_mace_model",
    "validate_partition_info",
]
