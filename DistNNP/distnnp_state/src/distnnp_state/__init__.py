import distnnp_state.distnnp_state_impl as distnnp_state_impl

# C++ structs
MolecularInfo = distnnp_state_impl.MolecularInfo
DistEnvironment = distnnp_state_impl.DistEnvironment
SimulationConfig = distnnp_state_impl.SimulationConfig
PartitionMessagePassingInfo = distnnp_state_impl.PartitionMessagePassingInfo
DistNNPState = distnnp_state_impl.DistNNPState

# C++ partitioners (partition + build_halo_maps are both methods)
SimpleSpatialPartitioner = distnnp_state_impl.SimpleSpatialPartitioner
KDTreePartitioner = distnnp_state_impl.KDTreePartitioner
SFCPartitioner = distnnp_state_impl.SFCPartitioner
SFCCurveType = distnnp_state_impl.SFCCurveType

# CUDA radius graph with PBC
from distnnp_state.radius_graph_pbc_cuda import radius_graph_pbc_cuda
from distnnp_state.state_wrapper import TorchDistEnv

# Phase 5: graph-aware distributed halo builder (no NCCL halo exchange)
from distnnp_state.dist_graph_builder import (
    DistGraphBuilder,
    DistBuildGraphResult,
    BuildMode,
)

# IO
TrajectoryWriter = distnnp_state_impl.TrajectoryWriter

# Timing
TimingRegistry = distnnp_state_impl.TimingRegistry
TimerClass = distnnp_state_impl.TimerClass
Where = distnnp_state_impl.Where
Level = distnnp_state_impl.Level
TIMING_ENABLED = distnnp_state_impl.TIMING_ENABLED

if TIMING_ENABLED:
    Timer = distnnp_state_impl.Timer
else:
    class Timer:  # type: ignore[no-redef]
        """No-op stand-in for when distnnp_state was built with timing off.

        Python has no preprocessor, so this swaps the exported name instead —
        call sites stay unconditional without checking TIMING_ENABLED.
        """

        __slots__ = ()

        def start(self, key, cls, where=None, parent_id=0) -> None:
            pass

        def stop(self) -> None:
            pass

        @staticmethod
        def current_span_id() -> int:
            return 0

        @staticmethod
        def set_level(level) -> None:
            pass

        @staticmethod
        def enabled() -> bool:
            return False

        @staticmethod
        def set_step(step) -> None:
            pass

__all__ = [
    "MolecularInfo",
    "DistEnvironment",
    "SimulationConfig",
    "PartitionMessagePassingInfo",
    "DistNNPState",
    "SimpleSpatialPartitioner",
    "KDTreePartitioner",
    "SFCPartitioner",
    "SFCCurveType",
    "TrajectoryWriter",
    "radius_graph_pbc_cuda",
    "TorchDistEnv",
    "DistGraphBuilder",
    "DistBuildGraphResult",
    "BuildMode",
    "SimState",
    "Timer",
    "TimingRegistry",
    "TimerClass",
    "Where",
    "Level",
    "TIMING_ENABLED",
]
