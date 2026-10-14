import dynamis_state.dynamis_state_impl as dynamis_state_impl

# C++ structs
MolecularInfo = dynamis_state_impl.MolecularInfo
DistEnvironment = dynamis_state_impl.DistEnvironment
SimulationConfig = dynamis_state_impl.SimulationConfig
PartitionMessagePassingInfo = dynamis_state_impl.PartitionMessagePassingInfo
DynamisState = dynamis_state_impl.DynamisState

# C++ partitioners (partition + build_halo_maps are both methods)
SimpleSpatialPartitioner = dynamis_state_impl.SimpleSpatialPartitioner
KDTreePartitioner = dynamis_state_impl.KDTreePartitioner
SFCPartitioner = dynamis_state_impl.SFCPartitioner
SFCCurveType = dynamis_state_impl.SFCCurveType

# CUDA radius graph with PBC
from dynamis_state.radius_graph_pbc_cuda import radius_graph_pbc_cuda
from dynamis_state.state_wrapper import TorchDistEnv

# Phase 5: graph-aware distributed halo builder (no NCCL halo exchange)
from dynamis_state.dist_graph_builder import (
    DistGraphBuilder,
    DistBuildGraphResult,
    BuildMode,
)

# IO
from dynamis_state.io import (
    TrajectoryReader,
    TrajectoryWriter,
    translate_dynamis_to_xyz,
)

# Timing
TimingRegistry = dynamis_state_impl.TimingRegistry
TimerClass = dynamis_state_impl.TimerClass
Where = dynamis_state_impl.Where
Level = dynamis_state_impl.Level
TIMING_ENABLED = dynamis_state_impl.TIMING_ENABLED

if TIMING_ENABLED:
    Timer = dynamis_state_impl.Timer
else:

    class Timer:  # type: ignore[no-redef]
        """No-op stand-in for when dynamis_state was built with timing off.

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
    "DynamisState",
    "SimpleSpatialPartitioner",
    "KDTreePartitioner",
    "SFCPartitioner",
    "SFCCurveType",
    "TrajectoryReader",
    "TrajectoryWriter",
    "translate_dynamis_to_xyz",
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
