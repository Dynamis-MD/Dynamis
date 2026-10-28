cmake_minimum_required(VERSION 3.25.0)

project(
    dynamis_state
    VERSION 0.1.0
    DESCRIPTION "A distributed state management library for neural network potentials"
    HOMEPAGE_URL "https://github.com/Dynamis-MD/Dynamis"
    LANGUAGES CXX CUDA
)

option(DYNAMIS_TIMING "Compile in timing instrumentation" ON)

# Single source of truth for the on/off decision, installed for downstream
# packages to inherit. Do NOT add_compile_definitions(DYNAMIS_TIMING) here —
# that would only reach this project.
set(DYNAMIS_STATE_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dynamis_state/timing_config.hpp.in"
    "${DYNAMIS_STATE_GENERATED_INCLUDE_DIR}/dynamis_state/timing_config.hpp"
    @ONLY
)

# Use the modern CUDAToolkit finder first to resolve the correct include path.
# In conda environments, nvcc lives in $PREFIX/bin but headers are under
# $PREFIX/targets/x86_64-linux/include — a non-standard layout that PyTorch's
# legacy FindCUDA module cannot discover on its own. We cache both variables
# with FORCE so that FindCUDA does not overwrite them when Torch calls it.
#
# CUDA_HOME is the conventional env variable on HPC systems; CUDAToolkit_ROOT
# is the CMake equivalent. If either is set, pass it as a hint to the finder.
if(DEFINED ENV{CUDA_HOME})
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "" FORCE)
elseif(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}" CACHE PATH "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" CACHE PATH "" FORCE)
set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}" CACHE PATH "" FORCE)


find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorchNCCL.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Locate torch_python after find_package(Torch) so TORCH_LIBRARIES paths are available.
if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()

find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS
  ${TORCH_LIB_DIR}
  ${TORCH_INSTALL_PREFIX}/lib
  $ENV{Python_SITELIB}/torch/lib64
  $ENV{Python_SITELIB}/torch/lib
  NO_DEFAULT_PATH)

find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)

# ---------------------------------------------------------------------------
# Timing library
# ---------------------------------------------------------------------------
# TimingRegistry is a singleton, so every module needs to share ONE copy of it —
# hence a real shared library with a SONAME, not the CPython extension (which
# carries none). Built only when timing is on; off, there's nothing to link.
if(DYNAMIS_TIMING)
    add_library(dynamis_state_timing SHARED src/csrc/timing.cpp)

    target_include_directories(dynamis_state_timing
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${DYNAMIS_STATE_GENERATED_INCLUDE_DIR}>
        PRIVATE
            ${TORCH_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    # c10_cuda (via ${TORCH_LIBRARIES}) provides getCurrentCUDAStream; cudart provides
    # the event API. No torch_python, no pybind11 — this is not a Python module.
    target_link_libraries(dynamis_state_timing
        PRIVATE
        ${TORCH_LIBRARIES}
        torch
        CUDA::cudart
    )

    target_compile_options(dynamis_state_timing PRIVATE -Wall -Wextra)

    # No VERSION/SOVERSION: the SONAME alone is enough, and a symlink chain
    # wouldn't survive a wheel anyway.
    set_target_properties(dynamis_state_timing
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF

        INSTALL_RPATH "$ORIGIN/../torch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH ON
    )

    install(TARGETS dynamis_state_timing LIBRARY DESTINATION .)
endif()

python_add_library(dynamis_state_impl
    MODULE
    WITH_SOABI
    src/dynamis_state_bindings.cpp
    src/csrc/partition/simple_spatial_partitioner.cpp
    src/csrc/partition/kdtree_partitioner.cpp
    src/csrc/partition/kdtree.cpp
    src/csrc/partition/kdtree_kernels.cu
    src/csrc/partition/sfc_partitioner.cpp
    src/csrc/partition/sfc.cpp
    src/csrc/partition/sfc_kernels.cu
    src/csrc/build_halo_maps.cpp
    src/csrc/graph/radius_graph.cpp
    src/csrc/graph/radius_graph_kernels.cu
    src/csrc/graph/dist_graph_ops.cpp
    src/csrc/graph/dist_graph_ops_kernels.cu
    src/csrc/trajectory_reader.cpp
    src/csrc/trajectory_writer.cpp
)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${TORCH_NCCL_INCLUDE_DIR}")
check_cxx_source_compiles("
  #include <nccl.h>
  int main() {
    ncclWaitSignalDesc_t d{};
    (void)&ncclPutSignal; (void)&ncclWaitSignal;
    (void)&ncclCommWindowRegister; (void)&ncclMemAlloc;
    (void)d.opCnt;
    return 0;
  }" DYNAMIS_HAVE_NCCL_RMA)

if(DYNAMIS_HAVE_NCCL_RMA)
  target_compile_definitions(dynamis_state_impl PRIVATE DYNAMIS_HAVE_NCCL_RMA=1)
else()
  message(STATUS "NCCL RMA (PutSignal/WaitSignal) unavailable — "
                 "RMATransport will raise at construction.")
endif()

target_include_directories(dynamis_state_impl
    PRIVATE
    include
    ${DYNAMIS_STATE_GENERATED_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(dynamis_state_impl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# torch_python provides pybind11 type_casters for at::Tensor and must be
# linked explicitly — it is not included in ${TORCH_LIBRARIES}.
# Do not link standalone pybind11::module here: torch bundles its own
# pybind11 and mixing the two causes ABI conflicts (undefined type_caster
# symbols at import time).
target_link_libraries(dynamis_state_impl
    PRIVATE
    ${TORCH_LIBRARIES}
    torch
    ${TORCH_PYTHON_LIBRARY}
    TorchNCCL::nccl
)

# Resolved by SONAME so this extension and dynamis_integrator share one TimingRegistry.
if(DYNAMIS_TIMING)
    target_link_libraries(dynamis_state_impl PRIVATE dynamis_state_timing)
endif()

set_target_properties(dynamis_state_impl
    PROPERTIES
    FILE_SET HEADERS
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "80;90;100"

    INSTALL_RPATH "$ORIGIN:$ORIGIN/../nvidia/nccl/lib:$ORIGIN/../torch/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS dynamis_state_impl
LIBRARY DESTINATION .)

# Install public headers so downstream packages (e.g. dynamis_model) can find
# them after `pip install dynamis_state`.  They land at:
#   site-packages/dynamis_state/include/dynamis_state/*.hpp
install(DIRECTORY include/ DESTINATION include
    PATTERN "*.hpp.in" EXCLUDE)
install(FILES "${DYNAMIS_STATE_GENERATED_INCLUDE_DIR}/dynamis_state/timing_config.hpp"
    DESTINATION include/dynamis_state)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/dynamis_state/py.typed"
    "src/dynamis_state/dynamis_state_impl.pyi"
    DESTINATION .
    OPTIONAL
)
