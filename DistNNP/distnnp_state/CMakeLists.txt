cmake_minimum_required(VERSION 3.25.0)

project(
    distnnp_state
    VERSION 0.1.0
    DESCRIPTION "A distributed state management library for neural network potentials"
    HOMEPAGE_URL "https://github.com/szaman19/DistNNP"
    LANGUAGES CXX CUDA
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

python_add_library(distnnp_state_impl
    MODULE
    WITH_SOABI
    src/distnnp_state_bindings.cpp
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
    src/csrc/trajectory_writer.cpp
)

target_include_directories(distnnp_state_impl
    PRIVATE
    include
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# torch_python provides pybind11 type_casters for at::Tensor and must be
# linked explicitly — it is not included in ${TORCH_LIBRARIES}.
# Do not link standalone pybind11::module here: torch bundles its own
# pybind11 and mixing the two causes ABI conflicts (undefined type_caster
# symbols at import time).
target_link_libraries(distnnp_state_impl
    PRIVATE
    ${TORCH_LIBRARIES}
    torch
    ${TORCH_PYTHON_LIBRARY}
)

set_target_properties(distnnp_state_impl
    PROPERTIES
    FILE_SET HEADERS
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "80;90;100"

    INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS distnnp_state_impl
LIBRARY DESTINATION .)

# Install public headers so downstream packages (e.g. distnnp_model) can find
# them after `pip install distnnp_state`.  They land at:
#   site-packages/distnnp_state/include/distnnp_state/*.hpp
install(DIRECTORY include/ DESTINATION distnnp_state/include)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/distnnp_state/py.typed"
    "src/distnnp_state/distnnp_state_impl.pyi"
    DESTINATION .
    OPTIONAL
)

