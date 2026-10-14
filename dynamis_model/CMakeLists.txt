cmake_minimum_required(VERSION 3.25.0)

project(
    dynamis_model
    VERSION 0.1.0
    DESCRIPTION "A distributed torch-based neural network potentials library"
    HOMEPAGE_URL "https://github.com/szaman19/Dynamis"
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

# Locate dynamis_state public headers using CMake-native find_path.
find_path(DYNAMIS_STATE_INCLUDE_DIR
    NAMES dynamis_state/dynamis_state.hpp
    HINTS
        "${Python_SITEARCH}/dynamis_state/include"
        "${Python_SITELIB}/dynamis_state/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../dynamis_state/include"
    NO_DEFAULT_PATH
)
if(NOT DYNAMIS_STATE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find dynamis_state headers. "
        "Install it first: pip install -e ./dynamis_state")
endif()
message(STATUS "dynamis_state include dir: ${DYNAMIS_STATE_INCLUDE_DIR}")

python_add_library(dynamis_model_impl
    MODULE
    WITH_SOABI
    src/dynamis_model_bindings.cpp
    src/csrc/exchange/ipc_exchanger.cpp
    src/csrc/exchange/ipc_exchange_kernels.cu
    src/csrc/push_buffer.cu
    src/csrc/exchanger.cu
)

target_include_directories(dynamis_model_impl
    PRIVATE
    include
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DYNAMIS_STATE_INCLUDE_DIR}
)

# torch_python provides pybind11 type_casters for at::Tensor and must be
# linked explicitly — it is not included in ${TORCH_LIBRARIES}.
# Do not link standalone pybind11::module here: torch bundles its own
# pybind11 and mixing the two causes ABI conflicts (undefined type_caster
# symbols at import time).
target_link_libraries(dynamis_model_impl
    PRIVATE
    ${TORCH_LIBRARIES}
    torch
    ${TORCH_PYTHON_LIBRARY}
    CUDA::cuda_driver
)

set_target_properties(dynamis_model_impl
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

install(TARGETS dynamis_model_impl DESTINATION .)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/dynamis_model/py.typed"
    "src/dynamis_model/dynamis_model_impl.pyi"
    DESTINATION .
    OPTIONAL
)
