cmake_minimum_required(VERSION 3.25.0)

project(
    distnnp_integrator
    VERSION 0.1.0
    DESCRIPTION "A distributed state management library for neural network potentials"
    HOMEPAGE_URL "https://github.com/szaman19/DistNNP"
    LANGUAGES CXX CUDA
)

option(DISTNNP_TIMING "Compile in timing instrumentation" ON)
if(DISTNNP_TIMING)
    add_compile_definitions(DISTNNP_TIMING)
endif()

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

if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()

# Locate distnnp_state public headers (same pattern as distnnp_model).
find_path(DISTNNP_STATE_INCLUDE_DIR
    NAMES distnnp_state/distnnp_state.hpp
    HINTS
        "${Python_SITEARCH}/distnnp_state/include"
        "${Python_SITELIB}/distnnp_state/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../distnnp_state/include"
    NO_DEFAULT_PATH
)
if(NOT DISTNNP_STATE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find distnnp_state headers. "
        "Install it first: pip install -e ./distnnp_state")
endif()
message(STATUS "distnnp_state include dir: ${DISTNNP_STATE_INCLUDE_DIR}")

# Python extensions use non-standard naming (distnnp_state_impl.cpython-XY-*.so,
# no 'lib' prefix), so ask Python's importlib for the exact path.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.util; s=importlib.util.find_spec('distnnp_state.distnnp_state_impl'); print(s.origin if s else '')"
    OUTPUT_VARIABLE DISTNNP_STATE_IMPL_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT DISTNNP_STATE_IMPL_LIBRARY)
    message(FATAL_ERROR
        "Could not locate distnnp_state_impl. "
        "Install it first: pip install -e ./distnnp_state")
endif()
message(STATUS "distnnp_state_impl library: ${DISTNNP_STATE_IMPL_LIBRARY}")

# distnnp_state_impl*.so has no 'lib' prefix (Python extension naming), so a
# bare target_link_libraries(... ${DISTNNP_STATE_IMPL_LIBRARY}) gets rewritten
# by CMake into "-ldistnnp_state_impl...", which the linker then hunts for as
# "libdistnnp_state_impl....so" and never finds. An IMPORTED target pins the
# exact file instead of letting CMake guess a library name from the path.
add_library(distnnp_state_impl SHARED IMPORTED)
set_target_properties(distnnp_state_impl PROPERTIES
    IMPORTED_LOCATION "${DISTNNP_STATE_IMPL_LIBRARY}")

find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS
  ${TORCH_LIB_DIR}
  $ENV{Python_SITELIB}/torch/lib64
  $ENV{Python_SITELIB}/torch/lib
  NO_DEFAULT_PATH)

find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

python_add_library(distnnp_integrator_impl
    MODULE
    WITH_SOABI
    src/distnnp_integrator_bindings.cpp
    src/csrc/euler_integrator.cu
    src/csrc/verlet_integrator.cu
    src/csrc/langevin_integrator.cu
)

target_include_directories(distnnp_integrator_impl
    PRIVATE
    include
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DISTNNP_STATE_INCLUDE_DIR}
)

# torch_python provides pybind11 type_casters for at::Tensor and must be
# linked explicitly — it is not included in ${TORCH_LIBRARIES}.
# Do not link standalone pybind11::module here: torch bundles its own
# pybind11 and mixing the two causes ABI conflicts (undefined type_caster
# symbols at import time).
target_link_libraries(distnnp_integrator_impl
    PRIVATE
    ${TORCH_LIBRARIES}
    torch
    ${TORCH_PYTHON_LIBRARY}
    # Passed through to the linker verbatim: the file has no 'lib' prefix, so
    # CMake would otherwise rewrite the path into an unresolvable -L/-l pair.
    "-Wl,${DISTNNP_STATE_IMPL_LIBRARY}"
)

set_target_properties(distnnp_integrator_impl
    PROPERTIES
    FILE_SET HEADERS
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF

    INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS distnnp_integrator_impl DESTINATION .)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/distnnp_integrator/py.typed"
    "src/distnnp_integrator/distnnp_integrator_impl.pyi"
    DESTINATION .
    OPTIONAL
)

