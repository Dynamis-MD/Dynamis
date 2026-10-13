cmake_minimum_required(VERSION 3.25.0)

project(
    distnnp_integrator
    VERSION 0.1.0
    DESCRIPTION "A distributed state management library for neural network potentials"
    HOMEPAGE_URL "https://github.com/szaman19/DistNNP"
    LANGUAGES CXX CUDA
)

# No DISTNNP_TIMING option here: it's inherited from dynamis_state's build via
# timing_config.hpp, so this project can't silently disagree with it.

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

# ---------------------------------------------------------------------------
# dynamis_state discovery
# ---------------------------------------------------------------------------
# Ask Python where dynamis_state is installed, so this works for a normal
# install, an editable install, or an isolated build env alike.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.util, os; s = importlib.util.find_spec('dynamis_state'); print(os.path.dirname(s.origin) if s and s.origin else '')"
    OUTPUT_VARIABLE DYNAMIS_STATE_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_path(DYNAMIS_STATE_INCLUDE_DIR
    NAMES dynamis_state/timing_config.hpp
    HINTS
        "${DYNAMIS_STATE_PACKAGE_DIR}/include"
        "${Python_SITEARCH}/dynamis_state/include"
        "${Python_SITELIB}/dynamis_state/include"
    NO_DEFAULT_PATH
)
if(NOT DYNAMIS_STATE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not find dynamis_state headers (looked for dynamis_state/timing_config.hpp). "
        "Install it first: pip install -e ./dynamis_state")
endif()
message(STATUS "dynamis_state include dir: ${DYNAMIS_STATE_INCLUDE_DIR}")

# Read the build's decision rather than re-deciding it; `#cmakedefine` expands
# to `#define DISTNNP_TIMING` when on, a comment when off.
file(READ "${DYNAMIS_STATE_INCLUDE_DIR}/dynamis_state/timing_config.hpp" _distnnp_timing_config)
if(_distnnp_timing_config MATCHES "(^|\n)[ \t]*#define[ \t]+DISTNNP_TIMING")
    set(DYNAMIS_STATE_TIMING_ENABLED TRUE)
else()
    set(DYNAMIS_STATE_TIMING_ENABLED FALSE)
endif()
message(STATUS "dynamis_state timing instrumentation: ${DYNAMIS_STATE_TIMING_ENABLED}")

# Only link dynamis_state when timing is on; off, timing.hpp is header-only
# no-ops and there's nothing to link.
if(DYNAMIS_STATE_TIMING_ENABLED)
    find_library(DYNAMIS_STATE_TIMING_LIBRARY
        NAMES dynamis_state_timing
        HINTS
            "${DYNAMIS_STATE_PACKAGE_DIR}"
            "${Python_SITEARCH}/dynamis_state"
            "${Python_SITELIB}/dynamis_state"
        NO_DEFAULT_PATH
    )
    if(NOT DYNAMIS_STATE_TIMING_LIBRARY)
        message(FATAL_ERROR
            "dynamis_state was built with timing enabled but libdynamis_state_timing.so "
            "was not found next to the installed package. Reinstall dynamis_state: "
            "pip install -e ./dynamis_state")
    endif()
    message(STATUS "dynamis_state timing library: ${DYNAMIS_STATE_TIMING_LIBRARY}")
endif()

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
    ${DYNAMIS_STATE_INCLUDE_DIR}
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
)

# Linked by SONAME, not absolute path, so reinstalling dynamis_state doesn't
# break this import, and both modules resolve to the same TimingRegistry.
if(DYNAMIS_STATE_TIMING_ENABLED)
    target_link_libraries(distnnp_integrator_impl PRIVATE ${DYNAMIS_STATE_TIMING_LIBRARY})
endif()

target_compile_options(distnnp_integrator_impl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

set_target_properties(distnnp_integrator_impl
    PROPERTIES
    FILE_SET HEADERS
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF

    INSTALL_RPATH "$ORIGIN/../dynamis_state:$ORIGIN/../torch/lib"
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

