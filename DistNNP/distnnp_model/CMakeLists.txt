cmake_minimum_required(VERSION 3.25.0)

project(
    distnnp_model
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


if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()

find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS
  ${TORCH_LIB_DIR}
  $ENV{Python_SITELIB}/torch/lib64
  $ENV{Python_SITELIB}/torch/lib
  NO_DEFAULT_PATH)

find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)


find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Locate distnnp_state public headers.
#
# Two cases:
#   Regular install : headers land at site-packages/distnnp_state/include/
#                     (installed by distnnp_state's CMakeLists.txt)
#   Editable install: scikit-build-core inplace mode places __init__.py in
#                     src/distnnp_state/, so __file__ resolves to the source
#                     tree.  We walk up two levels (src/distnnp_state ->
#                     src -> project root) to reach the project's include/.
#
# We use importlib.util.find_spec rather than importing the package so that
# this step never triggers distnnp_state's own imports (e.g. torch) and works
# regardless of which Python interpreter CMake resolves.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.util, os; \
spec = importlib.util.find_spec('distnnp_state'); \
pkg = spec.submodule_search_locations[0]; \
installed = os.path.join(pkg, 'include'); \
print(installed if os.path.isdir(installed) else \
      os.path.join(os.path.dirname(os.path.dirname(pkg)), 'include'))"
    OUTPUT_VARIABLE DISTNNP_STATE_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _distnnp_state_result
)
if(NOT _distnnp_state_result EQUAL 0)
    message(FATAL_ERROR
        "Could not find distnnp_state. "
        "Install it first: pip install -e ./distnnp_state")
endif()
message(STATUS "distnnp_state include dir: ${DISTNNP_STATE_INCLUDE_DIR}")

python_add_library(distnnp_model_impl
    MODULE
    WITH_SOABI
    src/distnnp_model_bindings.cpp
    src/csrc/push_buffer.cu
)

target_include_directories(distnnp_model_impl
    PRIVATE
    include
    ${TORCH_INCLUDE_DIRS}
    ${DISTNNP_STATE_INCLUDE_DIR}
)

# torch_python provides pybind11 type_casters for at::Tensor and must be
# linked explicitly — it is not included in ${TORCH_LIBRARIES}.
# Do not link standalone pybind11::module here: torch bundles its own
# pybind11 and mixing the two causes ABI conflicts (undefined type_caster
# symbols at import time).
target_link_libraries(distnnp_model_impl
    PRIVATE
    ${TORCH_LIBRARIES}
    torch
    ${TORCH_PYTHON_LIBRARY}
)

set_target_properties(distnnp_model_impl
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

install(TARGETS distnnp_model_impl DESTINATION .)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/distnnp_model/py.typed"
    "src/distnnp_model/distnnp_model.pyi"
    DESTINATION .
    OPTIONAL
)

