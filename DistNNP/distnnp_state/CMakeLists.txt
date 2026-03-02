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
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

python_add_library(distnnp_state_impl
    MODULE
    WITH_SOABI
    src/distnnp_state_bindings.cpp
)

target_include_directories(distnnp_state_impl
    PRIVATE
    include
)

target_link_libraries(distnnp_state_impl
    PRIVATE
    pybind11::module
    ${TORCH_LIBRARIES}
)

# Install the compiled extension
install(TARGETS distnnp_state_impl DESTINATION distnnp_state)

# Manually install the type marker and stubs if they exist
install(FILES
    "src/distnnp_state/py.typed"
    "src/distnnp_state/distnnp_state.pyi"
    DESTINATION distnnp_state
    OPTIONAL
)

