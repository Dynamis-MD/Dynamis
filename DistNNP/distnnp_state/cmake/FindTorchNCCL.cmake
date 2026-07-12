execute_process(
  COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch_nccl_paths.py"
  OUTPUT_VARIABLE _nccl_json OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

string(JSON TORCH_NCCL_INCLUDE_DIR GET "${_nccl_json}" include)
string(JSON TORCH_NCCL_LIBRARY     GET "${_nccl_json}" lib)
string(JSON _nccl_static           GET "${_nccl_json}" static)

if(_nccl_static)
  message(FATAL_ERROR
    "This PyTorch install has NCCL linked statically (no libnccl.so found).\n"
    "DistNNP only supports PyTorch's bundled *shared* NCCL. Install a pip "
    "wheel of torch (which depends on nvidia-nccl-cu12) and rebuild.")
endif()

if(NOT TORCH_NCCL_INCLUDE_DIR)
  message(FATAL_ERROR "Found ${TORCH_NCCL_LIBRARY} but no nccl.h alongside it.")
endif()

# Imported target — full path to the .so.2, so no system search happens.
add_library(TorchNCCL::nccl SHARED IMPORTED)
set_target_properties(TorchNCCL::nccl PROPERTIES
  IMPORTED_LOCATION "${TORCH_NCCL_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_NCCL_INCLUDE_DIR}")

message(STATUS "Torch NCCL header: ${TORCH_NCCL_INCLUDE_DIR}")
message(STATUS "Torch NCCL library: ${TORCH_NCCL_LIBRARY}")