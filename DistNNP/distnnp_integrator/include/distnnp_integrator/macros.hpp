#pragma once

#define CUDACHECK(cmd)                          \
  do                                            \
  {                                             \
    cudaError_t e = cmd;                        \
    if (e != cudaSuccess)                       \
    {                                           \
      printf("Failed: Cuda error %s:%d '%s'\n", \
             __FILE__, __LINE__,                \
             cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                       \
    }                                           \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)