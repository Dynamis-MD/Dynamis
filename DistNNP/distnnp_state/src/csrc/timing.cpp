#include "distnnp_state/timing.hpp"
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <torch/cuda.h>
using namespace std;

namespace distnnp_state {

// Check env variable "DISTNNP_TIMING" to determine whether timing is enabled
// We assume this won't change in the class lifetime
Timer::Timer()
    : is_timing_enabled([] {
          const char* v = getenv("DISTNNP_TIMING");
          return v != nullptr && *v != '\0' && strcmp(v, "0") != 0;
      }())
{
}

void Timer::start() {
    if (!is_timing_enabled) {
        return;
    }
    t0_ = chrono::steady_clock::now();
}

void Timer::stop() {
    if (!is_timing_enabled) {
        return;
    }
    wait_for_gpu();
    auto t1 = chrono::steady_clock::now();
    times_.push_back(chrono::duration_cast<chrono::nanoseconds>(t1 - t0_).count());
}

void Timer::wait_for_gpu() {
    if (torch::cuda::is_available()) {
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
    }
}

} // namespace distnnp_state