#include "distnnp_model/push_buffer.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_model_impl, m)
{
  m.def("push_buffer_create", &distnnp_model::push_buffer_create, "Gather rows into buffer by index");
}