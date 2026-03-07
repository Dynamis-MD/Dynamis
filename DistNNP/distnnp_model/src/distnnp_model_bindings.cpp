#include "distnnp_model/push_buffer.hpp"
#include <torch/extension.h>
#include "distnnp_model/feature_push.hpp"

PYBIND11_MODULE(distnnp_model_impl, m)
{
  m.def("push_buffer_create", &distnnp_model::push_buffer_create, "Gather rows into buffer by index");
  m.def("push_feature_forward", &distnnp_model::feature_push_forward, "Push feature forward implementation",
        py::arg("local_features"), py::arg("partition_info"));
  m.def("push_feature_backward", &distnnp_model::feature_push_backward, "Push feature backward implementation",
        py::arg("grad_output"), py::arg("grad_input_shape"), py::arg("partition_info"));
}
