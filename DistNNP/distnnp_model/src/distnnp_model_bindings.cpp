#include <torch/extension.h>
#include "distnnp_model/exchanger.hpp"
#include "distnnp_model/push_buffer.hpp"


PYBIND11_MODULE(distnnp_model_impl, m)
{
  m.def("push_buffer_create", &distnnp_model::push_buffer_create, "Gather rows into buffer by index");

  py::class_<distnnp_model::FeatureExchanger>(m, "FeatureExchanger")
    .def(py::init<>())
    .def("feature_push_forward", &distnnp_model::FeatureExchanger::feature_push_forward, "Push features forward to neighbors")
    .def("feature_push_backward", &distnnp_model::FeatureExchanger::feature_push_backward, "Push feature gradients backward to neighbors");  
}