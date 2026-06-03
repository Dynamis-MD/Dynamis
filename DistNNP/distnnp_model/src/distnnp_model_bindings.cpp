#include "distnnp_model/NCCLFeatureExchanger/NCCLExchanger.hpp"
#include "distnnp_model/NVFeatureExchanger/NVExchanger.hpp"

#include "distnnp_model/exchanger.hpp"
#include "distnnp_model/push_buffer.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_model_impl, m) {
  m.def("push_buffer_create", &distnnp_model::push_buffer_create,
        "Gather rows into buffer by index");
  py::class_<distnnp_model::NVFeatureExchanger>(m, "NVFeatureExchanger")
      .def(py::init<>())
      .def("setup", &distnnp_model::NVFeatureExchanger::setup,
           py::arg("dist_env"), py::arg("partition_comm_info"),
           py::arg("max_feature_dim"))
      .def("reset", &distnnp_model::NVFeatureExchanger::reset,
           py::arg("partition_comm_info"), py::arg("max_feature_dim"))
      .def("feature_push_forward",
           &distnnp_model::NVFeatureExchanger::feature_push_forward,
           py::arg("return_features"), py::arg("local_features"),
           py::arg("partition_info"))
      .def("feature_push_grad",
           &distnnp_model::NVFeatureExchanger::feature_push_grad,
           py::arg("grad_local_features"), py::arg("grad_output_features"),
           py::arg("partition_info"));

  py::class_<distnnp_model::NCCLFeatureExchanger>(m, "NCCLFeatureExchanger")
      .def(py::init<>())
      .def("setup", &distnnp_model::NCCLFeatureExchanger::setup,
           py::arg("dist_env"), py::arg("partition_comm_info"),
           py::arg("max_feature_dim"))
      .def("reset", &distnnp_model::NCCLFeatureExchanger::reset,
           py::arg("partition_comm_info"), py::arg("max_feature_dim"))
      .def("feature_push_forward",
           &distnnp_model::NCCLFeatureExchanger::feature_push_forward,
           py::arg("return_features"), py::arg("local_features"),
           py::arg("partition_info"))
      .def("feature_push_grad",
           &distnnp_model::NCCLFeatureExchanger::feature_push_grad,
           py::arg("grad_local_features"), py::arg("grad_output_features"),
           py::arg("partition_info"));
}
