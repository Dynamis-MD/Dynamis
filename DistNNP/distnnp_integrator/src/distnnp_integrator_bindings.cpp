#include "distnnp_integrator/integrator.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_integrator_impl, m)
{
  py::class_<distnnp_integrator::EulerIntegrator>(m, "EulerIntegrator")
      .def(py::init<>())
      .def("step", &distnnp_integrator::EulerIntegrator::step);
}