#include "distnnp_integrator/integrator.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_integrator_impl, m) {
  // Euler Integrator
  py::class_<distnnp_integrator::EulerIntegrator>(m, "EulerIntegrator")
      .def(py::init<>())
      .def("step", &distnnp_integrator::EulerIntegrator::step);
  // Verlet Integrator
  py::class_<distnnp_integrator::VerletIntegrator>(m, "VerletIntegrator")
      .def(py::init<>())
      .def("halfstep1", &distnnp_integrator::VerletIntegrator::halfstep1)
      .def("halfstep2", &distnnp_integrator::VerletIntegrator::halfstep2);

  // Langevin Integrator
  py::class_<distnnp_integrator::LangevinIntegrator>(m, "LangevinIntegrator")
      .def(py::init<>())
      .def("halfstep1", &distnnp_integrator::LangevinIntegrator::halfstep1)
      .def("halfstep2", &distnnp_integrator::LangevinIntegrator::halfstep2);
}
