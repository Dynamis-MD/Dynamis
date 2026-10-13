#include "dynamis_integrator/integrator.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(dynamis_integrator_impl, m)
{
  // Euler Integrator
  py::class_<dynamis_integrator::EulerIntegrator>(m, "EulerIntegrator")
      .def(py::init<>())
      .def("step", &dynamis_integrator::EulerIntegrator::step);
  // Verlet Integrator
  py::class_<dynamis_integrator::VerletIntegrator>(m, "VerletIntegrator")
      .def(py::init<>())
      .def("halfstep1", &dynamis_integrator::VerletIntegrator::halfstep1)
      .def("halfstep2", &dynamis_integrator::VerletIntegrator::halfstep2);

  // Langevin Integrator
  py::class_<dynamis_integrator::LangevinIntegrator>(m, "LangevinIntegrator")
      .def(py::init<>())
      .def("halfstep1", &dynamis_integrator::LangevinIntegrator::halfstep1)
      .def("halfstep2", &dynamis_integrator::LangevinIntegrator::halfstep2);
}
