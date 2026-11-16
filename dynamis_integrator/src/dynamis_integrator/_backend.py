"""Pick the integrator kernels: the CUDA extension if it was built, else Triton.

Whether dynamis_integrator_impl exists is decided at install time by
CMakeLists.txt (it is only built when a CUDA compiler is found).
"""

try:
    from dynamis_integrator.dynamis_integrator_impl import (
        EulerIntegrator,
        VerletIntegrator,
        LangevinIntegrator,
    )

    BACKEND = "cuda"
except ImportError:
    from dynamis_integrator._kernels import (
        EulerIntegrator,
        VerletIntegrator,
        LangevinIntegrator,
    )

    BACKEND = "triton"
