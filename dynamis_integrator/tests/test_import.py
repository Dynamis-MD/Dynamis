def test_custom_module_import():
    from dynamis_integrator import BACKEND
    from dynamis_integrator._backend import (
        EulerIntegrator,
        VerletIntegrator,
        LangevinIntegrator,
    )

    assert BACKEND in ("cuda", "triton")
