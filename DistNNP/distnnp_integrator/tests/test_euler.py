import pytest
import torch
from distnnp_integrator.distnnp_integrator_impl import EulerIntegrator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def make_tensors(*arrays, device="cuda"):
    """Helper: convert lists to contiguous float32 CUDA tensors."""
    return [
        torch.tensor(a, dtype=torch.float32, device=device).contiguous()
        for a in arrays
    ]


# ---------------------------------------------------------------------------
# Correctness tests — verify the Euler equations analytically
# ---------------------------------------------------------------------------

def test_zero_force_velocity_unchanged():
    """With F=0, velocity must not change and position advances by v*dt."""
    dt = 0.1
    pos, vel, frc, mass = make_tensors([0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [1.0, 1.0])

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([1.0, 2.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([0.1, 0.2], device="cuda"))


def test_constant_force_unit_mass():
    """Single atom, F=2, m=1, dt=0.5 → v_new=1.0, x_new=0.5."""
    dt = 0.5
    pos, vel, frc, mass = make_tensors([0.0], [0.0], [2.0], [1.0])

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # v(t+dt) = 0 + (2/1)*0.5 = 1.0
    # x(t+dt) = 0 + 1.0*0.5  = 0.5
    assert torch.allclose(vel, torch.tensor([1.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([0.5], device="cuda"))


def test_force_divided_by_mass():
    """Heavier atom should accelerate less under the same force."""
    dt = 1.0
    # Two atoms: same force F=4, different masses m=1 and m=2
    pos, vel, frc, mass = make_tensors([0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [1.0, 2.0])

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # atom 0: v = 4/1*1 = 4.0,  x = 4.0
    # atom 1: v = 4/2*1 = 2.0,  x = 2.0
    assert torch.allclose(vel, torch.tensor([4.0, 2.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([4.0, 2.0], device="cuda"))


def test_nonzero_initial_velocity():
    """Starting velocity contributes to the position update."""
    dt = 0.5
    # v0=3, F=2, m=1  →  v_new=3+1=4,  x_new=1+4*0.5=3
    pos, vel, frc, mass = make_tensors([1.0], [3.0], [2.0], [1.0])

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([4.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([3.0], device="cuda"))


def test_negative_force():
    """Negative force should decelerate the atom."""
    dt = 1.0
    pos, vel, frc, mass = make_tensors([0.0], [5.0], [-2.0], [1.0])

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # v_new = 5 + (-2/1)*1 = 3.0
    # x_new = 0 + 3.0*1   = 3.0
    assert torch.allclose(vel, torch.tensor([3.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([3.0], device="cuda"))


# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """step() must modify the original position and velocity tensors."""
    dt = 0.1
    pos, vel, frc, mass = make_tensors([0.0], [1.0], [0.0], [1.0])
    pos_id = id(pos)
    vel_id = id(vel)

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert id(pos) == pos_id
    assert id(vel) == vel_id
    assert not torch.allclose(pos, torch.zeros(1, device="cuda"))


def test_multiple_steps_accumulate():
    """Repeated calls to step() should accumulate correctly."""
    dt = 0.1
    # F=1, m=1: each step adds 0.1 to velocity
    pos, vel, frc, mass = make_tensors([0.0], [0.0], [1.0], [1.0])
    integrator = EulerIntegrator()

    integrator.step(pos, vel, frc, mass, dt)  # v=0.1, x=0.01
    integrator.step(pos, vel, frc, mass, dt)  # v=0.2, x=0.03

    assert torch.allclose(vel, torch.tensor([0.2], device="cuda"), atol=1e-6)
    assert torch.allclose(pos, torch.tensor([0.03], device="cuda"), atol=1e-6)


def test_large_atom_count():
    """Verify the kernel handles a large number of atoms without error."""
    n = 100_000
    device = "cuda"
    pos = torch.zeros(n, dtype=torch.float32, device=device)
    vel = torch.ones(n, dtype=torch.float32, device=device)
    frc = torch.full((n,), 2.0, dtype=torch.float32, device=device)
    mass = torch.ones(n, dtype=torch.float32, device=device)

    EulerIntegrator().step(pos, vel, frc, mass, 0.1)

    assert torch.allclose(vel, torch.full((n,), 1.2, device=device), atol=1e-5)
    assert torch.allclose(pos, torch.full((n,), 0.12, device=device), atol=1e-5)
