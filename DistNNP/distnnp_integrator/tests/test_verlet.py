import pytest
import torch
from distnnp_integrator.distnnp_integrator_impl import VerletIntegrator

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
# Correctness tests — verify the Velocity Verlet equations analytically
# ---------------------------------------------------------------------------

def test_zero_force_velocity_unchanged():
    """With F=0, velocity must not change and position advances by v*dt."""
    dt = 0.1
    pos, vel, frc, mass = make_tensors([0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    integrator = VerletIntegrator()

    integrator.halfstep1(pos, vel, frc, mass, dt)
    integrator.halfstep2(vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([1.0, 2.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([0.1, 0.2], device="cuda"))

def test_constant_force_unit_mass():
    """Single atom, F=2, m=1, dt=0.5."""
    dt = 0.5
    pos, vel, frc, mass = make_tensors([0.0], [0.0], [2.0], [1.0])
    integrator = VerletIntegrator()

    integrator.halfstep1(pos, vel, frc, mass, dt)
    
    # v(t + dt/2) = 0 + (2 / 2*1) * 0.5 = 0.5
    # x(t + dt) = 0 + 0.5 * 0.5 = 0.25
    assert torch.allclose(vel, torch.tensor([0.5], device="cuda"))
    assert torch.allclose(pos, torch.tensor([0.25], device="cuda"))

    # Re-evaluate forces (constant here)
    integrator.halfstep2(vel, frc, mass, dt)
    
    # v(t + dt) = 0.5 + (2 / 2*1) * 0.5 = 1.0
    assert torch.allclose(vel, torch.tensor([1.0], device="cuda"))

def test_multiple_steps_accumulate():
    """Repeated calls to halfstep1 and halfstep2 should accumulate correctly."""
    dt = 0.1
    # F=1, m=1
    pos, vel, frc, mass = make_tensors([0.0], [0.0], [1.0], [1.0])
    integrator = VerletIntegrator()

    # Step 1
    integrator.halfstep1(pos, vel, frc, mass, dt)  # v_half = 0.05, pos = 0.005
    integrator.halfstep2(vel, frc, mass, dt)       # v = 0.1

    # Step 2
    integrator.halfstep1(pos, vel, frc, mass, dt)  # v_half = 0.1 + 0.05 = 0.15, pos = 0.005 + 0.15*0.1 = 0.02
    integrator.halfstep2(vel, frc, mass, dt)       # v = 0.15 + 0.05 = 0.2

    assert torch.allclose(vel, torch.tensor([0.2], device="cuda"), atol=1e-6)
    assert torch.allclose(pos, torch.tensor([0.02], device="cuda"), atol=1e-6)

# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """halfstep1 and halfstep2 must modify the original position and velocity tensors."""
    dt = 0.1
    pos, vel, frc, mass = make_tensors([0.0], [1.0], [0.0], [1.0])
    pos_id = id(pos)
    vel_id = id(vel)

    integrator = VerletIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, dt)
    integrator.halfstep2(vel, frc, mass, dt)

    assert id(pos) == pos_id
    assert id(vel) == vel_id

def test_large_atom_count():
    """Verify the kernel handles a large number of atoms without error."""
    n = 100_000
    device = "cuda"
    pos = torch.zeros(n, dtype=torch.float32, device=device)
    vel = torch.ones(n, dtype=torch.float32, device=device)
    frc = torch.full((n,), 2.0, dtype=torch.float32, device=device)
    mass = torch.ones(n, dtype=torch.float32, device=device)

    integrator = VerletIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, 0.1)
    integrator.halfstep2(vel, frc, mass, 0.1)

    # v_new = 1.0 + (2/1)*0.1 = 1.2
    # x_new = 0.0 + (1.0 + (2/2*1)*0.1)*0.1 = 0.11
    assert torch.allclose(vel, torch.full((n,), 1.2, device=device), atol=1e-5)
    assert torch.allclose(pos, torch.full((n,), 0.11, device=device), atol=1e-5)
