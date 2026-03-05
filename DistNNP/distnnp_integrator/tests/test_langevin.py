import pytest
import math
import torch
from distnnp_integrator.distnnp_integrator_impl import LangevinIntegrator

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
# Correctness tests — verify the BBK Langevin equations analytically
# ---------------------------------------------------------------------------

def test_zero_friction_matches_verlet():
    """With gamma=0 and noise=0, Langevin should exactly match Velocity Verlet."""
    dt = 0.5
    gamma = 0.0
    kT = 1.0
    pos, vel, frc, mass, noise = make_tensors([0.0], [0.0], [2.0], [1.0], [0.0])
    
    integrator = LangevinIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, noise, gamma, kT, dt)
    integrator.halfstep2(vel, frc, mass, noise, gamma, kT, dt)

    # Same analytical result as test_constant_force_unit_mass in Verlet
    assert torch.allclose(vel, torch.tensor([1.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([0.25], device="cuda"))

def test_friction_damping():
    """Test velocity decay due to friction with no forces and no noise."""
    dt = 0.5
    gamma = 2.0
    kT = 0.0
    pos, vel, frc, mass, noise = make_tensors([0.0], [10.0], [0.0], [1.0], [0.0])
    
    integrator = LangevinIntegrator()
    
    # halfstep1: v_half = v * (1 - gamma*dt/2) = 10 * (1 - 0.5) = 5.0
    #        pos = 0 + 5.0 * 0.5 = 2.5
    integrator.halfstep1(pos, vel, frc, mass, noise, gamma, kT, dt)
    assert torch.allclose(vel, torch.tensor([5.0], device="cuda"))
    assert torch.allclose(pos, torch.tensor([2.5], device="cuda"))

    # halfstep2: v_new = v_half / (1 + gamma*dt/2) = 5.0 / 1.5 = 3.3333...
    integrator.halfstep2(vel, frc, mass, noise, gamma, kT, dt)
    assert torch.allclose(vel, torch.tensor([10.0 / 3.0], device="cuda"), atol=1e-5)

def test_pure_noise_kick():
    """Verify the amplitude of the thermal noise kick."""
    dt = 0.5
    gamma = 1.0
    kT = 1.0
    m = 1.0
    # Start at rest, no forces, known noise tensor
    pos, vel, frc, mass, noise1 = make_tensors([0.0], [0.0], [0.0], [m], [1.0])
    
    integrator = LangevinIntegrator()
    
    # c2 = sqrt(gamma * kT * dt / m) = sqrt(0.5) ≈ 0.7071
    # halfstep1: v_half = c2 * noise1 = 0.7071
    #        pos = 0.7071 * 0.5 = 0.3535
    integrator.halfstep1(pos, vel, frc, mass, noise1, gamma, kT, dt)
    
    expected_v_half = math.sqrt(0.5)
    expected_pos = expected_v_half * 0.5
    assert torch.allclose(vel, torch.tensor([expected_v_half], device="cuda"), atol=1e-5)
    assert torch.allclose(pos, torch.tensor([expected_pos], device="cuda"), atol=1e-5)

    # halfstep2: zero noise for the second half step
    noise2 = make_tensors([0.0])[0]
    # v_new = v_half / (1 + gamma*dt/2) = 0.7071 / 1.25 = 0.5657
    integrator.halfstep2(vel, frc, mass, noise2, gamma, kT, dt)
    
    expected_v_new = expected_v_half / 1.25
    assert torch.allclose(vel, torch.tensor([expected_v_new], device="cuda"), atol=1e-5)

# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """halfstep1 and halfstep2 must modify original tensors."""
    dt = 0.1
    pos, vel, frc, mass, noise = make_tensors([0.0], [1.0], [0.0], [1.0], [0.0])
    pos_id = id(pos)
    vel_id = id(vel)

    integrator = LangevinIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, noise, 1.0, 1.0, dt)
    integrator.halfstep2(vel, frc, mass, noise, 1.0, 1.0, dt)

    assert id(pos) == pos_id
    assert id(vel) == vel_id
