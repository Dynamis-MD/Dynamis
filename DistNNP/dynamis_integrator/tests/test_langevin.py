import pytest
import math
import torch
from dynamis_integrator.dynamis_integrator_impl import LangevinIntegrator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

def make_atoms(pos_list, vel_list, frc_list, mass_list, noise_list, device="cuda"):
    """
    Create MD tensors with correct 3D shapes.

    pos/vel/frc/noise: shape [N, 3]
    mass:              shape [N]
    """
    pos   = torch.tensor(pos_list,   dtype=torch.float32, device=device).contiguous()
    vel   = torch.tensor(vel_list,   dtype=torch.float32, device=device).contiguous()
    frc   = torch.tensor(frc_list,   dtype=torch.float32, device=device).contiguous()
    mass  = torch.tensor(mass_list,  dtype=torch.float32, device=device).contiguous()
    noise = torch.tensor(noise_list, dtype=torch.float32, device=device).contiguous()
    return pos, vel, frc, mass, noise

# ---------------------------------------------------------------------------
# Correctness tests — verify the BBK Langevin equations analytically
# ---------------------------------------------------------------------------

def test_zero_friction_matches_verlet():
    """With gamma=0 and noise=0, Langevin should exactly match Velocity Verlet."""
    dt = 0.5
    gamma = 0.0
    kT = 1.0
    pos, vel, frc, mass, noise = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[2., 0., 0.]],
        [1.],
        [[0., 0., 0.]],
    )

    integrator = LangevinIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, noise, gamma, kT, dt)
    integrator.halfstep2(vel, frc, mass, noise, gamma, kT, dt)

    # Identical to Verlet with Fx=2, m=1, dt=0.5:
    #   v_x_final = 1.0,  pos_x = 0.25,  y/z unchanged
    assert torch.allclose(vel, torch.tensor([[1., 0., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[0.25, 0., 0.]], device="cuda"))


def test_friction_damping():
    """Velocity decays due to friction with no forces and no noise."""
    dt = 0.5
    gamma = 2.0
    kT = 0.0
    pos, vel, frc, mass, noise = make_atoms(
        [[0., 0., 0.]],
        [[10., 0., 0.]],
        [[0., 0., 0.]],
        [1.],
        [[0., 0., 0.]],
    )

    integrator = LangevinIntegrator()

    # halfstep1: v_x_half = 10 * (1 - gamma*dt/2) = 10*(1-0.5) = 5.0
    #            pos_x     = 0 + 5.0*0.5 = 2.5
    integrator.halfstep1(pos, vel, frc, mass, noise, gamma, kT, dt)
    assert torch.allclose(vel[:, 0], torch.tensor([5.0], device="cuda"))
    assert torch.allclose(pos[:, 0], torch.tensor([2.5], device="cuda"))
    assert torch.allclose(vel[:, 1:], torch.zeros(1, 2, device="cuda"))

    # halfstep2: v_x_new = 5.0 / (1 + gamma*dt/2) = 5.0 / 1.5 = 3.333...
    integrator.halfstep2(vel, frc, mass, noise, gamma, kT, dt)
    assert torch.allclose(vel[:, 0], torch.tensor([10.0 / 3.0], device="cuda"), atol=1e-5)


def test_pure_noise_kick():
    """Verify the amplitude of the thermal noise kick per DOF."""
    dt = 0.5
    gamma = 1.0
    kT = 1.0
    m = 1.0
    # Unit noise in x only; y and z noise = 0
    pos, vel, frc, mass, noise1 = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [m],
        [[1., 0., 0.]],
    )

    integrator = LangevinIntegrator()

    # c2 = sqrt(gamma * kT * dt / m) = sqrt(0.5) ≈ 0.7071
    # halfstep1: v_x_half = c2 * 1.0 = 0.7071
    #            pos_x    = 0.7071 * 0.5 = 0.3535
    integrator.halfstep1(pos, vel, frc, mass, noise1, gamma, kT, dt)

    c2 = math.sqrt(0.5)
    assert torch.allclose(vel[:, 0], torch.tensor([c2],      device="cuda"), atol=1e-5)
    assert torch.allclose(pos[:, 0], torch.tensor([c2 * dt], device="cuda"), atol=1e-5)
    assert torch.allclose(vel[:, 1:], torch.zeros(1, 2, device="cuda"))

    # halfstep2 with zero noise: v_x_new = v_x_half / (1 + gamma*dt/2) = c2 / 1.25
    noise2 = torch.zeros(1, 3, dtype=torch.float32, device="cuda")
    integrator.halfstep2(vel, frc, mass, noise2, gamma, kT, dt)

    assert torch.allclose(vel[:, 0], torch.tensor([c2 / 1.25], device="cuda"), atol=1e-5)


def test_mass_indexed_by_atom_not_dof():
    """
    Each atom's scalar mass must be broadcast to all three of its DOFs.

    Two atoms with different masses; noise=0, gamma=0 reduces to Verlet so
    the expected values are analytically simple.  A kernel that uses masses[idx]
    instead of masses[idx/3] would apply atom 1's mass to atom 0's y/z DOFs
    and skip atom 1's DOFs entirely.
    """
    dt = 1.0
    # atom 0: m=1, F=[2,2,2]  → a=2,  v_half=1.0
    # atom 1: m=4, F=[4,4,4]  → a=1,  v_half=0.5
    pos, vel, frc, mass, noise = make_atoms(
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[2., 2., 2.], [4., 4., 4.]],
        [1., 4.],
        [[0., 0., 0.], [0., 0., 0.]],
    )

    integrator = LangevinIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, noise, 0.0, 0.0, dt)

    assert torch.allclose(vel, torch.tensor([[1., 1., 1.], [0.5, 0.5, 0.5]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[1., 1., 1.], [0.5, 0.5, 0.5]], device="cuda"))

    integrator.halfstep2(vel, frc, mass, noise, 0.0, 0.0, dt)

    assert torch.allclose(vel, torch.tensor([[2., 2., 2.], [1., 1., 1.]], device="cuda"))


# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """halfstep1 and halfstep2 must modify original tensors in-place."""
    dt = 0.1
    pos, vel, frc, mass, noise = make_atoms(
        [[0., 0., 0.]],
        [[1., 0., 0.]],
        [[0., 0., 0.]],
        [1.],
        [[0., 0., 0.]],
    )
    pos_id = id(pos)
    vel_id = id(vel)

    integrator = LangevinIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, noise, 1.0, 1.0, dt)
    integrator.halfstep2(vel, frc, mass, noise, 1.0, 1.0, dt)

    assert id(pos) == pos_id
    assert id(vel) == vel_id
