import pytest
import torch
from distnnp_integrator.distnnp_integrator_impl import VerletIntegrator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

def make_atoms(pos_list, vel_list, frc_list, mass_list, device="cuda"):
    """
    Create MD tensors with correct 3D shapes.

    pos/vel/frc: shape [N, 3]  (positions.size(0) == number of atoms)
    mass:        shape [N]
    """
    pos  = torch.tensor(pos_list,  dtype=torch.float32, device=device).contiguous()
    vel  = torch.tensor(vel_list,  dtype=torch.float32, device=device).contiguous()
    frc  = torch.tensor(frc_list,  dtype=torch.float32, device=device).contiguous()
    mass = torch.tensor(mass_list, dtype=torch.float32, device=device).contiguous()
    return pos, vel, frc, mass

# ---------------------------------------------------------------------------
# Correctness tests — verify the Velocity Verlet equations analytically
# ---------------------------------------------------------------------------

def test_zero_force_velocity_unchanged():
    """With F=0, velocity must not change and position advances by v*dt."""
    dt = 0.1
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.], [0., 0., 0.]],
        [[1., 0., 0.], [0., 2., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [1., 1.],
    )
    integrator = VerletIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, dt)
    integrator.halfstep2(vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([[1., 0., 0.], [0., 2., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[0.1, 0., 0.], [0., 0.2, 0.]], device="cuda"))


def test_constant_force_unit_mass():
    """Single atom, Fx=2, m=1, dt=0.5."""
    dt = 0.5
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[2., 0., 0.]],
        [1.],
    )
    integrator = VerletIntegrator()

    integrator.halfstep1(pos, vel, frc, mass, dt)

    # v_x(t + dt/2) = 0 + 0.5*(2/1)*0.5 = 0.5
    # x(t + dt)     = 0 + 0.5*0.5       = 0.25
    # y, z unchanged
    assert torch.allclose(vel, torch.tensor([[0.5, 0., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[0.25, 0., 0.]], device="cuda"))

    integrator.halfstep2(vel, frc, mass, dt)

    # v_x(t + dt) = 0.5 + 0.5*(2/1)*0.5 = 1.0
    assert torch.allclose(vel, torch.tensor([[1., 0., 0.]], device="cuda"))


def test_mass_indexed_by_atom_not_dof():
    """
    Each atom's scalar mass must be broadcast to all three of its DOFs.

    Two atoms with different masses and forces.  If the kernel uses masses[idx]
    instead of masses[idx/3] the y- and z-components of atom 0 would pick up
    atom 1's mass (=4) and give wrong accelerations; atom 1's DOFs would be
    skipped entirely.
    """
    dt = 1.0
    # atom 0: m=1, F=[2,2,2]  → a=2,  v_half=1.0, pos=1.0
    # atom 1: m=4, F=[4,4,4]  → a=1,  v_half=0.5, pos=0.5
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[2., 2., 2.], [4., 4., 4.]],
        [1., 4.],
    )
    integrator = VerletIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([[1., 1., 1.], [0.5, 0.5, 0.5]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[1., 1., 1.], [0.5, 0.5, 0.5]], device="cuda"))

    integrator.halfstep2(vel, frc, mass, dt)

    # v_final: atom 0 → 1.0 + 0.5*(2/1)*1 = 2.0
    #          atom 1 → 0.5 + 0.5*(4/4)*1 = 1.0
    assert torch.allclose(vel, torch.tensor([[2., 2., 2.], [1., 1., 1.]], device="cuda"))


def test_multiple_steps_accumulate():
    """Repeated halfstep1/halfstep2 calls should accumulate correctly."""
    dt = 0.1
    # Single atom, Fx=1, m=1
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[1., 0., 0.]],
        [1.],
    )
    integrator = VerletIntegrator()

    # Step 1: v_half=0.05, x=0.005, v_end=0.1
    integrator.halfstep1(pos, vel, frc, mass, dt)
    integrator.halfstep2(vel, frc, mass, dt)

    # Step 2: v_half=0.15, x=0.005+0.15*0.1=0.02, v_end=0.2
    integrator.halfstep1(pos, vel, frc, mass, dt)
    integrator.halfstep2(vel, frc, mass, dt)

    assert torch.allclose(vel[:, 0], torch.tensor([0.2], device="cuda"), atol=1e-6)
    assert torch.allclose(pos[:, 0], torch.tensor([0.02], device="cuda"), atol=1e-6)
    assert torch.allclose(vel[:, 1:], torch.zeros(1, 2, device="cuda"))


# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """halfstep1 and halfstep2 must modify the original tensors in-place."""
    dt = 0.1
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[1., 0., 0.]],
        [[0., 0., 0.]],
        [1.],
    )
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
    pos  = torch.zeros(n, 3, dtype=torch.float32, device=device)
    vel  = torch.zeros(n, 3, dtype=torch.float32, device=device)
    vel[:, 0] = 1.0                                               # velocity in x only
    frc  = torch.zeros(n, 3, dtype=torch.float32, device=device)
    frc[:, 0] = 2.0                                               # force in x only
    mass = torch.ones(n, dtype=torch.float32, device=device)

    integrator = VerletIntegrator()
    integrator.halfstep1(pos, vel, frc, mass, 0.1)
    integrator.halfstep2(vel, frc, mass, 0.1)

    # x: v_new = 1.0 + (2/1)*0.1 = 1.2
    #    pos_x = (1.0 + 0.5*(2/1)*0.1)*0.1 = 1.1*0.1 = 0.11
    # y, z: unchanged (zero)
    assert torch.allclose(vel[:, 0], torch.full((n,), 1.2,  device=device), atol=1e-5)
    assert torch.allclose(pos[:, 0], torch.full((n,), 0.11, device=device), atol=1e-5)
    assert torch.allclose(vel[:, 1:], torch.zeros(n, 2, device=device))
    assert torch.allclose(pos[:, 1:], torch.zeros(n, 2, device=device))
