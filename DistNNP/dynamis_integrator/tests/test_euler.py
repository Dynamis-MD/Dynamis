import pytest
import torch
from dynamis_integrator.dynamis_integrator_impl import EulerIntegrator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

def make_atoms(pos_list, vel_list, frc_list, mass_list, device="cuda"):
    """
    Create MD tensors with correct 3D shapes.

    pos/vel/frc: shape [N, 3]  (size(0) == number of atoms)
    mass:        shape [N]
    """
    pos  = torch.tensor(pos_list,  dtype=torch.float32, device=device).contiguous()
    vel  = torch.tensor(vel_list,  dtype=torch.float32, device=device).contiguous()
    frc  = torch.tensor(frc_list,  dtype=torch.float32, device=device).contiguous()
    mass = torch.tensor(mass_list, dtype=torch.float32, device=device).contiguous()
    return pos, vel, frc, mass


# ---------------------------------------------------------------------------
# Correctness tests — verify the Euler equations analytically
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
    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([[1., 0., 0.], [0., 2., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[0.1, 0., 0.], [0., 0.2, 0.]], device="cuda"))


def test_constant_force_unit_mass():
    """Single atom, Fx=2, m=1, dt=0.5 → v_x_new=1.0, x_new=0.5."""
    dt = 0.5
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[2., 0., 0.]],
        [1.],
    )
    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # v_x(t+dt) = 0 + (2/1)*0.5 = 1.0
    # x(t+dt)   = 0 + 1.0*0.5   = 0.5
    assert torch.allclose(vel, torch.tensor([[1., 0., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[0.5, 0., 0.]], device="cuda"))


def test_force_divided_by_mass():
    """Heavier atom should accelerate less under the same force."""
    dt = 1.0
    # Two atoms: both receive Fx=4, but atom 0 has m=1 and atom 1 has m=2.
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
        [[4., 4., 4.], [4., 4., 4.]],
        [1., 2.],
    )
    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # atom 0: a = 4/1 = 4;  v = [4,4,4],  x = [4,4,4]
    # atom 1: a = 4/2 = 2;  v = [2,2,2],  x = [2,2,2]
    assert torch.allclose(vel, torch.tensor([[4., 4., 4.], [2., 2., 2.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[4., 4., 4.], [2., 2., 2.]], device="cuda"))


def test_nonzero_initial_velocity():
    """Starting velocity contributes to the position update."""
    dt = 0.5
    # v0_x=3, Fx=2, m=1  →  v_new_x = 3 + (2/1)*0.5 = 4,  x_new = 1 + 4*0.5 = 3
    pos, vel, frc, mass = make_atoms(
        [[1., 0., 0.]],
        [[3., 0., 0.]],
        [[2., 0., 0.]],
        [1.],
    )
    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert torch.allclose(vel, torch.tensor([[4., 0., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[3., 0., 0.]], device="cuda"))


def test_negative_force():
    """Negative force should decelerate the atom."""
    dt = 1.0
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[5., 0., 0.]],
        [[-2., 0., 0.]],
        [1.],
    )
    EulerIntegrator().step(pos, vel, frc, mass, dt)

    # v_x_new = 5 + (-2/1)*1 = 3.0
    # x_new   = 0 + 3.0*1   = 3.0
    assert torch.allclose(vel, torch.tensor([[3., 0., 0.]], device="cuda"))
    assert torch.allclose(pos, torch.tensor([[3., 0., 0.]], device="cuda"))


# ---------------------------------------------------------------------------
# Integration behaviour tests
# ---------------------------------------------------------------------------

def test_in_place_modification():
    """step() must modify the original position and velocity tensors."""
    dt = 0.1
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[1., 0., 0.]],
        [[0., 0., 0.]],
        [1.],
    )
    pos_id = id(pos)
    vel_id = id(vel)

    EulerIntegrator().step(pos, vel, frc, mass, dt)

    assert id(pos) == pos_id
    assert id(vel) == vel_id
    assert not torch.allclose(pos, torch.zeros(1, 3, device="cuda"))


def test_multiple_steps_accumulate():
    """Repeated calls to step() should accumulate correctly."""
    dt = 0.1
    # Fx=1, m=1: each step adds 0.1 to v_x
    pos, vel, frc, mass = make_atoms(
        [[0., 0., 0.]],
        [[0., 0., 0.]],
        [[1., 0., 0.]],
        [1.],
    )
    integrator = EulerIntegrator()

    integrator.step(pos, vel, frc, mass, dt)  # v_x=0.1, x=0.01
    integrator.step(pos, vel, frc, mass, dt)  # v_x=0.2, x=0.03

    assert torch.allclose(vel[:, 0], torch.tensor([0.2],  device="cuda"), atol=1e-6)
    assert torch.allclose(pos[:, 0], torch.tensor([0.03], device="cuda"), atol=1e-6)
    assert torch.allclose(vel[:, 1:], torch.zeros(1, 2, device="cuda"))


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

    EulerIntegrator().step(pos, vel, frc, mass, 0.1)

    # x: v_x_new = 1.0 + (2/1)*0.1 = 1.2
    #    pos_x   = 0.0 + 1.2*0.1    = 0.12
    # y, z: unchanged (zero)
    assert torch.allclose(vel[:, 0], torch.full((n,), 1.2,  device=device), atol=1e-5)
    assert torch.allclose(pos[:, 0], torch.full((n,), 0.12, device=device), atol=1e-5)
    assert torch.allclose(vel[:, 1:], torch.zeros(n, 2, device=device))
    assert torch.allclose(pos[:, 1:], torch.zeros(n, 2, device=device))
