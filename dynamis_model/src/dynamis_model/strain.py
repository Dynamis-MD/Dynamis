"""Strain-derivative stress shared by every DynamisModel backend.

Forces come from ``F = -dE/dx``; stress comes from the same autograd pass by
differentiating the energy with respect to a homogeneous strain ``eps`` applied
to every position and to the cell::

    x' = x + x @ eps          (row vectors: x' = x (1 + eps))
    h' = h + h @ eps          (rows of h are the lattice vectors)

    stress = (1 / V) dE/d eps |_{eps = 0}

This is the ASE convention: positive stress is tensile, and the pressure is
``-trace(stress) / 3``. The kinetic contribution is NOT included; the barostat
adds it from the velocities.

Only fully periodic systems have a physical stress. With ``pbc`` given, strain
components involving a non-periodic axis are masked to zero, so those stress
entries are exactly zero, and the periodic block is normalised by the cell
volume, which for a slab depends on the vacuum height. A barostat must couple
only the periodic block and refuse a system with no periodic axis. Units follow the energy the model returns, so a
backend returning amu * A^2 / fs^2 yields stress in amu / (A * fs^2).

Backend recipe (one strain leaf per call, applied *before* the position
exchange so ghosts inherit it from their owners)::

    strain = zero_strain(device, dtype)
    strained_pos, strained_cell = apply_strain(local_positions, cell, strain)
    ...exchange strained_pos, build edge vectors from strained_pos and
       unit_shifts @ strained_cell, evaluate the owned energy...
    grad_pos, grad_strain = torch.autograd.grad(E, [local_positions, strain])
    stress = stress_from_strain_grad(grad_strain, cell)

Distributed: every rank differentiates its own owned energy with respect to its
own strain leaf. The backward of the halo exchange routes the gradient that
other ranks' energies place on this rank's atoms into this rank's leaf, exactly
as it already does for forces, so the sum of the per-rank gradients is
``dE_total/d eps`` and ``stress_from_strain_grad`` all-reduces it.
"""

import torch
import torch.distributed as dist


def zero_strain(device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A zero 3x3 strain leaf that records gradients."""
    return torch.zeros(3, 3, device=device, dtype=dtype, requires_grad=True)


def _symmetrise(strain: torch.Tensor, pbc=None) -> torch.Tensor:
    """Symmetric part of ``strain``; with ``pbc``, only periodic x periodic entries.

    A component that involves a non-periodic axis would rescale vacuum or tilt a
    lattice vector out of its plane, and its stress has no volume to divide by.
    Masking it makes the derivative exactly zero, so no consumer can act on it.
    """
    sym = 0.5 * (strain + strain.transpose(-1, -2))
    if pbc is not None:
        periodic = torch.as_tensor(pbc, device=sym.device).reshape(3).to(sym.dtype)
        sym = sym * torch.outer(periodic, periodic)
    return sym


def strain_cell(cell: torch.Tensor, strain: torch.Tensor, pbc=None) -> torch.Tensor:
    """The strained cell ``[3, 3]``, in the strain's dtype."""
    cell = cell.reshape(3, 3).to(dtype=strain.dtype)
    return cell + cell @ _symmetrise(strain, pbc)


def apply_strain(
    positions: torch.Tensor, cell: torch.Tensor, strain: torch.Tensor, pbc=None
):
    """Apply a symmetric strain to positions ``[N, 3]`` and cell ``[3, 3]``.

    The strain is symmetrised so the gradient is a symmetric stress and a
    backend cannot pick up a rotation. At ``strain == 0`` the outputs equal the
    inputs and the gradient with respect to ``positions`` is the identity, so
    forces are unchanged. Pass ``pbc`` (bool, 3 entries) to restrict the strain
    to the periodic axes; ``None`` means fully periodic.
    """
    return (
        positions + positions @ _symmetrise(strain, pbc),
        strain_cell(cell, strain, pbc),
    )


def strain_grad_from_gradients(
    positions: torch.Tensor,
    position_grad: torch.Tensor,
    cell: torch.Tensor,
    cell_grad: torch.Tensor,
    pbc=None,
) -> torch.Tensor:
    """``dE/d eps`` assembled from the gradients at the positions and the cell.

    For ``x' = x (1 + eps)`` and ``h' = h (1 + eps)`` the chain rule gives
    ``dE/d eps = x^T (dE/dx) + h^T (dE/dh)``, symmetrised and ``pbc``-masked
    exactly like :func:`apply_strain`. This is the route for a backend whose
    library differentiates the energy with respect to positions and cell but
    frees the graph afterwards, so a strain leaf cannot be added from outside
    (UMA). ``position_grad`` must be the *total* gradient on these positions,
    i.e. already routed from ghost copies to their owners.
    """
    x = positions.to(torch.float64)
    h = cell.reshape(3, 3).to(torch.float64)
    virial = x.T @ position_grad.to(torch.float64) + h.T @ cell_grad.reshape(3, 3).to(
        torch.float64
    )
    return _symmetrise(virial, pbc).to(positions.dtype)


def stress_from_strain_grad(
    strain_grad: torch.Tensor, cell: torch.Tensor, reduce: bool = True
) -> torch.Tensor:
    """Convert ``dE/d eps`` to stress: all-reduce over ranks, divide by volume.

    ``cell`` must be the unstrained cell. Pass ``reduce=False`` when the
    gradient already covers the whole system (a single-process reference).
    """
    if reduce and dist.is_available() and dist.is_initialized():
        if dist.get_world_size() > 1:
            strain_grad = strain_grad.clone()
            dist.all_reduce(strain_grad, op=dist.ReduceOp.SUM)
    volume = torch.linalg.det(cell.reshape(3, 3).to(torch.float64)).abs()
    return (strain_grad.to(torch.float64) / volume).to(strain_grad.dtype)


__all__ = [
    "apply_strain",
    "strain_cell",
    "strain_grad_from_gradients",
    "stress_from_strain_grad",
    "zero_strain",
]
