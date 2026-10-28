"""Homogeneous deformation of a DynamisState: positions and cell together.

Integrators only receive the raw ``DynamisState`` (never ``SimState``), so a
barostat changes the box through these functions. Both update tensors in place,
so every holder of ``state.sim_config.cell`` or the local positions (a model's
cached handle, a CUDA graph) keeps seeing the live values.

Convention (the same as ``dynamis_model.strain``): rows are vectors, and a
deformation ``A`` maps ``x -> x A`` and ``h -> h A``, so fractional coordinates
``x h^-1`` are preserved. Only the *state* is changed here. Velocities are the
barostat's business, and the halo is kept valid by the drift watchdog in
``SimState``, which accounts for the cell change (``write_local_drift_flag``).
"""

import torch


def deform_state(state, transform: torch.Tensor) -> None:
    """Apply ``x -> x A`` to the local positions and ``h -> h A`` to the cell.

    ``transform`` is a ``[3, 3]`` tensor, on the state's device for no host
    sync. It is applied as ``x += x (A - I)``, so ``A = I`` leaves both exactly
    unchanged. Intended for the hot path: no validation. Use
    :func:`set_state_cell` to install an arbitrary cell.
    """
    positions = state.local_molecular_info.atom_positions
    cell = state.sim_config.cell
    with torch.no_grad():
        change = transform.reshape(3, 3).to(device=cell.device, dtype=torch.float64)
        change = change - torch.eye(3, dtype=torch.float64, device=cell.device)
        h = cell.view(3, 3)
        h.add_((h.to(torch.float64) @ change).to(h.dtype))
        positions.add_(positions @ change.to(device=positions.device, dtype=positions.dtype))


def set_state_cell(state, cell: torch.Tensor) -> None:
    """Overwrite the cell in place, leaving the positions alone.

    This is not an affine deformation of the system (atoms keep their
    Cartesian positions), so the drift watchdog will see the atoms as displaced
    relative to the lattice and rebuild the halo when the change is large enough
    to matter. Raises on a malformed cell; this forces a host sync, so it is for
    setup, not the per-step path.
    """
    new = torch.as_tensor(cell)
    if new.numel() != 9:
        raise ValueError(f"cell must have 9 entries, got shape {tuple(new.shape)}")
    new = new.reshape(3, 3)
    if not bool(torch.isfinite(new).all()):
        raise ValueError("cell contains non-finite entries")
    if not float(torch.linalg.det(new.to(torch.float64))) > 0.0:
        raise ValueError("cell must be right-handed with positive volume")
    current = state.sim_config.cell
    with torch.no_grad():
        current.view(3, 3).copy_(new.to(device=current.device, dtype=current.dtype))


__all__ = ["deform_state", "set_state_cell"]
