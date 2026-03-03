import torch


def check_devices(pos, vel, frc, mass):
    """
    Check that all input tensors are on the same device.

    Args:
        pos (torch.Tensor): Positions of shape (N, D).
        vel (torch.Tensor): Velocities of shape (N, D).
        frc (torch.Tensor): Forces of shape (N, D).
        mass (torch.Tensor): Masses of shape (N,).

    Raises:
        ValueError: If the tensors are not all on the same device.
    """
    devices = {pos.device, vel.device, frc.device, mass.device}
    if len(devices) > 1:
        raise ValueError(
            f"All input tensors must be on the same device. Found devices: {devices}"
        )


def check_shapes(pos, vel, frc, mass):
    """
    Check that input tensors have compatible shapes.

    Args:
        pos (torch.Tensor): Positions of shape (N, D).
        vel (torch.Tensor): Velocities of shape (N, D).
        frc (torch.Tensor): Forces of shape (N, D).
        mass (torch.Tensor): Masses of shape (N,).

    Raises:
        ValueError: If the tensor shapes are not compatible.
    """
    if len(pos.shape) != 2:
        raise ValueError(
            f"Position tensor must be 2D of shape (N, D). Got: {pos.shape}"
        )

    if pos.shape != vel.shape or pos.shape != frc.shape:
        raise ValueError(
            f"Position, velocity, and force tensors must have the same shape. "
            f"Got pos: {pos.shape}, vel: {vel.shape}, frc: {frc.shape}"
        )
    if mass.shape != (pos.shape[0],):
        raise ValueError(
            f"Mass tensor must have shape (N,), where N is the number of atoms. "
            f"Got mass: {mass.shape}, expected: {(pos.shape[0],)}"
        )
