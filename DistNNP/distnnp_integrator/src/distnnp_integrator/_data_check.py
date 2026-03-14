import torch


def check_devices(*args):
    """
    Check that all input tensors are on the same device.

    Raises:
        ValueError: If the tensors are not all on the same device.
    """
    devices = {arg.device for arg in args}

    if len(devices) > 1:
        raise ValueError(
            f"All input tensors must be on the same device. Found devices: {devices}"
        )


def check_shape_2D(pos):
    if len(pos.shape) != 2:
        raise ValueError(
            f"Position tensor must be 2D of shape (N, D). Got: {pos.shape}"
        )


def check_mass_shape(mass, pos):
    if mass.shape != (pos.shape[0],):
        raise ValueError(
            f"Mass tensor must have shape (N,), where N is the number of atoms. "
            f"Got mass: {mass.shape}, expected: {(pos.shape[0],)}"
        )


def check_shapes_match(*args):
    """
    Check that input tensors have compatible shapes.

    Args:
        pos (torch.Tensor): Positions of shape (N, D).
        vel (torch.Tensor): Velocities of shape (N, D).
        frc (torch.Tensor): Forces of shape (N, D).

    Raises:
        ValueError: If the tensor shapes are not compatible.
    """
    all_shapes = {arg.shape for arg in args}

    if len(all_shapes) > 1:
        raise ValueError(
            f"All input tensors must have the same shape. Found shapes: {all_shapes}"
        )
