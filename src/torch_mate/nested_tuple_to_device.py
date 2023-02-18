from typing import Iterable

import torch

def nested_tuple_to_device(nested_tuple: Iterable[Iterable[torch.Tensor]],
                           device: torch.device):
    """Move all elements of a nested tuple to a device.

    Args:
        nested_tuple (Iterable[Iterable[torch.Tensor]]): Nested tuple of tensors to move.
        device (torch.device): Device to move the tensors to.

    Returns:
        tuple[tuple[Tensor]]: Nested tuple of tensors moved to the device.
    """

    return tuple(tuple(e.to(device) for e in elem) for elem in nested_tuple)