from typing import Iterable

import torch


def enumerate_to_device(sequence: Iterable[Iterable[torch.Tensor]],
                        device: torch.device,
                        start=0):
    """Enumerate an iterable and move all elements to a device. Enumerate code
    taken from: https://docs.python.org/3/library/functions.html#enumerate.

    Args:
        sequence (Iterable[Iterable[torch.Tensor]]): Sequence of tensor tuples to enumerate.
        device (torch.device): Device to move the tensors to.
        start (int, optional): Starting index. Defaults to 0.

    Yields:
        tuple[int, list[Tensor]: Tuple of the current index and a list of tensors moved to the device.
    """

    n = start

    for elem in sequence:
        yield n, [e.to(device) for e in elem]

        n += 1
