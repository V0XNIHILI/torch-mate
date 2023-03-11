from typing import Iterable

import torch


def iterate_to_device(sequence: Iterable[Iterable[torch.Tensor]],
                        device: torch.device,
                        non_blocking: bool = False):
    """Move all elements to a device while iterating.

    Args:
        sequence (Iterable[Iterable[torch.Tensor]]): Sequence of tensor tuples to enumerate.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and the source is in pinned memory, the copy will be asynchronous with respect to the host. Defaults to False.

    Yields:
        list[Tensor]: A list of tensors moved to the device.
    """

    for elem in sequence:
        yield [e.to(device, non_blocking=non_blocking) for e in elem]
