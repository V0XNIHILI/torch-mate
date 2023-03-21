from typing import Iterable

import torch


def iterate_to_device(sequence: Iterable[Iterable[torch.Tensor]],
                        device: torch.device,
                        non_blocking=False):
    """Iterate over a sequence of tensor tuples and move them to the device.

    Args:
        sequence (Iterable[Iterable[torch.Tensor]]): Sequence of tensor tuples to enumerate.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect. Defaults to False.

    Yields:
        list[Tensor]: A list of tensors moved to the device.
    """

    for elem in sequence:
        yield [e.to(device, non_blocking=non_blocking) for e in elem]

