from typing import Iterable, Tuple, Union

import torch

TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]


def tuple_to_device(item: NestedTupleOfTensors, device: torch.device, non_blocking=False):
    """Move a (nested) tuple of tensors to the device.

    Args:
        item (NestedTupleOfTensors): (Nested) tuple of tensors to move.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect. Defaults to False.

    Returns:
        NestedTupleOfTensors: (Nested) tuple of tensors moved to the device.
    """

    if type(item) is tuple:
        return tuple(tuple_to_device(e, device, non_blocking) for e in item)
    else:
        return item.to(device, non_blocking=non_blocking)


def iterate_to_device(sequence: Iterable[NestedTupleOfTensors],
                        device: torch.device,
                        non_blocking=False):
    """Iterate over a sequence of tensor tuples and move them to the device.

    Args:
        sequence (NestedTupleOfTensors): Sequence of (nested) tensor tuples to enumerate.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect. Defaults to False.

    Yields:
        tuple[Tensor]: A tuple of tensors moved to the device.
    """

    for elem in sequence:
        elem_on_device = tuple([tuple_to_device(e, device, non_blocking) for e in elem])
        
        yield elem_on_device
