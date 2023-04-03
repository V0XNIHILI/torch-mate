from typing import Iterable, Tuple, Union

import torch

from torch_mate.nested_tuple_to_device import nested_tuple_to_device

TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]


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
        elem_on_device = tuple([nested_tuple_to_device(e, device, non_blocking) for e in elem])
        
        yield elem_on_device
