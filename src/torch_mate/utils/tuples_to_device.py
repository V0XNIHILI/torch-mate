import torch

from ..typing import NestedTupleOfTensors

def nested_tuple_to_device(item: NestedTupleOfTensors, device: torch.device, non_blocking=False):
    """Move a (nested) tuple of tensors to the device.

    Args:
        item (NestedTupleOfTensors): (Nested) tuple of tensors to move.
        device (torch.device): Device to move the tensors to.
        non_blocking (bool, optional): If True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect. Defaults to False.

    Returns:
        NestedTupleOfTensors: (Nested) tuple of tensors moved to the device.
    """

    if type(item) is tuple or type(item) is list:
        return tuple(nested_tuple_to_device(e, device, non_blocking) for e in item)
    else:
        return item.to(device, non_blocking=non_blocking)
