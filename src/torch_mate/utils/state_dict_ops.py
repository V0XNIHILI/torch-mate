from typing import OrderedDict

import torch


def add_state_dicts(state_dict_a: OrderedDict[str, torch.Tensor],
                    state_dict_b: OrderedDict[str, torch.Tensor]):
    """Add two state dicts in place of the first one.

    Args:
        state_dict_a (OrderedDict[str, torch.Tensor]): State dict to be added to.
        state_dict_b (OrderedDict[str, torch.Tensor]): State dict to be added.
    """

    for param_a, param_b in zip(state_dict_a, state_dict_b):
        state_dict_a[param_a].data.add_(state_dict_b[param_b].data)


def div_state_dict(state_dict: OrderedDict[str, torch.Tensor], divisor: float):
    """Divide a state dict in place.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): State dict to be divided.
        divisor (float): Divisor.
    """

    for param in state_dict:
        if state_dict[param].data.is_floating_point():
            state_dict[param].data.div_(divisor)
        else:
            # Perform in-place integer division
            state_dict[param].data.div_(divisor, rounding_mode='trunc')


def set_state_dict_to_zero(state_dict: OrderedDict[str, torch.Tensor]):
    """Set a state dict to zero in place.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): State dict to be set to zero.
    """

    for param in state_dict:
        state_dict[param].data.zero_()
