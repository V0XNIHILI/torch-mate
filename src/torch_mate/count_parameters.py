import torch.nn as nn


def count_parameters(model: nn.Module):
    """Counts the number of parameters in a model.

    Implementation copied from: # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

    Args:
        model (nn.Module): The model to count the parameters of.

    Returns:
        int: Total number of parameters in the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
