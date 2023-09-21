import torch.nn as nn


def count_parameters(model: nn.Module, biases_only: bool = False) -> int:
    """Counts the number of parameters in a model.

    Implementation copied from: # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

    Args:
        model (nn.Module): The model to count the parameters of.
        biases_only (bool, optional): Whether to only count biases. Defaults to False.

    Returns:
        int: Total number of parameters in the model.
    """

    total_parameters = 0

    for name, p in model.named_parameters():
        if p.requires_grad and ((biases_only and name.endswith(".bias") or not biases_only)):
            total_parameters += p.numel()

    return total_parameters
