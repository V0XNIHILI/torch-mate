import torch


def calc_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Calculate the accuracy of the predictions.

    Args:
        predictions (torch.Tensor): Predictions. Shape: (batch_size, num_classes)
        targets (torch.Tensor): Targets. Shape: (batch_size,) or (batch_size, num_classes)
        k (int, optional): Top-k accuracy. Defaults to 1.

    Returns:
        torch.Tensor: Accuracy.
    """

    _, top_k_indices = predictions.topk(k, dim=1)

    if len(targets.size()) == 2:
        targets = targets.argmax(dim=1)
    else:
        targets = targets.repeat(k, 1).t()

    return (top_k_indices == targets).sum().float() / targets.size(0)
