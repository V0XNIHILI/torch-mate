import torch


def calc_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Calculate the accuracy of the predictions.

    Args:
        predictions (torch.Tensor): Predictions. Shape: (batch_size, num_classes)
        targets (torch.Tensor): Targets. Shape: (batch_size)
        k (int, optional): Top-k accuracy. Defaults to 1.

    Returns:
        torch.Tensor: Accuracy.
    """

    _, top_k_indices = predictions.topk(k, dim=1)
    repeated_targets = targets.repeat(k, 1).t()

    return (top_k_indices == repeated_targets).sum().float() / targets.size(0)
