import torch

def calc_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.argmax(dim=1).view(targets.shape)

    return (predictions == targets).sum().float() / targets.size(0)
