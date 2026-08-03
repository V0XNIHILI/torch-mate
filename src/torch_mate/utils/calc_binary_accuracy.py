import torch


def calc_binary_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = torch.round(torch.sigmoid(predictions))

    correct = (predictions == targets).float()
    acc = correct.sum() / len(correct)

    return acc
