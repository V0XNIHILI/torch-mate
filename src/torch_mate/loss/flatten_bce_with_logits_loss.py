import torch
import torch.nn as nn


class FlattenBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    BCEWithLogitsLoss variant that flattens the input logits to [n,]
    before computing the loss, so shapes like [n, 1] match labels of shape [n,].
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.view(-1)
        return super().forward(input, target.float())
