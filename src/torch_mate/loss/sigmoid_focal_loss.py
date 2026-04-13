import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert targets to one-hot encoding
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1)).to(inputs.dtype)

        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
