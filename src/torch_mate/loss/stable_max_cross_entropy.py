"""Copied from: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/7de0d20c8f26df706e2c7b3a21ceaf0b3542c953/models/losses.py
Converted stablemax_cross_entropy function into an nn.Module."""

from typing import Optional, Literal

import torch
from torch import nn


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


class StableMaxCrossEntropy(nn.Module):
    def __init__(self, reduction: Literal["sum", "mean"], ignore_index: int = -100) -> None:
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

        if valid_mask is None:
            valid_mask = labels != self.ignore_index

        valid_counts = valid_mask.sum(-1)
        loss_divisor = valid_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

        transformed_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        prediction_logprobs = torch.gather(
            logprobs,
            dim=-1,
            index=transformed_labels.to(torch.long).unsqueeze(-1),
        ).squeeze(-1)

        loss = -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))
        loss = (loss / loss_divisor).sum()

        if self.reduction == "mean":
            return loss / logits.shape[0]

        return loss
