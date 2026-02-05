"""Copied from: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/7de0d20c8f26df706e2c7b3a21ceaf0b3542c953/models/losses.py
Converted stablemax_cross_entropy function into an nn.Module."""

import torch
from torch import nn
from typing import Optional


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
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
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

        transformed_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        prediction_logprobs = torch.gather(
            logprobs,
            dim=-1,
            index=transformed_labels.to(torch.long).unsqueeze(-1),
        ).squeeze(-1)

        loss = -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))
        return loss.sum() / logits.shape[0]
