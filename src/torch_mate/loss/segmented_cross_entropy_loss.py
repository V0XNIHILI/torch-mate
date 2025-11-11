import torch
import torch.nn as nn
from typing import List, Optional


class SegmentedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        segments: List[int],
        reduction: str = "mean",
        ce_kwargs: Optional[dict] = None
    ) -> None:
        """
        Segmented cross-entropy loss.

        Args:
            segments: list of segment lengths, e.g. [3, 5, 2]
                      first segment: logits[:, 0:3], second: logits[:, 3:8], third: logits[:, 8:10]
            reduction: 'mean' or 'sum' applied after summing segment losses.
        """
        super().__init__()

        self.segments = segments
        self.reduction = reduction

        self.loss_fn = nn.CrossEntropyLoss(**(ce_kwargs or {}))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        total_loss = 0.0
        start = 0

        for i, seg_len in enumerate(self.segments):
            end = start + seg_len

            seg_logits = logits[:, start:end]
            seg_loss = self.loss_fn(seg_logits, targets[:, i])
            total_loss += seg_loss.mean()

            start = end

        if self.reduction == "mean":
            return total_loss / len(self.segments)
        
        return total_loss
