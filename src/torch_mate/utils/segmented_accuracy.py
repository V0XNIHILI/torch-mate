from typing import List
from itertools import accumulate

import torch


class SegmentedAccuracy:
    def __init__(self, segments: List[int]) -> None:
        """
        Computes accuracy for segmented outputs.
        An output is correct only if all segments are correct.

        Args:
            segments: list of segment lengths
        """
        self.segments = segments

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        # targets: list/tuple of length len(segments), each of shape (batch_size,)

        segment_starts = [0] + list(accumulate(self.segments[:-1]))
        segment_ends = list(accumulate(self.segments))
        
        correct = 0.0

        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            seg_logits = logits[:, start:end]
            seg_preds = seg_logits.argmax(dim=1)

            if i == 0:
                correct = (seg_preds == targets[i])
            else:
                correct &= (seg_preds == targets[i])

        return correct.sum().float() / logits.size(0)
