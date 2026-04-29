import torch
import torch.nn as nn
from typing import Optional


class Slice(nn.Module):
    def __init__(
        self,
        dim: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slices = [slice(None)] * x.dim()
        slices[self.dim] = slice(self.start, self.end, self.step)
        return x[tuple(slices)]
