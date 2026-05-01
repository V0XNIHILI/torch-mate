import torch
import torch.nn as nn
from typing import Union


class Scale(nn.Module):
    def __init__(self, scale: Union[float, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
