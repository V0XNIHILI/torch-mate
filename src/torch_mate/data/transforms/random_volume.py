import random

import torch

from typing import Optional


class RandomVolume:
    def __init__(self, base_volume: float = 1.0, offset: Optional[float] = None):
        self.base_volume = base_volume
        self.offset = offset

        if self.offset:
            if self.offset > self.base_volume:
                raise ValueError("Offset cannot be larger than base volume.")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.offset:
            low, high = self.base_volume - self.offset, self.base_volume + self.offset
        else:
            low, high = 0.0, self.base_volume
        
        volume = random.uniform(low, high)

        return waveform * volume
