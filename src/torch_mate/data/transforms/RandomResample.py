from typing import Tuple

import random

import torch
from torchaudio.functional import resample

class RandomResample:
    def __init__(self, sample_rate: int, resample_range: Tuple[int, int]):
        self.sample_rate = sample_rate
        self.resample_range = resample_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        new_sample_rate = random.randint(*self.resample_range)
        return resample(waveform, self.sample_rate, new_sample_rate)