from typing import Tuple

import random

import torch
from torchaudio.functional import resample


class RandomResample:
    def __init__(self, orig_freq: int, resample_range: Tuple[int, int]):
        self.orig_freq = orig_freq
        self.resample_range = resample_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        new_sample_freq = random.randint(*self.resample_range)
        resampled_waveform = resample(waveform, self.orig_freq, new_sample_freq)

        # Pad with zeros at the start if the resampled waveform is shorter,
        # otherwise, cut the waveform
        
        if resampled_waveform.size(1) < waveform.size(1):
            resampled_waveform = torch.cat(
                (torch.zeros(waveform.size(0), waveform.size(1) - resampled_waveform.size(1), device=waveform.device), resampled_waveform),
                dim=1
            )
        else:
            resampled_waveform = resampled_waveform[:, :waveform.size(1)]

        return resampled_waveform
