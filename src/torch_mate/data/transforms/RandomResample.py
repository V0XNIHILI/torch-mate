from typing import List

import random

import torch
from torchaudio.transforms import Resample


class RandomResample:
    def __init__(self, orig_freq: int, resample_rates: List[float]):
        self.orig_freq = orig_freq
        self.resample_rates = resample_rates
        
        self.resamplers = [Resample(orig_freq, orig_freq*resample_rate) for resample_rate in resample_rates]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        resample = random.choice(self.resamplers)
        resampled_waveform = resample(waveform)

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
