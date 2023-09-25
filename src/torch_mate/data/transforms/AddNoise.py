import random
from typing import List

import torch


class AddNoise:

    def __init__(self,
                 noise_samples: List[torch.Tensor],
                 noise_class_indices: List[int],
                 p: float = 0.8,
                 max_noise_level: float = 1.0):
        """Randomly add noise to a data sample, where noise is only added if the input sample is not
        already a noise signal itself. Data is clipped at (-1, 1) when noise is added.

        Args:
            noise_samples (List[torch.Tensor]): List of noisy data to add to input data
            noise_class_indices (List[int]): Indices of classes that represent noise classes
            p (float, optional): Probability of applying noise. Defaults to 0.8.
            max_noise_level (float, optional): Maximum noise level multiplier. Defaults to 1.0.
        """
        self.noise_samples = noise_samples
        self.noise_class_indices = noise_class_indices

        self.p = p
        self.max_noise_level = max_noise_level

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        noise_level = random.random() * self.max_noise_level

        # https://github.com/castorini/honk/blob/c3aae750c428520ba340961bddd526f9c999bb93/utils/model.py#L301
        if not y in self.noise_class_indices:
            if random.random() < self.p:
                bg_noise = random.choice(self.noise_samples)

                return torch.clip(noise_level * bg_noise + x, -1, 1)
            else:
                return x
        else:
            return torch.clip(noise_level * x, -1, 1)
