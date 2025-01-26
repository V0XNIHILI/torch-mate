import random

import torch
from torch.utils.data import Dataset

from torch_mate.data.utils.Triplet import Triplet

class Siamese(Triplet):
    def __init__(self, dataset: Dataset, same_prob: float = 0.5):
        super().__init__(dataset)

        self.same_prob = same_prob

    def generate(self):
        same_label = random.random() < self.same_prob

        samples, _ = super().generate()

        if same_label:
            samples = samples[:2]
        else:
            samples = samples[0::2]
        
        return samples, same_label
