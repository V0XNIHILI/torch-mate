from torch.utils.data import Dataset

import random

from torch_mate.data.utils.Triplet import Triplet

class Siamese(Triplet):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def generate(self):
        same_label = random.random() < 0.5

        samples, _ = super().generate()

        if same_label:
            samples = samples[:2]
        else:
            samples = samples[0::2]
        
        return samples, same_label
