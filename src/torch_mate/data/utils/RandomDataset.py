import torch

from torch.utils.data import Dataset

class RandomDataset(Dataset):
    """Empty dataset that always returns the same label."""

    def __init__(self, total_classes = 100, samples_per_class = 100, sample_shape = (3, 4, 4)):
        self.total_classes = total_classes
        self.samples_per_class = samples_per_class

        self.X = torch.randn(total_classes * samples_per_class, *sample_shape)
        self.y = torch.arange(total_classes).repeat(samples_per_class)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.total_classes * self.samples_per_class
