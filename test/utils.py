import torch
from torch.utils.data import Dataset

class LabelsAreData(Dataset):
    def __init__(self, labels, length, transposed=False):
        super().__init__()

        self.labels = labels
        self.length = length
        self.transposed = transposed

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of length {len(self)}")

        if self.transposed:
            index = index // (len(self) // len(self.labels))
        else:
            index = index % len(self.labels)

        item = self.labels[index]

        return (torch.tensor(item), item)

    def __len__(self):
        return self.length
