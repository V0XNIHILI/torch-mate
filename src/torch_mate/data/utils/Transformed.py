from typing import Callable, Optional

from torch.utils.data import Dataset

class Transformed(Dataset):

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        """Add a transform to a dataset that does not support transforms.

        Args:
            dataset (Dataset): The dataset to use.
            transform (Optional[Callable], optional): Transform to apply to each sample's data. Defaults to None.
        """

        self.transform = transform

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        X, y = self.dataset[index]

        if self.transform is not None:
            X = self.transform(X)

        return X, y
