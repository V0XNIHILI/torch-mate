from typing import Callable, Optional

from torch.utils.data import Dataset
from tqdm import tqdm


class PreLoaded(Dataset):

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        """Dataset that pre-loads all samples from another dataset. Usage only
        advised if the dataset is small.

        Args:
            dataset (Dataset): The dataset to use. The dataset must be indexable.
            transform (Optional[Callable], optional): Transform to apply to each sample's data. Transforms will not be pre-applied. Defaults to None.
        """

        self.transform = transform

        all_samples = []

        for i in tqdm(range(len(dataset)), desc='Pre-loading samples'):
            all_samples.append(dataset[i])

        self.dataset = all_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        X, y = self.dataset[index]

        if self.transform is not None:
            X = self.transform(X)

        return X, y
