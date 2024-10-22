from typing import Union, Optional, Callable

from torch.utils.data import Dataset

from hasy.hasy_tools import load_data


class HASYv2(Dataset):
    def __init__(self, transform: Union[Callable, None] = None, target_transform: Optional[Callable] = None) -> None:
        """HASYv2 dataset, see https://arxiv.org/pdf/1701.08380 for details. Requires the `hasy` package.
        """

        data = load_data(mode="complete")

        self.X = data['x']
        self.y = data['y']

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        X = self.X[index]
        y = self.y[index][0]

        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            y = self.target_transform(y)

        return X, y

    def __len__(self):
        return len(self.y)
