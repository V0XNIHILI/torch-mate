from typing import Union, Optional, Callable

from torch.utils.data import Dataset

from hasy.hasy_tools import load_data

class HASYv2(Dataset):
    def __init__(self, transform: Union[Callable, None] = None) -> None:
        """HASYv2 dataset, see https://arxiv.org/pdf/1701.08380 for details.
        """

        data = load_data(mode="complete")

        self.X = data['x']
        self.y = data['y']

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)
