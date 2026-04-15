from typing import Callable, Optional

from torch.utils.data import Dataset


def apply_if_not_none(transform: Optional[Callable], data):
    if transform is not None:
        return transform(data)

    return data


class Transformed(Dataset):

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        """Add a transform to a dataset that does not support transforms.

        Args:
            dataset (Dataset): The dataset to use.
            transform (Optional[Callable], optional): Transform to apply to each sample's data. Defaults to None.
            target_transform (Optional[Callable], optional): Transform to apply to each sample's target. Defaults to None.
        """

        self.transform = transform
        self.target_transform = target_transform

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]

        has_target_transform = self.target_transform is not None

        if type(data) is tuple or type(data) is list:
            if len(data) == 2:
                X, y = data

                X = apply_if_not_none(self.transform, X)
                y = apply_if_not_none(self.target_transform, y)

                return X, y
            if len(data) > 2 and has_target_transform:
                Xs = data[:-1]
                y = data[-1]

                if self.transform is not None:
                    for i in range(len(Xs)):
                        Xs[i] = self.transform(Xs[i])

                y = self.target_transform(y)

                return Xs + [y]
            
        if has_target_transform:
            raise ValueError("Target transform provided but no target data found.")

        return apply_if_not_none(self.transform, data)
