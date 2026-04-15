from .Transformed import apply_if_not_none

from typing import Callable, Optional, Iterator
from torch.utils.data import IterableDataset


class TransformedIterable(IterableDataset):

    def __init__(self, dataset: IterableDataset, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        """Add a transform to a dataset that does not support transforms.

        Args:
            dataset (Dataset): The dataset to use.
            transform (Optional[Callable], optional): Transform to apply to each sample's data. Defaults to None.
            target_transform (Optional[Callable], optional): Transform to apply to each sample's target. Defaults to None.
        """

        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

    def __iter__(self) -> Iterator:
        for data in self.dataset:
            has_target_transform = self.target_transform is not None

            if isinstance(data, (tuple, list)):
                if len(data) == 2:
                    X, y = data
                    X = apply_if_not_none(self.transform, X)
                    y = apply_if_not_none(self.target_transform, y)
                    yield X, y
                elif len(data) > 2 and has_target_transform:
                    Xs = data[:-1]
                    y = data[-1]
                    if self.transform is not None:
                        Xs = [self.transform(x) for x in Xs]
                    y = self.target_transform(y)
                    yield Xs + [y]
                else:
                    raise ValueError("Target transform provided but no target data found.")
            else:
                yield apply_if_not_none(self.transform, data)