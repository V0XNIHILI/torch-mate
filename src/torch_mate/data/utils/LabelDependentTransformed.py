from typing import Callable, Union

from torch.utils.data import Dataset


class LabelDependentTransformed(Dataset):

    def __init__(self,
                 dataset: Dataset,
                 dependent_transform: Callable,
                 transform: Union[Callable, None] = None):
        """Dataset where a transform can be applied that takes in both the x-value (i.e. an image) and the label.

        Args:
            dataset (Dataset): Dataset to use
            dependent_transform (Callable): Transform that takes in both the x-value and the label and outputs a transformed x-value.
            transform (Union[Callable, None], optional): Transform to apply after the label-dependent transform. Defaults to None.
        """

        self.dataset = dataset
        self.dependent_transform = dependent_transform
        self.transform = transform

    def __getitem__(self, index: int):
        x, y = self.dataset[index]

        x = self.dependent_transform(x, y)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)
