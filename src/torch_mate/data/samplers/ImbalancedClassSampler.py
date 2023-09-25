import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class ImbalancedClassSampler(WeightedRandomSampler):

    def __init__(self,
                 dataset: Dataset,
                 length: int,
                 replacement: bool = True):
        """Sampler that samples all classes in a balanced fashion when the number of samples per classes are not evenly distributed.

        Args:
            dataset (Dataset): Dataset to sample balancedly.
            length (int): Length of newly 'created' dataset
            replacement (bool, optional): Whether to sample classes with replacement. Defaults to True.
        """

        labels = torch.tensor([label for _, label in dataset])

        class_count = torch.bincount(labels.squeeze())
        class_weighting = 1. / class_count
        sample_weights = class_weighting[labels]

        super().__init__(sample_weights, length, replacement=replacement)
