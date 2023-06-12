from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, BatchSampler, SequentialSampler

from torch_mate.data.samplers import InfiniteClassSampler


def get_indices_per_class(dataset: Dataset) -> List[List[int]]:
    indices_per_class = {}

    for i, (_, label) in enumerate(dataset):
        if label not in indices_per_class:
            indices_per_class[label] = []

        indices_per_class[label].append(i)

    # Sort dict by key and return a list of the values
    return [indices_per_class[key] for key in sorted(indices_per_class.keys())]


class FewShot(IterableDataset):

    def __init__(self,
                 dataset: Dataset,
                 n_way: int,
                 k_shot: int,
                 query_shots: int,
                 incremental: bool = False,
                 always_include_classes: Optional[List[int]] = None,
                 transform: Optional[Callable] = None,
                 per_class_transform: Optional[Callable] = None):
        """Dataset for few shot learning with support for pre-loading all
        samples.

        Args:
            dataset (Dataset): The dataset to use. Labels should be integers.
            n_way (int): Number of classes per batch.
            k_shot (int): Number of samples per class in the support set.
            incremental (bool, optional): Whether to incrementally sample classes. Defaults to False.
            query_shots (int): Number of samples per class in the query set.
            always_include_classes (Optional[List[int]], optional): List of classes to always include in the batch. Defaults to None.
            transform (Optional[Callable], optional): Transform applied to every data sample. Will be reapplied every time a batch is servedd. Defaults to None.
            per_class_transform (Optional[Callable], optional): Transform applied to every data sample. Will only be applied once per class. Defaults to None.
        """

        self.dataset = dataset
        self.indices_per_class = get_indices_per_class(self.dataset)

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots

        self.always_include_classes = always_include_classes

        self.transform = transform
        self.per_class_transform = per_class_transform

        if incremental:
            self.class_sampler = BatchSampler(SequentialSampler(range(len(self.indices_per_class))), batch_size=self.n_way, drop_last=True)
        else:
            self.class_sampler = iter(
                InfiniteClassSampler(len(self.indices_per_class), self.n_way))

    def __iter__(self):
        """Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: The support and query sets in the form of ((X_support, X_query), (y_support, y_query))
        """

        for class_indices in (self.class_sampler):
            X_train_samples = []
            X_test_samples = []

            y_train_samples = []
            y_test_samples = []

            if self.always_include_classes is not None:
                class_indices = list(
                    set(class_indices) - set(self.always_include_classes)
                )[:len(class_indices) - len(self.always_include_classes
                                            )] + self.always_include_classes

            for i, cls in enumerate(class_indices):
                y_train_samples.extend([i] * self.k_shot)
                y_test_samples.extend([i] * self.query_shots)

                within_class_indices = np.random.choice(
                    self.indices_per_class[cls],
                    self.k_shot + self.query_shots,
                    replace=False)

                class_samples = torch.stack(
                    [self.dataset[i][0] for i in within_class_indices])

                if self.per_class_transform is not None:
                    class_samples = self.per_class_transform(class_samples)

                X_train_samples.extend(class_samples[:self.k_shot])
                X_test_samples.extend(class_samples[self.k_shot:])

            X_samples = torch.stack(X_train_samples + X_test_samples)

            if self.transform is not None:
                X_samples = self.transform(X_samples)

            X_train_samples = X_samples[:len(class_indices) * self.k_shot]
            X_test_samples = X_samples[len(class_indices) * self.k_shot:]

            out = ((X_train_samples,
                    X_test_samples), (torch.tensor(y_train_samples),
                                      torch.tensor(y_test_samples)))

            yield out
