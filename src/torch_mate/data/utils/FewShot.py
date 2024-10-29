from typing import Callable, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset

from torch_mate.data.samplers import InfiniteClassSampler
from torch_mate.data.utils.get_indices_per_class import get_indices_per_class

class FewShot(IterableDataset):

    def __init__(self,
                 dataset: Dataset,
                 n_way: int,
                 k_shot: int,
                 query_shots: int = -1,
                 query_ways: int = -1,
                 support_query_split: Optional[Tuple[int, int]] = None,
                 samples_per_class: Optional[int] = None,
                 always_include_classes: Optional[List[int]] = None,
                 transform: Optional[Callable] = None,
                 per_class_transform: Optional[Callable] = None,
                 keep_original_labels: bool = False):
        """Dataset for few shot learning.

        Example usage:

        >>> # Few-shot learning
        >>> dataset = FewShot(dataset, n_way=5, k_shot=1)
        >>> # N + M way, K shot learning (https://arxiv.org/abs/1812.10233)
        >>> dataset = FewShot(dataset, n_way=12, k_shot=1, always_include_classes=[0, 1]) # 10 + 2 way 1 shot learning

        Args:
            dataset (Dataset): The dataset to use. Labels should be integers or torch Scalars.
            n_way (int): Number of classes in the query and query set.
            k_shot (int, optional): Number of samples per class in the support set.
            query_shots (Optional[int]): Number of samples per class in the query set. If not set, query_shots is set to k_shot. Defaults to -1.
            support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.
            samples_per_class (Optional[int], optional): Number of samples per class to use. Can be used for large datasets where the classes are ordered (class_0_sample_0, c0s1, c0s2, c1s0, c1s1, c1s2, ...) to avoid iterating over the whole dataset for index per class computation. Defaults to None.
            always_include_classes (Optional[List[int]], optional): List of classes to always include in both in the support and query set. Defaults to None.
            transform (Optional[Callable], optional): Transform applied to every data sample. Will be reapplied every time a batch is served. Defaults to None.
            per_class_transform (Optional[Callable], optional): Transform applied to every data sample. Will be applied to every class separately. Defaults to None.
        """

        if always_include_classes is not None:
            if len(always_include_classes) > n_way:
                raise ValueError("always_include_classes cannot have more elements than n_way.")

            if len(set(always_include_classes)) != len(always_include_classes):
                raise ValueError("always_include_classes cannot have duplicate elements.")

        self.dataset = dataset
        self.support_query_split = support_query_split
        self.indices_per_class = get_indices_per_class(self.dataset, self.support_query_split, samples_per_class)

        if len(self.indices_per_class) < n_way:
            raise ValueError("The dataset does not have enough classes to create a batch of size n_way.")

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot
        self.query_ways = query_ways if query_ways != -1 else n_way

        self.always_include_classes = always_include_classes

        self.transform = transform
        self.per_class_transform = per_class_transform

        self.total_classes = len(self.indices_per_class)
        
        self.class_sampler = InfiniteClassSampler(list(self.indices_per_class.keys()), self.n_way)

        self.keep_original_labels = keep_original_labels

    def infinite_generate(self):
        """Get a batch of samples for a k-shot n-way task.

        Yields:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: The support and query sets in the form of ((X_support, X_query), (y_support, y_query))
        """

        for new_class_indices in self.class_sampler:
            X_train_samples = []
            X_test_samples = []

            y_train_samples = []
            y_test_samples = []

            class_indices = new_class_indices

            if self.always_include_classes is not None:
                # This line also makes sure that the always include classes are always
                # in the same position in the batch
                class_indices = list(
                    set(class_indices) - set(self.always_include_classes)
                )[:len(class_indices) - len(self.always_include_classes
                                            )] + self.always_include_classes

            # get self.query_ways ints between 0 (inc.) and len(class_indices) (exc.)
            test_class_indices = np.random.choice(len(class_indices), self.query_ways, replace=False)

            for i, class_index in enumerate(class_indices):
                if self.support_query_split:
                    within_class_indices = np.concatenate([np.random.choice(self.indices_per_class[class_index][j], shot, replace=False) for j, shot in [(0, self.k_shot), (1, self.query_shots)]])
                else:
                    within_class_indices = np.random.choice(
                        self.indices_per_class[class_index],
                        self.k_shot + self.query_shots,
                        replace=False)

                Xs, ys = zip(*[self.dataset[j] for j in within_class_indices])
                original_label = ys[0]
                new_label = original_label if self.keep_original_labels else i

                class_samples = torch.stack(Xs)

                if self.per_class_transform is not None:
                    class_samples = self.per_class_transform(class_samples)

                y_train_samples.extend([new_label] * self.k_shot)

                X_train_samples.extend(class_samples[:self.k_shot])

                if i in test_class_indices:
                    y_test_samples.extend([new_label] * self.query_shots)

                    X_test_samples.extend(class_samples[self.k_shot:])

            X_samples = torch.stack(X_train_samples + X_test_samples)

            if self.transform is not None:
                X_samples = self.transform(X_samples)

            num_train_samples = len(y_train_samples)
            X_train_samples = X_samples[:num_train_samples]
            X_test_samples = X_samples[num_train_samples:]

            out = ((X_train_samples,
                    torch.tensor(y_train_samples)), (X_test_samples,
                                    torch.tensor(y_test_samples)))

            yield out

    def __iter__(self):
        return iter(self.infinite_generate())
