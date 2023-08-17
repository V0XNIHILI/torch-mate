from typing import Callable, List, Optional, Tuple, Union

from itertools import repeat, chain

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, BatchSampler, SequentialSampler

from torch_mate.data.samplers import InfiniteClassSampler


def get_indices_per_class(dataset: Dataset, support_query_split: Optional[Tuple[int, int]] = None) -> List[Union[List[int], Tuple[List[int], List[int]]]]:
    indices_per_class = {}

    for i, (_, label) in enumerate(dataset):
        if not isinstance(label, int):
            label = label.item()

        if label not in indices_per_class:
            indices_per_class[label] = []

        indices_per_class[label].append(i)

    if support_query_split is not None:
        n_support, n_query = support_query_split

        for key in indices_per_class.keys():
            indices_per_class[key] = (indices_per_class[key][:n_support], indices_per_class[key][n_support:n_support+n_query])

    # Sort dict by key and return a list of the values
    return [indices_per_class[key] for key in sorted(indices_per_class.keys())]


class FewShot(IterableDataset):

    def __init__(self,
                 dataset: Dataset,
                 n_way: int,
                 k_shot: int,
                 query_shots: int = -1,
                 first_iter_ways_shots: Optional[Tuple[int, int]] = None,
                 support_query_split: Optional[Tuple[int, int]] = None,
                 incremental: bool = False,
                 cumulative: bool = False,
                 always_include_classes: Optional[List[int]] = None,
                 transform: Optional[Callable] = None,
                 per_class_transform: Optional[Callable] = None):
        """Dataset for few shot learning.

        Example usage:

        >>> # Few-shot learning
        >>> dataset = FewShot(dataset, n_way=5, k_shot=1)
        >>> # FSCIL (https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.pdf)
        >>> dataset = FewShot(dataset, n_way=5, k_shot=5, query_shots=50,
        >>>                   incremental=True, cumulative=True, 
        >>>                   first_iter_ways_shots=(20, 50),
        >>>                   support_query_split=(250, 250)
        >>> )
        >>> # N + M way, K shot learning (https://arxiv.org/abs/1812.10233)
        >>> dataset = FewShot(dataset, n_way=12, k_shot=1, always_include_classes=[0, 1]) # 10 + 2 way 1 shot learning

        Args:
            dataset (Dataset): The dataset to use. Labels should be integers or torch Scalars.
            n_way (int): Number of classes per batch.
            k_shot (int, optional): Number of samples per class in the support set.
            query_shots (Optional[int]): Number of samples per class in the query set. If not set, query_shots is set to k_shot. Defaults to -1.
            first_iter_ways_shots (Optional[Tuple[int, int]], optional): Number of classes and samples per class for the first iteration. Defaults to None.
            support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.
            incremental (bool, optional): Whether to incrementally sample classes. Defaults to False.
            cumulative (bool, optional): Whether to increase the query set size with each iteration. This flag will only work when incremental is set to True. Defaults to False.
            always_include_classes (Optional[List[int]], optional): List of classes to always include in the batch, both in the support and query set. Defaults to None.
            transform (Optional[Callable], optional): Transform applied to every data sample. Will be reapplied every time a batch is served. Defaults to None.
            per_class_transform (Optional[Callable], optional): Transform applied to every data sample. Will only be applied once per class. Defaults to None.
        """

        if cumulative and not incremental:
            raise ValueError(
                "cumulative can only be set to True when incremental is True."
            )

        if first_iter_ways_shots and not incremental:
            raise ValueError(
                "first_iter_ways_shots can only be set when incremental is True."
            )

        if always_include_classes is not None:
            if len(always_include_classes) > n_way:
                raise ValueError("always_include_classes cannot have more elements than n_way.")

            if len(set(always_include_classes)) != len(always_include_classes):
                raise ValueError("always_include_classes cannot have duplicate elements.")

            if cumulative:
                raise ValueError("always_include_classes cannot be used when cumulative is True.")

        self.dataset = dataset
        self.support_query_split = support_query_split
        self.indices_per_class = get_indices_per_class(self.dataset, self.support_query_split)

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot

        self.always_include_classes = always_include_classes

        self.transform = transform
        self.per_class_transform = per_class_transform

        self.first_iter_ways_shots = first_iter_ways_shots

        self.incremental = incremental
        self.cumulative = cumulative

        total_classes = len(self.indices_per_class)

        if self.incremental:
            first_iter_classes = 0
            first_iter = []

            if self.first_iter_ways_shots:
                first_iter_classes = self.first_iter_ways_shots[0]
                # Use negative values as first_iter_classes is later added to all indices
                # when self.first_iter_ways_shots is defined. This is to make sure that the
                # iterations after the first iteration do not contain the same classes as
                # the first iteration. The SequentialSampler used for all but the first
                # iteration will start at 0, so we need to make sure that the first iteration
                # is negative to avoid overlap.
                first_iter = [list(range(-first_iter_classes, 0))]

            end = total_classes

            self.class_sampler = chain(first_iter, BatchSampler(SequentialSampler(range(total_classes - first_iter_classes)), batch_size=self.n_way, drop_last=True))
        else:
            self.class_sampler = InfiniteClassSampler(total_classes, self.n_way)

    def __iter__(self):
        """Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: The support and query sets in the form of ((X_support, X_query), (y_support, y_query))
        """

        cumulative_classes = []

        if self.incremental:
            class_mapping = np.random.permutation(total_classes)
        else:
            class_mapping = np.arange(total_classes)

        # Change the way and shots for the first iteration
        if self.first_iter_ways_shots:
            n_way, k_shot = self.first_iter_ways_shots
            query_shots = k_shot
        else:
            n_way = self.n_way
            k_shot = self.k_shot
            query_shots = self.query_shots

        for new_class_indices in self.class_sampler:
            X_train_samples = []
            X_test_samples = []

            y_train_samples = []
            y_test_samples = []

            new_class_indices = class_mapping[new_class_indices]

            # Add the first iteration way count to the class indexes to make the negative first iteration indices positive
            new_class_indices = list(np.array(new_class_indices) + self.first_iter_ways_shots[0]) if self.first_iter_ways_shots else new_class_indices

            class_indices = new_class_indices + cumulative_classes

            if self.cumulative:
                cumulative_classes.extend(new_class_indices)

            if self.always_include_classes is not None:
                # This line also makes sure that the always include classes are always
                # in the same position in the batch
                class_indices = list(
                    set(class_indices) - set(self.always_include_classes)
                )[:len(class_indices) - len(self.always_include_classes
                                            )] + self.always_include_classes

            for i, cls in enumerate(class_indices):
                if self.support_query_split:
                    within_class_indices = np.concatenate([np.random.choice(self.indices_per_class[cls][j], shot, replace=False) for j, shot in [(0, k_shot), (1, query_shots)]])
                else:
                    within_class_indices = np.random.choice(
                        self.indices_per_class[cls],
                        k_shot + query_shots,
                        replace=False)

                class_samples = torch.stack(
                    [self.dataset[j][0] for j in within_class_indices])

                if self.per_class_transform is not None:
                    class_samples = self.per_class_transform(class_samples)

                class_index = i if not self.incremental else cls

                # Only in the case of cumulative we need to make sure that we only
                # include the new classes in the support set and not the previous classes
                if i < n_way:
                    y_train_samples.extend([class_index] * k_shot)
                    X_train_samples.extend(class_samples[:k_shot])

                y_test_samples.extend([class_index] * query_shots)
                X_test_samples.extend(class_samples[k_shot:])

            X_samples = torch.stack(X_train_samples + X_test_samples)

            if self.transform is not None:
                X_samples = self.transform(X_samples)

            num_train_samples = len(class_indices) * k_shot
            X_train_samples = X_samples[:num_train_samples]
            X_test_samples = X_samples[num_train_samples:]

            out = ((X_train_samples,
                    X_test_samples), (torch.tensor(y_train_samples),
                                      torch.tensor(y_test_samples)))

            # Reset the way and shots for the next iteration if the first iteration was different
            if self.first_iter_ways_shots:
                n_way = self.n_way
                k_shot = self.k_shot
                query_shots = self.query_shots

            yield out
