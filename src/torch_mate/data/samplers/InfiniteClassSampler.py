import random
from typing import List

from torch.utils.data import Sampler


class InfiniteClassSampler(Sampler[List[int]]):

    def __init__(self, class_indices: List[int], n_way: int):
        """Creates a sampler that samples classes infinitely (or effectively
        samples sys.maxsize classes)

        Args:
            class_indices (List[int]): All classes the dataset has.
            n_way (int): Number of classes per batch.
        """

        self.classes_list = class_indices
        self.n_way = n_way

    def __iter__(self):
        while True:
            yield random.sample(self.classes_list, self.n_way)
