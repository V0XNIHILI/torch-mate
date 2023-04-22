import random
from typing import List

from torch.utils.data import Sampler


class InfiniteClassSampler(Sampler[List[int]]):

    def __init__(self, total_classes: int, n_way: int):
        """Creates a sampler that samples classes infinitely (or effectively
        samples sys.maxsize classes)

        Args:
            total_classes (int): Total classes the dataset has.
            n_way (int): Number of classes per batch.
        """

        self.total_classes = total_classes
        self.n_way = n_way

        self.classes_list = list(range(total_classes))

    def __iter__(self):
        while True:
            yield random.sample(self.classes_list, self.n_way)
