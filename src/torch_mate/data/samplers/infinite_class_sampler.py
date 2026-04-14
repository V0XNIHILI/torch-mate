import random
from typing import List, Optional

from torch.utils.data import Sampler


class InfiniteClassSampler(Sampler[List[int]]):

    def __init__(self, class_indices: List[int], n_way: int, seed: Optional[int] = None):
        """Creates a sampler that samples classes infinitely (or effectively
        samples sys.maxsize classes)

        Args:
            class_indices (List[int]): All classes the dataset has.
            n_way (int): Number of classes per batch.
            seed (Optional[int]): If set, uses a local RNG with this seed for deterministic sampling. Defaults to None.
        """

        self.classes_list = class_indices
        self.n_way = n_way
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        while True:
            yield rng.sample(self.classes_list, self.n_way)
