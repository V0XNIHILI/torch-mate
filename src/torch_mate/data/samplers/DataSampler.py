import random
from typing import List, Sized

from torch.utils.data import Sampler

class DataSampler(Sampler):

    def __init__(self, data_source: Sized, index_sampler: Sampler):
        """Creates a sampler that samples data from a data source.

        Args:
            data_source (Sized): Data source to sample from.
            index_sampler (Sampler): Sampler to sample indices from.
        """

        self.data_source = data_source
        self.index_sampler = index_sampler

    def __iter__(self):
        for index in self.index_sampler:
            yield self.data_source[index]