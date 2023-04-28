from typing import Callable, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

from torch_mate.data.utils.Transformed import Transformed


class PreLoaded(Transformed):

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        """Dataset that pre-loads all samples from another dataset. Usage only
        advised if the dataset is small.

        Args:
            dataset (Dataset): The dataset to use. The dataset must be indexable.
            transform (Optional[Callable], optional): Transform to apply to each sample's data. Transforms will not be pre-applied. Defaults to None.
        """

        all_samples = []

        for i in tqdm(range(len(dataset)), desc='Pre-loading samples'):
            all_samples.append(dataset[i])

        super().__init__(all_samples, transform)
