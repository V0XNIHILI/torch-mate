import random

import tqdm

from torch.utils.data import Dataset

class Balanced(Dataset):
    def __init__(self, dataset: Dataset, strategy: str):
        """Dataset wrapper to balance the number of samples per class.

        Args:
            dataset (Dataset): The dataset to balance.
            strategy (str): The balancing strategy to use. Options are 'over', 'under' or 'uniform'.
        """

        self.dataset = dataset
        self.strategy = strategy

        self._indices_by_class = {}

        for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Balancing dataset"):
            if label not in self._indices_by_class:
                self._indices_by_class[label] = []
            self._indices_by_class[label].append(i)

        class_counts = {label: len(indices) for label, indices in self._indices_by_class.items()}

        if self.strategy == 'over':
            self._samples_per_class = min(class_counts.values())
        elif self.strategy == 'under':
            self._samples_per_class = max(class_counts.values())
        elif self.strategy == 'uniform':
            self._samples_per_class = sum(class_counts.values()) // len(class_counts)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        self.length = self._samples_per_class * len(class_counts)
        
    def __getitem__(self, index):
        class_index = index // self._samples_per_class
        class_key = list(self._indices_by_class.keys())[class_index]
        samples_to_use = self._indices_by_class[class_key]

        if len(samples_to_use) == self._samples_per_class:
            return self.dataset[samples_to_use[index % self._samples_per_class]]

        return self.dataset[random.choice(samples_to_use)]
        
    def __len__(self):
        return self.length
