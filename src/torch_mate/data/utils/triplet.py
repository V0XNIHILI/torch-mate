import random

from torch.utils.data import Dataset, IterableDataset

from torch_mate.data.utils.get_indices_per_class import get_indices_per_class

class Triplet(IterableDataset):
    def __init__(self, dataset: Dataset):
        super(Triplet).__init__()

        self.dataset = dataset

        self.indices_per_class = get_indices_per_class(self.dataset)

        self.classes = list(self.indices_per_class.keys())

    def generate(self):
        labels = random.sample(self.classes, 2)

        # Anchor, positive, negative
        indices = [random.choice(self.indices_per_class[labels[0]])] + \
            [random.choice(self.indices_per_class[label]) for label in labels]

        return list(map(lambda index: self.dataset[index][0], indices)) + labels
    
    def infinite_generate(self):
        while True:
            yield self.generate()

    def __iter__(self):
        return iter(self.infinite_generate())
