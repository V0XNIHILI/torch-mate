from torch.utils.data import IterableDataset, Dataset

import random

from torch_mate.data.utils.get_indices_per_class import get_indices_per_class


class Siamese(IterableDataset):
    def __init__(self, dataset: Dataset):
        super(Siamese).__init__()

        self.dataset = dataset

        self.indices_per_class = get_indices_per_class(self.dataset)

        self.classes = list(self.indices_per_class.keys())

    def generate(self):
        while True:
            same_label = random.random() < 0.5

            label1 = random.choice(self.classes)

            if same_label:
                indices = random.sample(self.indices_per_class[label1], 2)
            else:
                classes_without_label1 = [label for label in self.classes if label != label1]

                # Make sure that the two labels are different
                label2 = random.choice(classes_without_label1)

                indices = [random.choice(self.indices_per_class[label1]), random.choice(self.indices_per_class[label2])]

            yield (self.dataset[indices[0]][0], self.dataset[indices[1]][0]), same_label

    def __iter__(self):
        return iter(self.generate())
