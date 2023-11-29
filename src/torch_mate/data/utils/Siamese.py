from torch.utils.data import Dataset

import random


class Siamese(Dataset):
    def __init__(self, dataset: Dataset, length: int):
        self.dataset = dataset

        self.indices_per_class = get_indices_per_class(self.dataset, self.support_query_split, samples_per_class)

        self.classes = list(self.indices_per_class.keys())

        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        same_label = random.random() < 0.5

        label1 = random.choice(self.classes)

        if same_label:
            indices = random.sample(self.indices_per_class[label1], 2)
        else:
            classes_without_label1 = [label for label in self.classes if label != label1]

            # Make sure that the two labels are different
            label2 = random.choice(classes_without_label1)

            indices = [random.choice(self.indices_per_class[label1]), random.choice(self.indices_per_class[label2])]

        return (self.dataset[indices[0]][0], self.dataset[indices[1]][0]), same_label
