import matplotlib.pyplot as plt

from torch.utils.data import Dataset

import numpy as np

from torch_mate.data.utils.get_class_counts import get_class_counts

def plot_class_distribution(dataset: Dataset, title: str, shift: int, width=0.2):
    class_counts = get_class_counts(dataset)

    unique_labels = sorted(class_counts.keys())
    unique_label_count = len(unique_labels)
    class_dist = [class_counts[label] for label in unique_labels]

    plt.bar(np.arange(unique_label_count)+shift*width, class_dist, width=width, label=title)

    return unique_label_count

def plot_class_distributions(train_set: Dataset, val_set: Dataset, test_set: Dataset):
    label_count = 0

    for dataset, title, shift in zip([train_set, val_set, test_set], ['train', 'val', 'test'], [-1, 0, 1]):
        new_label_count = plot_class_distribution(dataset, title, shift)

        if new_label_count != label_count:
            if  label_count != 0:
                raise ValueError('Number of classes in train, val and test sets do not match')
            else:
                label_count = new_label_count

    plt.legend()
    plt.title('Class distribution')
    plt.xlabel('Class index')
    plt.ylabel('Percentage')
    plt.xticks(np.arange(label_count))
    plt.show()
