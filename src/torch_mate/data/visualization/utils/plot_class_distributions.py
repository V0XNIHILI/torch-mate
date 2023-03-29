import matplotlib.pyplot as plt

from torch.utils.data import Dataset

import numpy as np

def plot_class_distribution(dataset: Dataset, title: str, shift: int, width=0.2):
    labels = np.array([label for _, label in dataset])

    unique_label_count = len(np.unique(labels))
    
    # Calculate percentage of each class
    class_dist = np.array([np.sum(labels == i) for i in range(unique_label_count)]) / len(labels)

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
