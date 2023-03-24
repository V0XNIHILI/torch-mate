import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


def plot_class_distribution(dataset: Dataset,
                            title: str,
                            shift: int,
                            width=0.2):
    labels = np.array([label for _, _, label, _, _ in dataset])

    unique_label_count = len(np.unique(labels))

    # Calculate percentage of each class
    class_dist = np.array(
        [np.sum(labels == i) for i in range(unique_label_count)]) / len(labels)

    plt.bar(np.arange(unique_label_count) + shift * width,
            class_dist,
            width=width,
            label=title)


def plot_class_distributions(train_set: Dataset, val_set: Dataset,
                             test_set: Dataset):
    """Plot class distributions of the given dataset splits.

    Args:
        train_set (Dataset): Training set.
        val_set (Dataset): Validation set.
        test_set (Dataset): Test set.
    """

    for dataset, title, shift in zip([train_set, val_set, test_set],
                                     ['train', 'val', 'test'], [0, 1, 2]):
        plot_class_distribution(dataset, title, shift)

    plt.legend()
    plt.title('Class distribution')
    plt.xlabel('Class index')
    plt.ylabel('Percentage')
    plt.show()
