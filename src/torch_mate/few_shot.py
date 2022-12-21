import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def sample_classes(dataset: Dataset, k: int, class_size: int):
    total_classes = int(len(dataset) / class_size)

    # Sample k classes from the dataset
    classes = np.random.choice(total_classes, k, replace=False)

    return classes


def sample_n_shot_k_way(dataset: Dataset, n: int, k: int, class_size: int):
    """Given a dataset, sample n*k examples from k classes.

    Args:
        dataset (Dataset): The dataset to sample from.
        n (int): Number of examples per class.
        k (int): Number of classes to sample.
        class_size (int): Total number of examples per class in the dataset.

    Returns:
        tuple[torch.Tensor, Subset]: A tuple with the indices of the k sampled classes and a subset of the dataset containing n*k examples from k classes.
    """

    classes = sample_classes(dataset, k, class_size)

    # Create an array of length k*n, where each n elements starts with a different class index and counts down up class index + n
    class_indices = np.array(
        [np.arange(c, c + n, 1) for c in classes * class_size]).flatten()

    label_mapping = create_label_mapping(classes)

    return label_mapping, Subset(dataset, class_indices)


def split_train_test_n_shot_k_way(dataset: Dataset, n: int, k: int,
                                  test_shots: int, class_size: int):
    """Split a few-shot learning dataset into a training and test set.

    Args:
        dataset (Dataset): The dataset to split.
        n (int): Number of examples per class.
        k (int): Number of classes to sample.
        class_size (int): Total number of examples per class in the dataset.
        test_shots (int): Number of examples per class to use for testing.

    Returns:
        tuple[torch.Tensor, Subset, Subset]: A tuple containing label mapping, the training and test sets.
    """

    if test_shots >= n:
        raise ValueError("test_shots must be less than n")

    classes = sample_classes(dataset, k, class_size)
    actual_classes_indices = classes * class_size

    train_class_indices = np.array(
        [np.arange(c + n - test_shots, c + n, 1) for c in actual_classes_indices]).flatten()
    test_class_indices = np.array(
        [np.arange(c, c + n - test_shots, 1) for c in actual_classes_indices]).flatten()

    label_mapping = create_label_mapping(classes)

    return label_mapping, Subset(dataset, train_class_indices), Subset(dataset, test_class_indices)


def create_label_mapping(unique_labels: np.ndarray):
    """Create a mapping from the original labels to new labels.

    Args:
        unique_labels (np.ndarray): An array of unique labels.

    Returns:
        torch.Tensor: A tensor of shape (k, 2) where the first column contains the original labels and the second column contains the new labels.
    """

    # Create a mapping from the original labels to new labels
    mapping = np.array([unique_labels, np.arange(len(unique_labels))]).T

    return torch.from_numpy(mapping)


def remap_labels(labels: torch.Tensor, mapping: torch.Tensor):
    """Remap labels in a tensor according to a mapping. Remapping code taken
    from: https://stackoverflow.com/questions/62185188/recommended-way-to-
    replace-several-values-in-a-tensor-at-once.

    Args:
        labels (torch.Tensor): The labels to remap.
        mapping (torch.Tensor): A tensor of shape (k, 2) where the first column contains the original labels and the second column contains the new labels.

    Returns:
        torch.Tensor: The remapped labels.
    """

    # Unreadable vectorized version of the code below:
    # for m in mapping:
    #     labels[labels == m[0]] = m[1]
    mask = labels == mapping[:, :1]
    remapped_labels = (1 - mask.sum(dim=0)) * labels + (
        mask * mapping[:, 1:]).sum(dim=0)

    return remapped_labels
