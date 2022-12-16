import numpy as np
from torch.utils.data import Dataset, Subset


def sample_classes(dataset: Dataset, k: int, class_size: int):
    total_classes = int(len(dataset) / class_size)

    # Sample k classes from the dataset
    classes = np.random.choice(total_classes, k, replace=False) * class_size

    return classes


def sample_n_shots_k_way(dataset: Dataset, n: int, k: int, class_size: int):
    """Given a dataset, sample n*k examples from k classes.

    Args:
        dataset (Dataset): The dataset to sample from.
        n (int): Number of examples per class.
        k (int): Number of classes to sample.
        class_size (int): Total number of examples per class in the dataset.

    Returns:
        Subset: A subset of the dataset containing n*k examples from k classes.
    """

    classes = sample_classes(dataset, k, class_size)

    # Create an array of length k*n, where each n elements starts with a different class index and counts down up class index + n
    class_indices = np.array([np.arange(c, c + n, 1)
                              for c in classes]).flatten()

    return Subset(dataset, class_indices)


def split_train_test_n_shots_k_way(dataset: Dataset, n: int, k: int,
                                   test_shots: int, class_size: int):
    """Split a few-shot learning dataset into a training and test set.

    Args:
        dataset (Dataset): The dataset to split.
        n (int): Number of examples per class.
        k (int): Number of classes to sample.
        class_size (int): Total number of examples per class in the dataset.
        test_shots (int): Number of examples per class to use for testing.

    Returns:
        tuple[Subset, Subset]: A tuple containing the training and test sets.
    """

    if test_shots >= n:
        raise ValueError("test_shots must be less than n")

    classes = sample_classes(dataset, k, class_size)

    train_class_indices = np.array(
        [np.arange(c + n - test_shots, c + n, 1) for c in classes]).flatten()
    test_class_indices = np.array(
        [np.arange(c, c + n - test_shots, 1) for c in classes]).flatten()

    return Subset(dataset,
                  train_class_indices), Subset(dataset, test_class_indices)
