from typing import List, Optional, Tuple, Union, Dict

from torch.utils.data import Dataset

from tqdm import tqdm

def get_indices_per_class(dataset: Dataset,
                          support_query_split: Optional[Tuple[int, int]] = None,
                          samples_per_class: Optional[int] = None) -> Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    """Retrieve the indices per class in a dataset.

    Args:
        dataset (Dataset): Dataset to get the indices per class from.
        support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.
        samples_per_class (Optional[int], optional):  Number of samples per class to use. Can be used for large datasets where the classes are ordered (class_0_sample_0, c0s1, c0s2, c1s0, c1s1, c1s2, ...) to avoid iterating over the whole dataset for index per class computation. For the ordering (class_0_sample_0, c1s0, c2s0, c0s1, c1s1, c2s1, ...) use a negative value. Defaults to None.

    Returns:
        Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]: Indices per class, optionally split based on the provided `support_query_split`.
    """

    indices_per_class = {}

    if not samples_per_class:
        for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Getting indices per class"):
            if not isinstance(label, int):
                label = label.item()

            if label not in indices_per_class:
                indices_per_class[label] = []

            indices_per_class[label].append(i)
    else:
        num_classes = len(dataset) // abs(samples_per_class)

        for i in range(num_classes):
            if samples_per_class > 0:
                # Order is: [class0sample0, class0sample1, ..., class0sampleN, class1sample0, class1sample1, ..., class1sampleN, ...]
                _, label = dataset[i*samples_per_class]
                indices_per_class[label] = list(range(i * samples_per_class, (i + 1) * samples_per_class))
            else:
                # Order is: [class0sample0, class1sample0, ..., classNsample0, class0sample1, class1sample1, ..., classNsample1, ...]
                _, label = dataset[i]
                indices_per_class[label] = [j for j in range(i, len(dataset), num_classes)]

    if support_query_split is not None:
        n_support, n_query = support_query_split

        for key in indices_per_class.keys():
            indices_per_class[key] = (indices_per_class[key][:n_support], indices_per_class[key][n_support:n_support+n_query])

    return indices_per_class
