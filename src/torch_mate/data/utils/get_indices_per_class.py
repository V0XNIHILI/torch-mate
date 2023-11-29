from typing import List, Optional, Tuple, Union, Dict

from torch.utils.data import Dataset

from tqdm import tqdm

def get_indices_per_class(dataset: Dataset,
                          support_query_split: Optional[Tuple[int, int]] = None,
                          samples_per_class: Optional[int] = None) -> Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    indices_per_class = {}

    if not samples_per_class:
        for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Getting indices per class"):
            if not isinstance(label, int):
                label = label.item()

            if label not in indices_per_class:
                indices_per_class[label] = []

            indices_per_class[label].append(i)
    else:
        for i in range(len(dataset) // samples_per_class):
            _, label = dataset[i*samples_per_class]
            indices_per_class[label] = list(range(i * samples_per_class, (i + 1) * samples_per_class))

    if support_query_split is not None:
        n_support, n_query = support_query_split

        for key in indices_per_class.keys():
            indices_per_class[key] = (indices_per_class[key][:n_support], indices_per_class[key][n_support:n_support+n_query])

    return indices_per_class
