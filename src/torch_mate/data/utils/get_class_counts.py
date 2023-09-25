from typing import Dict, Any

import torch

def get_class_counts(dataset: torch.utils.data.Dataset) -> Dict[Any, int]:
    """Get the number of samples for each class in a dataset.

    Args:
        dataset (torch.utils.data.Dataset): A dataset.
    
    Returns:
        dict: A dictionary of class counts.
    """

    class_counts = {}

    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
    
        class_counts[label] += 1

    return class_counts
