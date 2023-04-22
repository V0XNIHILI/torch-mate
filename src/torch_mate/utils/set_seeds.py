import os
import random

import numpy as np
import torch


def set_seeds(seed=2147483647,
              device: torch.device = torch.device('cpu'),
              fully_deterministic=False):
    """Set seeds for reproducibility. Sets the Python, NumPy, and PyTorch
    seeds.

    Args:
        seed (int, optional): Seed to use. Defaults to 2147483647.
        device (torch.device, optional): Device to instantiate generator for. Defaults to 'cpu'.
        fully_deterministic (bool, optional): Whether or not to use fully deterministic algorithms. If True, it will lead to lower performance. Defaults to False.

    Returns:
        torch.Generator: Seeded generator for the given device.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # https://pytorch.org/docs/stable/notes/randomness.html
    if device.type == 'cuda' and fully_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return torch.Generator(device).manual_seed(seed)
