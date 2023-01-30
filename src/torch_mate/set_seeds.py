import random

import numpy as np
import torch


def set_seeds(device: torch.device = 'cpu', seed=2147483647):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return torch.Generator(device).manual_seed(seed)
