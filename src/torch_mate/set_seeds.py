import random

import numpy as np
import torch


def set_seeds(seed=2147483647, device: torch.device = 'cpu'):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return torch.Generator(device).manual_seed(seed)
