import torch

import numpy as np

class PermuteSequence:
    def __init__(self, length: int):
        self.length = length

        self._permutation = torch.Tensor(np.random.permutation(length).astype(np.float64)).long()

    def __call__(self, x: torch.Tensor):
        return x[..., self._permutation]
