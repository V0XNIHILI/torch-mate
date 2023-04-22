import random

import torch


class TimeShift:

    def __init__(self, min_shift: int, max_shift: int):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Shifts the data along the time dimension. The data is padded with
        zeros and remains the same length.

        Args:
            x (torch.Tensor): Data to be shifted, shape = (batch dim (optional), #channels, sequence length)

        Returns:
            torch.Tensor: Shifted data
        """

        ignored_dimensions = [0] * (len(x.shape) - 1)

        # A positive shift means that the data is shifted to the right
        shift = random.randint(self.min_shift, self.max_shift)

        x_rolled = torch.roll(x, (*ignored_dimensions, shift),
                              dims=(*ignored_dimensions, 1))

        return x_rolled
