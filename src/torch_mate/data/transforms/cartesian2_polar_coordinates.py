import math

import torch


class Cartesian2PolarCoordinates:

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a batch of 2D points from Cartesian to Polar coordinates.

        Args:
            x (torch.Tensor): Batch of 2D points, shape = (2, batch size)

        Returns:
            torch.Tensor: Batch of 2D points in Polar coordinates, shape = (2, batch size)
        """

        x_x = x[0, :]
        x_y = x[1, :]

        r = torch.sqrt(x_x**2 + x_y**2)
        theta = torch.atan2(x_y, x_x) / (2 * math.pi)

        return torch.stack([r, theta])
