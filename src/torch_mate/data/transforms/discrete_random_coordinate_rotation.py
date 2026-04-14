import math
import random
from typing import Sequence

import torch


class DiscreteRandomCoordinateRotation:

    def __init__(self, angles: Sequence[int]):
        """Randomly rotate a batch of coordinates of shape (2, batch size) with one of the gives angles.

        Inspiration taken from: https://discuss.pytorch.org/t/rotation-matrix/128260.

        Args:
            angles (Sequence[int]): Possible angles to rotate the inputs with
        """

        self.angles = angles

        self.rotation_matrices = {}

        for angle in angles:
            phi = angle * math.pi / 180

            s = math.sin(phi)
            c = math.cos(phi)

            self.rotation_matrices[angle] = torch.tensor([[c, -s], [s, c]]).t()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly rotates a batch of 2D points with the same random angle.

        Args:
            x (torch.Tensor): Batch of 2D points, shape = (2, batch size)

        Returns:
            torch.Tensor: Batch of rotated 2D points, shape = (2, batch size)
        """

        angle = random.choice(self.angles)

        return self.rotation_matrices[angle] @ x
