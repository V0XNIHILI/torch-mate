import random
from typing import Sequence

import torchvision.transforms.functional as TF


class DiscreteRandomRotation:

    def __init__(self, angles: Sequence[int], p: float):
        """Randomly rotate an image by one of the given angles. Taken from:

        https://github.com/pytorch/vision/issues/566#issuecomment-535854734.

        Args:
            angles (Sequence[int]): Angles to rotate by.
            p (float): Probability of rotation.
        """

        self.angles = angles
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)

            return TF.rotate(x, angle)

        return x
