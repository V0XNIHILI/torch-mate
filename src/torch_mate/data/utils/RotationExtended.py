from typing import Sequence

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class RotationExtended(Dataset):

    def __init__(self,
                 dataset: Dataset,
                 samples_per_class: int,
                 angles: Sequence[int] = [0, 90, 180, 270]):
        """Dataset that extends a dataset by rotating each sample by a given
        angle.

        Args:
            dataset (Dataset): Dataset to extend.
            samples_per_class (int): Number of samples per class in the dataset, used to calculate the label offset.
            angles (Sequence[int], optional): Rotation angles to extend the dataset with. Defaults to [0, 90, 180, 270].
        """

        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.angles = angles

    def __len__(self):
        return len(self.dataset) * len(self.angles)

    def __getitem__(self, index: int):
        image, label = self.dataset[index % len(self.dataset)]
        rotation_index = index // len(self.dataset)
        angle = self.angles[rotation_index]
        image = TF.rotate(image, angle)

        return image, label + rotation_index * (len(self.dataset) //
                                                self.samples_per_class)
