from typing import Dict

import torch
from torchvision.datasets import MNIST

from torch_mate.lightning import ConfigurableLightningDataModule

from torch.utils.data import random_split
from torch_mate.data.utils import Transformed


class MNISTData(ConfigurableLightningDataModule):
    def __init__(self, cfg: Dict, root: str, download: bool = True):
        super().__init__(cfg)

        self.root = root
        self.download = download

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'validate':
            mnist_full = MNIST(self.root, train=True, download=self.download)
            
            train_percentage = self.hparams.task["train"]["percentage"]

            mnist_train, mnist_val = random_split(
                mnist_full, [train_percentage, 1-train_percentage], generator=torch.Generator().manual_seed(self.hparams.seed)
            )

            self.mnist_train = Transformed(mnist_train, self.train_transforms, self.train_target_transforms)
            self.mnist_val = Transformed(mnist_val, self.val_transforms, self.val_target_transforms)
        elif stage == 'test':
            self.mnist_test = MNIST(self.root, train=False, transform=self.test_transforms, target_transform=self.test_target_transforms, download=self.download)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def get_dataset(self, phase: str):
        if phase == 'train':
            return self.mnist_train
        elif phase == 'val':
            return self.mnist_val
        elif phase == 'test':
            return self.mnist_test
        
        raise ValueError(f"Unsupported phase: {phase}")
