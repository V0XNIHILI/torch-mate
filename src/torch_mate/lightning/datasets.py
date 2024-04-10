from typing import Dict

import torch
from torchvision.datasets import MNIST

from torch_mate.lightning import ConfigurableLightningDataModule

from torch.utils.data import random_split
from torch_mate.data.utils import Transformed


class MNISTData(ConfigurableLightningDataModule):
    def __init__(self, cfg: Dict, root: str, download: bool = True):
        """MNIST dataset.

        Configuration:
        ```yaml
        cfg.dataset.cfg:
            # Percentage of the training set to use for validation (float). Defaults to 500/60000.
            val_percentage (float): 0.1
        ```

        Args:
            cfg (Dict): Configuration dictionary.
            root (str): Root directory for the dataset.
            download (bool, optional): Whether to download the dataset or not. Defaults to True.
        """

        super().__init__(cfg)

        self.root = root
        self.download = download

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'validate':
            mnist_full = MNIST(self.root, train=True, download=self.download)
            
            val_percentage = self.hparams.dataset.get("cfg", {}).get("val_percentage", 500/60000)

            mnist_train, mnist_val = random_split(
                mnist_full, [1 - val_percentage, val_percentage], generator=torch.Generator().manual_seed(self.hparams.seed)
            )

            self.mnist_train = mnist_train
            self.mnist_val = mnist_val
        elif stage == 'test':
            self.mnist_test = MNIST(self.root, train=False, download=self.download)
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
