from typing import Dict

import torch.nn as nn
import torch.optim as optim

import torchvision

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torch_mate.utils import get_class, calc_accuracy


class ConfigurableLightningModule(L.LightningModule):
    def __init__(self, cfg: Dict):
        """Lightweight wrapper around PyTorch LightningModule that adds support for a configuration dictionary.

        Adds the following three attributes:
        
        - self.model: the created model
        - self.criterion: the created criterion
        - self.generic_step(self, batch, batch_idx, phase): a generic step function that is shared across all steps (train, val, test, predict)

        Based on these, the following methods are automatically implemented:

        - self.forward(self, x): calls self.model(x)
        - self.training_step(self, batch, batch_idx): calls self.generic_step(batch, batch_idx, "train")
        - self.validation_step(self, batch, batch_idx): calls self.generic_step(batch, batch_idx, "val")
        - self.test_step(self, batch, batch_idx): calls self.generic_step(batch, batch_idx, "test")
        - self.predict_step(self, batch, batch_idx): calls self.generic_step(batch, batch_idx, "predict")

        As a baseline, self.generic_step support (top-k) classification and regression tasks.

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__()

        self.save_hyperparameters(cfg)

        self.model = get_class(torchvision.models, self.hparams.model["name"])(**self.hparams.model["cfg"])
        self.criterion = get_class(nn, self.hparams.criterion["name"])()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # TODO: support multiple optimizers and schedulers
        optimizer = get_class(optim, self.hparams.optimizer["name"])(self.model.parameters(),  **self.hparams.optimizer["cfg"])
        scheduler = get_class(optim.lr_scheduler, self.hparams.lr_scheduler["name"])(optimizer, **self.hparams.lr_scheduler["cfg"]) if "lr_scheduler" in self.hparams else None

        if scheduler is not None:
            return [optimizer], [scheduler]
        
        return optimizer
    
    def forward(self, x):
        return self.model(x)
    
    def generic_step(self, batch, batch_idx, phase: str):
        x, y = batch

        output = self(x)

        loss = self.criterion(*output if isinstance(output, tuple) else output, y)

        prog_bar = phase == 'val'

        self.log(f"{phase}/loss", loss, prog_bar=prog_bar)

        # TODO: add top-k support
        if self.hparams.task.get("classification", False) == True:
            self.log(f"{phase}/accuracy", calc_accuracy(output, y), prog_bar=prog_bar)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        # TODO: make sure that predict step works even when there are no labels
        return self.generic_step(batch, batch_idx, "predict")
