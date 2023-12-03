from typing import Optional

import torch.nn as nn
import torch.optim as optim

import torchvision

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from dotmap import DotMap

from torch_mate.utils import get_class, calc_accuracy


class ConfigurableLightningModule(L.LightningModule):
    def __init__(self, cfg: DotMap, model: Optional[nn.Module] = None):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(cfg.toDict())

        self.model = model if model else get_class(torchvision.models, cfg.model.name)(**cfg.model.cfg.toDict())
        self.loss = get_class(nn, cfg.criterion.name)()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = get_class(optim, self.cfg.optimizer.name)(self.model.parameters(),  **self.cfg.optimizer.cfg.toDict())
        scheduler = get_class(optim.lr_scheduler, self.cfg.lr_scheduler.name)(optimizer, **self.cfg.lr_scheduler.cfg.toDict()) if self.cfg.lr_scheduler else None

        if scheduler is not None:
            return [optimizer], [scheduler]
        
        return optimizer
    
    def forward(self, x):
        return self.model(x)
    
    def generic_step(self, batch, batch_idx, phase: str):
        x, y = batch

        output = self(x)

        loss = self.loss(output, y)

        prog_bar = phase == 'val'

        self.log(f"{phase}/loss", loss, prog_bar=prog_bar)

        if self.cfg.task.classification and self.cfg.task.classification == True:
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
