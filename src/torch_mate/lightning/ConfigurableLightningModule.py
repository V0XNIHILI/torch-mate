import torch.nn as nn
import torch.optim as optim

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from dotmap import DotMap

from torch_mate.utils import get_class


class ConfigurableLightningModule(L.LightningModule):
    def __init__(self, cfg: DotMap, model: nn.Module):
        super().__init__()

        self.model = model
        self.cfg = cfg

        self.save_hyperparameters(cfg.toDict())

        self.loss = get_class(nn, cfg.criterion.name)()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = get_class(optim, self.cfg.optimizer.name)(self.model.parameters(),  **self.cfg.optimizer.cfg.toDict())
        scheduler = get_class(optim.lr_scheduler, self.cfg.lr_scheduler.name)(optimizer, **self.cfg.lr_scheduler.cfg.toDict()) if self.cfg.lr_scheduler else None

        if scheduler is not None:
            return [optimizer], [scheduler]
        
        return optimizer