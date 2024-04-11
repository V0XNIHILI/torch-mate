from typing import Dict

import torch.nn as nn
import torch.optim as optim

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torch_mate.utils import get_class, get_modules


class ConfigurableLightningModule(L.LightningModule):
    def __init__(self, cfg: Dict):
        """Lightweight wrapper around PyTorch Lightning LightningModule that adds support for a configuration dictionary.
        Based on this configuration, it creates the model, criterion, optimizer, and scheduler. Overall, compared to the
        PyTorch Lightning LightningModule, the following three attributes are added:
        
        - `self.get_model()`: the created model
        - `self.get_criteria`: the created criterion
        - `self.generic_step(self, batch, batch_idx, phase)`: a generic step function that is shared across all steps (train, val, test, predict)

        Based on these, the following methods are automatically implemented:

        - `self.forward(self, x)`: calls `self.model(x)`
        - `self.training_step(self, batch, batch_idx)`: calls `self.generic_step(batch, batch_idx, "train")`
        - `self.validation_step(self, batch, batch_idx)`: calls `self.generic_step(batch, batch_idx, "val")`
        - `self.test_step(self, batch, batch_idx)`: calls `self.generic_step(batch, batch_idx, "test")`
        - `self.predict_step(self, batch, batch_idx)`: calls `self.generic_step(batch, batch_idx, "predict")`
        - `self.configure_optimizers(self)`: creates the optimizer and scheduler based on the configuration dictionary

        You can also override `self.configure_model(self)`, `self.configure_criteria(self)` and `self.configure_configuration(self, cfg: Dict)` to customize the model and criterion creation and the hyperparameter setting.

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__()

        self.save_hyperparameters(self.configure_configuration(cfg))

        self._model = self.configure_model()
        self._criteria = self.configure_criteria()

    def configure_configuration(self, cfg: Dict):
        return cfg

    def configure_model(self):
        return get_modules(None, self.hparams.model)
    
    def get_model(self):
        return self._model

    def configure_criteria(self):
        return get_modules(nn, self.hparams.criterion)
    
    def get_criteria(self):
        return self._criteria

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # TODO: support multiple schedulers
        opt_configs = self.hparams.optimizer if type(self.hparams.optimizer) is list else [self.hparams.optimizer]

        optimizers = [get_class(optim, opt_cfg["name"])(self.get_model().parameters(), **opt_cfg["cfg"]) for opt_cfg in opt_configs]

        scheduler = get_class(optim.lr_scheduler, self.hparams.lr_scheduler["name"])(optimizers[0], **self.hparams.lr_scheduler["cfg"]) if "lr_scheduler" in self.hparams else None

        if scheduler is not None:
            return optimizers, [scheduler]
        
        return optimizers
    
    def forward(self, x):
        return self.get_model()(x)
    
    def generic_step(self, batch, batch_idx, phase: str):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.generic_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        # TODO: make sure that predict step works even when there are no labels
        return self.generic_step(batch, batch_idx, "predict")
