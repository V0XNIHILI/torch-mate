from typing import Dict

import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch_mate.lightning import ConfigurableLightningModule

import torch_mate.lightning.lm.functional.prototypical as LFPrototypical
import torch_mate.lightning.lm.functional.supervised as LFSupervised


class PrototypicalNetwork(ConfigurableLightningModule):
    def generic_step(self, batch, batch_idx, phase: str):
        return LFPrototypical.generic_step(self, batch, batch_idx, phase)


class MetaBaseline(PrototypicalNetwork):
    def __init__(self, cfg: Dict):
        """Implements Meta-Baseline (https://arxiv.org/abs/2003.04390)

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__(cfg)

        self.linear = nn.Linear(self.hparams.pre_training["embedding_size"], self.hparams.task["train"]["num_classes"])

    def generic_step(self, batch, batch_idx, phase: str):
        if phase == 'train':
            x, y = batch

            output = self.linear(self.model(x))
            loss = LFSupervised.compute_loss(self.criterion, output, y)

            return output, loss
        
        return super().generic_step(batch, batch_idx, phase)
