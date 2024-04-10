from typing import Dict

import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch_mate.lightning import ConfigurableLightningModule

import torch_mate.lightning.lm.functional.prototypical as LFPrototypical
import torch_mate.lightning.lm.functional.supervised as LFSupervised
# import torch_mate.lightning.lm.functional.maml as LFMAML

# from torch_mate.lightning.utils.MAML import MAML as PTMAML

class PrototypicalNetwork(ConfigurableLightningModule):
    def generic_step(self, batch, batch_idx, phase: str):
        return LFPrototypical.generic_step(self, batch, batch_idx, phase)
    
    def configure_model(self):
        model = super().configure_model()

        if self.hparams.learner.get("cfg", {}).get("embedder_key", None):
            return dict(model.named_modules())[self.hparams.learner["cfg"]["embedder_key"]]
        
        return model

# class MAML(ConfigurableLightningModule):
#     def __init__(self, cfg: Dict):
#         """Implements Model-Agnostic Meta-Learning (https://arxiv.org/abs/1703.03400)

#         Args:
#             cfg (Dict): configuration dictionary
#         """
            
#         super().__init__(cfg)
    
#         self.maml = PTMAML(self.model, self.hparams.few_shot["cfg"]["inner_loop_lr"], self.hparams.few_shot["cfg"].get("first_order", False))

#     def generic_step(self, batch, batch_idx, phase: str):
#         return LFMAML.generic_step(self, batch, batch_idx, phase)
    
#     def on_before_optimizer_step(self, optimizer: Optimizer):
#         # Average the accumulated gradients and optimize
#         for p in self.maml.parameters():
#             if p.grad is not None:
#                 p.grad.data.mul_(1.0 / meta_batch_size)
            