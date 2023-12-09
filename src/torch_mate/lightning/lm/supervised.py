from torch_mate.lightning import ConfigurableLightningModule

import torch_mate.lightning.lm.functional.supervised as LFSupervised

class SupervisedLearner(ConfigurableLightningModule):
    def generic_step(self, batch, batch_idx, phase: str):
        return LFSupervised.generic_step(self, batch, batch_idx, phase)
