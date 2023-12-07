from torch_mate.lightning import ConfigurableLightningModule
import torch_mate.lightning.lm.functional.prototypical as LFPrototypical

class PrototypicalNetwork(ConfigurableLightningModule):
    def generic_step(self, batch, batch_idx, phase: str):
        return LFPrototypical.generic_step(self, batch, batch_idx, phase)
