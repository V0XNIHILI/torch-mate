import torch_mate.lightning.lm as lm
from torch_mate.lightning import ConfigurableLightningModule

from torch_mate.lightning.lm.functional import siamese as LFSiamese
from torch_mate.lightning.lm.functional import triplet as LFTriplet

class SiameseNetwork(lm.SupervisedLearner):
    def forward(self, x):
        return LFSiamese.forward(self, x)


class TripletNetwork(ConfigurableLightningModule):
    def forward(self, x):
        return LFTriplet.forward(self, x)

    def shared_step(self, batch, batch_idx, phase: str):
        return LFTriplet.shared_step(self, batch, batch_idx, phase)
