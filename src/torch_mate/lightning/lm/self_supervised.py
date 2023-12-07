from torch_mate.lightning.lm import SupervisedLearner
from torch_mate.lightning import ConfigurableLightningModule

from torch_mate.lightning.lm.functional import siamese as LFSiamese
from torch_mate.lightning.lm.functional import triplet as LFTriplet

class SiameseNetwork(SupervisedLearner):
    def forward(self, x):
        LFSiamese.forward(self, x)


class TripletNetwork(ConfigurableLightningModule):
    def forward(self, x):
        LFTriplet.forward(self, x)

    def generic_step(self, batch, batch_idx, phase: str):
        return LFTriplet.generic_step(self, batch, batch_idx, phase)
