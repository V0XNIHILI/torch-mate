import torch
import torch.nn.functional as F

from torch_mate.lightning import ConfigurableLightningModule

def forward(module: ConfigurableLightningModule, x):
    output = module(torch.cat(x))

    anchor_output, unknown_output = output.chunk(2)

    # Compute the distance between the anchor and the unknown, both of shape (batch size, embedding size)
    return F.sigmoid(F.pairwise_distance(anchor_output, unknown_output, p = 1))
