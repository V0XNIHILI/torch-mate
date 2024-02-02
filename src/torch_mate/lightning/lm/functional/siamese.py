import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(module: nn.Module, x):
    output = module(torch.cat(x))

    anchor_output, unknown_output = output.chunk(2)

    # Compute the distance between the anchor and the unknown, both of shape (batch size, embedding size)
    return F.sigmoid(F.pairwise_distance(anchor_output, unknown_output, p = 1, keepdim=True))
