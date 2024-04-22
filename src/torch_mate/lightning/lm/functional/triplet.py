import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_mate.lightning import ConfigurableLightningModule

def forward(module: nn.Module, x):
    output = module(torch.cat(x))

    anchor_output, positive_output, negative_output = output.chunk(3)

    return anchor_output, positive_output, negative_output


def shared_step(module: ConfigurableLightningModule, batch, batch_idx, phase: str):
    anchor_output, positive_output, negative_output = forward(module, batch)

    loss = module.get_criteria()(anchor_output, positive_output, negative_output)

    p = module.hparams.self_supervised["cfg"]["norm_degree"]

    d_i_att_p = F.pairwise_distance(anchor_output, positive_output, p=p)
    d_i_att_n = F.pairwise_distance(anchor_output, negative_output, p=p)

    margin = module.hparams.criterion["cfg"]["margin"] if "cfg" in module.hparams.criterion else 0.0

    accuracy = torch.sum((d_i_att_n - d_i_att_p) > margin).item() / len(d_i_att_p)

    prog_bar = phase == 'val'

    module.log_dict({
        f"{phase}/loss": loss,
        f"{phase}/accuracy": accuracy
    }, prog_bar=prog_bar)

    return loss
