import torch.nn as nn

from torch_mate.lightning import ConfigurableLightningModule

from torch_mate.utils import calc_accuracy


def compute_loss(criterion: nn.Module, output, target):
    if isinstance(output, tuple):
        return criterion(*output, target)
    else:
        return criterion(output, target)


def process_supervised_batch(model: nn.Module, batch, criterion: nn.Module):
    x, y = batch

    output = model(x)
    loss = compute_loss(criterion, output, y)

    return output, loss


def generic_step(module: ConfigurableLightningModule, batch, batch_idx, phase: str):
    output, loss = process_supervised_batch(module, batch, module.get_criteria())

    prog_bar = phase == 'val'

    module.log(f"{phase}/loss", loss, prog_bar=prog_bar)

    if module.hparams.learner.get("cfg", {}).get("classification", False) == True:
        if "topk" in module.hparams.learner["cfg"]:
            for i, k in enumerate(module.hparams.learner["cfg"]["topk"]):
                module.log(f"{phase}/accuracy@{k}", calc_accuracy(output, batch[1], k), prog_bar=(i == 0 and prog_bar))
        else:
            module.log(f"{phase}/accuracy", calc_accuracy(output, batch[1]), prog_bar=prog_bar)

    return loss
