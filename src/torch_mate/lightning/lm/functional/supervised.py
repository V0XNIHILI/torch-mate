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
    output, loss = process_supervised_batch(module, batch, module.criterion)

    prog_bar = phase == 'val'

    module.log(f"{phase}/loss", loss, prog_bar=prog_bar)

    # TODO: add top-k support
    if module.hparams.task.get("classification", False) == True:
        module.log(f"{phase}/accuracy", calc_accuracy(output, batch[1]), prog_bar=prog_bar)

    return loss
