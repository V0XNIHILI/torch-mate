from torch_mate.utils import calc_accuracy

def generic_step(self, batch, batch_idx, phase: str):
    x, y = batch

    output = self(x)

    loss = self.criterion(*output if isinstance(output, tuple) else output, y)

    prog_bar = phase == 'val'

    self.log(f"{phase}/loss", loss, prog_bar=prog_bar)

    # TODO: add top-k support
    if self.hparams.task.get("classification", False) == True:
        self.log(f"{phase}/accuracy", calc_accuracy(output, y), prog_bar=prog_bar)

    return loss
