import torch
import torch.nn as nn


class OneHot(nn.Module):
    def __init__(self, num_classes: int = -1):
        super(OneHot, self).__init__()

        self.num_classes = num_classes

    def forward(self, input: torch.Tensor):
        return nn.functional.one_hot(input, num_classes=self.num_classes)
