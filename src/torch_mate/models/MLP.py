from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, layer_sizes: List[int], with_last_activation: bool = False, dropout: float = 0.0):
        super(MLP, self).__init__()

        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i != len(layer_sizes) - 2 or with_last_activation:
                layers.append(nn.ReLU())

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == '__main__':
    from torchsummary import summary

    model = MLP([28 * 28, 100, 100, 10], with_last_activation=True, dropout=0.5)

    summary(model, (5, 28 * 28))
