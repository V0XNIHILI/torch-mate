import torch
import torch.nn as nn

from torch_mate.nn import Lambda


def conv_block(in_channels: int,
               out_channels: int,
               batch_norm: bool = True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))


class OmniglotCNN(nn.Module):

    def __init__(self, num_classes: int, batch_norm: bool = True):
        """Omniglot model as described in the original MAML paper. Code based
        on Tensorflow implementation from Reptile repository
        https://github.com/openai/supervised-
        reptile/blob/master/supervised_reptile/models.py, PyTorch
        implementation from https://github.com/gabrielhuang/reptile-
        pytorch/blob/master/models.py and original MAML code
        https://github.com/cbfinn/maml/blob/master/utils.py.

        Args:
            num_classes (int): How many classes to classify with this model.
            batch_norm (bool): Whether to use batch normalization or not.
        """

        super(OmniglotCNN, self).__init__()

        self.embedder = nn.Sequential(conv_block(1, 64, batch_norm),
                                    conv_block(64, 64, batch_norm),
                                    conv_block(64, 64, batch_norm),
                                    conv_block(64, 64, batch_norm),
                                    Lambda(lambda x: x.view(len(x), -1)))

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        out = self.embedder(x)
        out = self.classifier(out)

        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = OmniglotCNN(5)

    summary(model, (1, 28, 28))
