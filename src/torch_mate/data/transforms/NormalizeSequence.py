import torch
import torch.nn as nn


class NormalizeSequence(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        # Reshape to (num_channels, 1) to allow broadcasting
        # during normalization
        self.mean = torch.tensor(mean).unsqueeze(1)
        self.std = torch.tensor(std).unsqueeze(1)

    def forward(self, x):
        """Assumes that X has shape: (batch_size, num_channels, timesteps)"""

        device = x.device

        if self.mean.device != device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

        return (x - self.mean) / self.std
