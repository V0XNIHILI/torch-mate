import torch
import torch.nn as nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix.

    Code taken from: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L180

    Args:
        x (torch.Tensor): A square matrix

    Returns:
        torch.Tensor: A flattened view of the off-diagonal elements of the input
    """

    n, m = x.shape
    assert n == m

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class EmpiricalCrossCorrelation(nn.Module):

    def __init__(self, lambd: float = 0.0051):
        """Empirical Cross Correlation loss function.

        Code taken from: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L212

        Args:
            lambd (float, optional): Weight on off-diagonal entries. Defaults to 0.0051.
        """
        super(EmpiricalCrossCorrelation, self).__init__()

        self.lambd = lambd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # Empirical cross-correlation matrix
        c = z1.T @ z2

        batch_size = z1.shape[0]

        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag

        return loss
