import torch


class CenterCropOrPad(torch.nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("Input must have shape (1, x)")

        length = x.shape[1]
        if length > self.target_length:
            # Center crop
            start = (length - self.target_length) // 2
            return x[:, start:start + self.target_length]
        else:
            # Center pad
            pad_before = (self.target_length - length) // 2
            pad_after = self.target_length - length - pad_before
            return torch.nn.functional.pad(x, (pad_before, pad_after), mode="constant", value=0)
