import torch
import torch.nn as nn
from typing import Optional


class ClassificationRNNBase(nn.Module):
    def __init__(self, rnn_layer: nn.Module, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = rnn_layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (N, C_in, L_in) → (N, L_in, C_in)
        x = x.transpose(1, 2)
        out = self.rnn(x, h)

        if isinstance(self.rnn, nn.LSTM):
            _, (h_n, _) = out
        else:
            _, h_n = out

        h_t = h_n.squeeze(0)
        return self.linear(h_t)


class ClassificationRNN(ClassificationRNNBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(
            nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh"),
            hidden_size,
            output_size,
        )


class ClassificationGRU(ClassificationRNNBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(
            nn.GRU(input_size, hidden_size, batch_first=True),
            hidden_size,
            output_size,
        )


class ClassificationLSTM(ClassificationRNNBase):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(
            nn.LSTM(input_size, hidden_size, batch_first=True),
            hidden_size,
            output_size,
        )
