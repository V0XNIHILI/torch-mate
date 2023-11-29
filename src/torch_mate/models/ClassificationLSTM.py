import torch
import torch.nn as nn


from typing import Optional

import torch
import torch.nn as nn


class ClassificationLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """This is a simple LSTM that with a linear classifier attached to the
        LAST hidden state.

        Args:
            input_size (int): Input size of the LSTM.
            hidden_size (int): Hidden size of the LSTM.
            output_size (int): Output size of the linear layer.
        """

        super(ClassificationLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input data. Shape: (batch_size, input_size, seq_len)
            h (Optional[torch.Tensor], optional): Hidden state. Defaults to None.
        """

        # (N, C_in, L_in) -> (N, L_in, C_in)
        # LSTM with batch_first=True expects (N, L_in, C_in)
        x = x.transpose(1, 2)

        _, (h_t, _) = self.lstm(x, h)
        h_t = h_t.squeeze(0)

        return self.linear(h_t)
