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

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input data. Shape: (batch_size, input_size, seq_len)
            h (Optional[torch.Tensor], optional): Hidden state. Defaults to None.
        """

        # (N, C_in, L_in) -> (L_in, N, C_in)
        #  0  1     2         2     0  1
        # LSTM expects (N, L_in, C_in)
        x = x.transpose(1, 2)

        _, (h_t, _) = self._lstm(x, h)
        h_t = h_t.squeeze(0)

        return self._linear(h_t)
