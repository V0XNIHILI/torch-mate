from typing import List, Optional, Dict, Any

import torch.nn as nn


class MLP(nn.Sequential):

    def __init__(self, layer_sizes: List[int],
                 activation_function: str = 'ReLU',
                 activation_kwargs: Optional[Dict[str, Any]] = None,
                 with_last_activation: bool = False,
                 dropout: float = 0.0):
        """Create a multi-layer perceptron (MLP) of given size(s) and activation function.

        Args:
            layer_sizes (List[int]): The sizes of the layers, where the first entry determines the input size of the MLP.
            activation_function (str, optional): Which `nn.*` activation function to use. Defaults to 'ReLU'.
            activation_kwargs (Optional[Dict[str, Any]], optional): Arguments to pass to activation function constructor. Defaults to None.
            with_last_activation (bool, optional): Whether to have an activation function after the last fully connected layer. Defaults to False.
            dropout (float, optional): How much dropout to use. Defaults to 0.0.
        """
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i != len(layer_sizes) - 2 or with_last_activation:
                activation_function_class = getattr(nn, activation_function)

                if activation_kwargs is not None:
                    activation_function_instance = activation_function_class(**activation_kwargs)
                else:
                    activation_function_instance = activation_function_class()

                layers.append(activation_function_instance)

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*layers)


if __name__ == '__main__':
    from torchsummary import summary

    model = MLP([28 * 28, 100, 100, 10], with_last_activation=False, dropout=0.5, activation_function='ELU', activation_kwargs={'alpha': 0.1})

    summary(model, (5, 28 * 28))
