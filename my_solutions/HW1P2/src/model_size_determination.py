import torch.nn as nn
import numpy as np


class DummyModel(nn.Module):
    """
    A model for determining the size of a neural network based on specified hidden layer sizes.
    """

    def __init__(
        self,
        input_size,
        output_size,
        batch_norm=False,
        activation="relu",
        hidden_sizes=None,  # Accept a list of hidden layer sizes directly
        input_dropout=0.1,
        hidden_dropout=0.3,
    ):
        """
        :param input_size: Size of the input layer.
        :param output_size: Size of the output layer.
        :param hidden_sizes: List of hidden layer sizes.
        """
        super(DummyModel, self).__init__()
        layers = []
        in_features = input_size
        if hidden_sizes is None:
            raise ValueError("hidden_sizes must be provided as a list of hidden layer sizes.")
        layers.append(nn.Dropout(p=input_dropout))  # Add a slight dropout at the input layer
        activation_function = self._get_activation_function(activation)
        # Create the hidden layers based on the specified sizes
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(
                nn.Linear(in_features, hidden_size, bias=not batch_norm)
            )  # Add bias only if batch normalization is not used
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_function)
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(p=hidden_dropout))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _get_activation_function(activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def summary(self, input_shape, device="cpu"):
        """
        Print a summary of the model using torchsummaryX.
        :param input_shape: Shape of the input tensor (excluding batch dimension).
        :param device: Device to use for the summary.
        """
        try:
            from torchsummaryX import summary
            import torch

            dummy_input = torch.zeros((2, *input_shape)).to(device)
            summary(self, dummy_input)
        except ImportError:
            print("torchsummaryX is not installed. Please install it to use the summary feature.")


# Example usage:
from models.sweep_model import CUSTOM_SHAPES

hiddens = CUSTOM_SHAPES
for hidden_name, hidden_sizes in hiddens.items():
    print(f"Model with hidden sizes {hidden_name}:")
    model = DummyModel(input_size=1428, output_size=42, hidden_sizes=hidden_sizes)
    model.summary((1428,), device="cpu")
    print("\n" + "=" * 50 + "\n")
