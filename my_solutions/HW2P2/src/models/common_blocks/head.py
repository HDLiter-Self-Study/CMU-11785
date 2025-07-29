import torch
import torch.nn as nn

from ..utils import get_activation, get_2d_normalization
from typing import Dict, Any


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        norm: str = "batch_norm",
        activation: str = "relu",
        activation_params: Dict[str, Any] = None,
        norm_params: Dict[str, Any] = None,
        hidden_dims: int = None,
    ):
        """
        Initialize the classification head.

        Args:
            in_features: Number of input features after global pooling.
            num_classes: Number of output classes for classification.
            dropout_rate: Dropout rate to apply before the final classifier.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling, output shape will be (batch_size, in_features, 1, 1)
            nn.Flatten(),  # Flatten to (batch_size, in_features)
            nn.Linear(in_features, hidden_dims) if hidden_dims else nn.Identity(),
            get_activation(activation, activation_params),
            get_2d_normalization(norm, norm_params),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims if hidden_dims else in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Args:
            x: Input tensor of shape (batch_size, in_features, height, width).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.layers(x)
