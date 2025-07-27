"""
Basic convolution block used across different architectures
"""

import torch
import torch.nn as nn

from ..utils import get_activation, get_2d_normalization


class ConvolutionBlock(torch.nn.Module):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        activation_params=None,
        normalization="batch_norm",
        normalization_params=None,
        **kwargs  # Additional parameters for conv2d
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs),
            get_2d_normalization(normalization, out_channels, **(normalization_params or {})),
            get_activation(activation, **(activation_params or {})),
        )

    def forward(self, x):
        return self.layers(x)
