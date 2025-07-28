"""
Basic convolution blocks used across different architectures
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
        # Simplified activation and normalization (most common usage)
        activation="relu",
        norm="batch_norm",
        # Advanced configuration (optional)
        activation_params=None,
        norm_params=None,
        dropout=0.0,  # Dropout after activation
        # Conv2d parameters passed directly
        **conv_kwargs
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs),
            get_2d_normalization(norm, out_channels, **(norm_params or {})),
            get_activation(activation, **(activation_params or {})),
        ]

        # Add dropout after activation
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PreActivationConvBlock(torch.nn.Module):
    """Pre-activation convolution block with BatchNorm + ReLU + Conv2d"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        # Simplified activation and normalization (most common usage)
        activation="relu",
        norm="batch_norm",
        # Advanced configuration (optional)
        activation_params=None,
        norm_params=None,
        dropout=0.0,  # Dropout after activation
        # Conv2d parameters passed directly
        **conv_kwargs
    ):
        super().__init__()

        layers = [
            get_2d_normalization(norm, in_channels, **(norm_params or {})),
            get_activation(activation, **(activation_params or {})),
        ]

        # Add dropout after activation (semantic consistency)
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
