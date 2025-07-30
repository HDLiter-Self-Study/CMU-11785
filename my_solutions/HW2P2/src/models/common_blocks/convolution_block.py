"""
Basic convolution blocks used across different architectures
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from ..utils import get_activation, get_2d_normalization


class BaseConvolutionBlock(torch.nn.Module, ABC):
    """Base class for convolution blocks to reduce code duplication"""

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
        deep_separable=False,  # Use depthwise separable convolution
        # Conv2d parameters passed directly
        **conv_kwargs,
    ):
        super().__init__()

        # Validate deep_separable configuration
        if deep_separable and "groups" in conv_kwargs and conv_kwargs["groups"] != in_channels:
            raise ValueError(
                f"Conflicting groups parameter: {conv_kwargs['groups']} != {in_channels} when using depthwise separable convolution"
            )
        if deep_separable:
            conv_kwargs["groups"] = in_channels

        # Store parameters for layer building
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.norm = norm
        self.activation_params = activation_params or {}
        self.norm_params = norm_params or {}
        self.dropout = dropout
        self.conv_kwargs = conv_kwargs

        # Build layers using template method
        self.layers = nn.Sequential(*self._build_layers())

    @abstractmethod
    def _build_layers(self):
        """Build the layers in the specific order for this block type"""
        pass

    def _get_conv_layer(self):
        """Get the convolution layer"""
        return nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, **self.conv_kwargs
        )

    def _get_norm_layer(self, channels):
        """Get the normalization layer"""
        return get_2d_normalization(self.norm, channels, **self.norm_params)

    def _get_activation_layer(self):
        """Get the activation layer"""
        return get_activation(self.activation, **self.activation_params)

    def _get_dropout_layer(self):
        """Get the dropout layer if needed"""
        return nn.Dropout2d(self.dropout) if self.dropout > 0 else None

    def forward(self, x):
        return self.layers(x)


class ConvolutionBlock(BaseConvolutionBlock):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""

    def _build_layers(self):
        """Build layers in post-activation order: Conv → Norm → Activation → (Dropout)"""
        layers = [
            self._get_conv_layer(),
            self._get_norm_layer(self.out_channels),
            self._get_activation_layer(),
        ]

        # Add dropout after activation
        dropout_layer = self._get_dropout_layer()
        if dropout_layer is not None:
            layers.append(dropout_layer)

        return layers


class PreActivationConvBlock(BaseConvolutionBlock):
    """Pre-activation convolution block with BatchNorm + ReLU + Conv2d"""

    def _build_layers(self):
        """Build layers in pre-activation order: Norm → Activation → (Dropout) → Conv"""
        layers = [
            self._get_norm_layer(self.in_channels),
            self._get_activation_layer(),
        ]

        # Add dropout after activation (semantic consistency)
        dropout_layer = self._get_dropout_layer()
        if dropout_layer is not None:
            layers.append(dropout_layer)

        layers.append(self._get_conv_layer())
        return layers
