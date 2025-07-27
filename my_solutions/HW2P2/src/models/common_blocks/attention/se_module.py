"""
Squeeze-and-Excitation (SE) Module

This is a general-purpose attention mechanism that can be integrated
into any convolutional architecture to improve channel-wise feature representation.

Reference:
- Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module

    A lightweight attention mechanism that adaptively recalibrates
    channel-wise feature responses.
    """

    def __init__(self, channels: int, reduction: int = 16, activation: str = "relu"):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
            activation: Activation function to use in the excitation step
        """
        super().__init__()

        reduced_channels = max(1, channels // reduction)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            self._get_activation(activation),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "elu":
            return nn.ELU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W) with SE attention applied
        """
        b, c, _, _ = x.size()

        # Squeeze: Global average pooling
        y = self.global_pool(x).view(b, c)

        # Excitation: FC layers to learn channel weights
        y = self.fc(y).view(b, c, 1, 1)

        # Scale: Apply learned weights to input
        return x * y.expand_as(x)

    @classmethod
    def from_config(cls, channels: int, config: Dict[str, Any]) -> "SEModule":
        """Create SE module from configuration"""
        return cls(
            channels=channels, reduction=config.get("se_reduction", 16), activation=config.get("se_activation", "relu")
        )
