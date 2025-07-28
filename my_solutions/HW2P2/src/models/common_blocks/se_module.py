"""
Squeeze-and-Excitation (SE) Module

This is a general-purpose attention mechanism that can be integrated
into any convolutional architecture to improve channel-wise feature representation.

Reference:
- Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..common_blocks.pooling import VarAdaptivePool2d, SkewAdaptivePool2d, StackedPooling
from ..utils import get_activation


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module (ECANet style)

    A lightweight attention mechanism that adaptively recalibrates
    channel-wise feature responses.
    """

    # The best kernel sizes for skew and var pooling fusion conv1d layers
    # Got from <https://arxiv.org/abs/2403.01713v1>
    FUSION_KERNEL_SIZE = {"skew": 3, "var": 11}

    def __init__(
        self,
        channels: int,
        pooling: str = "avg",
    ):
        """
        Args:
            channels: Number of input channels
            pooling: Pooling method to use in the squeeze step (avg, max, skew, var)

        Note:
            - Use conv1d instead of FC for excitation as in ECAN, see <https://arxiv.org/abs/1910.03151>
            - If pooling is 'skew' or 'var', it will be concatenated with avg pooling, and conv1d will be used for channel fusion.
            - See <https://arxiv.org/abs/2403.01713v1> for more details on skew and var pooling.
        """
        super().__init__()
        self.channels = channels
        self.activation = get_activation("sigmoid")
        self.squeeze_layer = self._get_squeeze_layer(pooling)
        self.excitation_layer = self._get_excitation_layer(pooling)

    def _get_squeeze_layer(self, pooling: str):
        """Get the pooling layer based on the specified method"""
        if pooling == "avg":
            return nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            return nn.AdaptiveMaxPool2d(1)
        elif pooling == "skew":
            return StackedPooling([nn.AdaptiveAvgPool2d(1), SkewAdaptivePool2d(1)])
        elif pooling == "var":
            return StackedPooling([nn.AdaptiveAvgPool2d(1), VarAdaptivePool2d(1)])
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

    def _get_heuristic_kernel_size(self):
        # Heuristic kernel size for non-fusion case, see <https://arxiv.org/pdf/1910.03151>
        kernel_size = int(torch.log2(self.channels) / 2 + 1 / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _get_excitation_layer(self, pooling: str):

        class DimensionRearrange(nn.Module):

            def __init__(self, transpose: Tuple):
                super().__init__()
                self.transpose = transpose

            def forward(self, x):
                return x.transpose(**self.transpose)

        """Get the excitation layer based on whether to use conv or FC"""
        # Get the kernel size based on pooling type
        default_kernel_size = self._get_heuristic_kernel_size()
        kernel_size = self.FUSION_KERNEL_SIZE.get(pooling, default_kernel_size)
        if pooling in ["skew", "var"]:
            num_layers = 2  # avg + skew/var
        else:
            num_layers = 1

        excitation_layers = nn.Sequential(
            DimensionRearrange(transpose=(1, 2)),  # Rearranged (B, C, num_layers) -> (B, num_layers, C)
            nn.Conv1d(
                in_channels=num_layers,
                out_channels=1,
                kernel_size=kernel_size,
                padding="same",
                bias=False,  # No bias in conv layers
            ),  # Reduced to (B, 1, C)
            self.activation,
            DimensionRearrange(transpose=(1, 2)),  # Rearranged back to (B, C, 1)
        )

        return excitation_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W) with SE attention applied
        """

        # Squeeze: Global pooling
        y = self.squeeze_layer(x)
        if y.dim() == 4:
            y = y.squeeze(-1)  # Squeeze to (B, C, num_layers)

        # Excitation: Conv1d layers to learn channel weights
        y = self.excitation_layer(y)  # Shape: (B, C, 1)
        y = y.unsqueeze(-1)  # Reshape to (B, C, 1, 1)

        # Scale: Apply learned weights to input
        return x * y
