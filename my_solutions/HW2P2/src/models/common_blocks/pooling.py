"""2D Adaptive Pooling for higher Moments"""

import torch
from torch import nn
import torch.nn.functional as F


class VarAdaptivePool2d(nn.Module):
    """Variance Adaptive Pooling layer for 2D inputs"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # Compute variance adaptive pooling
        mean = F.adaptive_avg_pool2d(x, self.output_size)
        mean_sq = F.adaptive_avg_pool2d(x**2, self.output_size)
        var = mean_sq - mean**2
        return var


class SkewAdaptivePool2d(nn.Module):
    """Skewness Adaptive Pooling layer for 2D inputs"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        # Compute skewness adaptive pooling
        mean = F.adaptive_avg_pool2d(x, self.output_size)
        mean_sq = F.adaptive_avg_pool2d(x**2, self.output_size)
        mean_cubed = F.adaptive_avg_pool2d(x**3, self.output_size)

        var = mean_sq - mean**2
        skewness = (mean_cubed - 3 * mean * var - mean**3) / (var**1.5 + 1e-8)
        return skewness


class StackedPooling(nn.Module):
    """
    Stacked pooling module that combines multiple pooling methods.
    Useful for SE modules that require multiple channels for squeeze operations.
    """

    def __init__(self, pooling_layers: list[torch.nn.Module] = None):
        super().__init__()
        self.pooling_layers = pooling_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_outputs = []
        for layer in self.pooling_layers:
            pooled_outputs.append(layer(x).squeeze(-1).squeeze(-1))  # Squeeze to (B, C) shape
        return torch.stack(pooled_outputs, dim=-1)  # Shape: (B, C, num_layers)
