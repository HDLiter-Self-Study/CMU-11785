"""
Basic convolution block used across different architectures
"""

import torch
import torch.nn as nn


class ConvolutionBlock(torch.nn.Module):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
