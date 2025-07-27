"""
Basic convolution block used across different architectures
"""

import torch
import torch.nn as nn


class ConvolutionBlock(torch.nn.Module):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.layers = None  # TODO

    def forward(self, x):
        return None  # TODO
