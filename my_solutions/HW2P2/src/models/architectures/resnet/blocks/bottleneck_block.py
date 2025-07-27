"""
Bottleneck ResNet Block with optional SE module
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.attention import SEModule


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet block with optional SE module"""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int, config: Dict[str, Any]):
        super().__init__()

        # TODO: get norm and activation layers from config
        norm_layer = None  # TODO
        activation = None  # TODO

        # TODO: Bottleneck design: 1x1 -> 3x3 -> 1x1
        self.conv1 = None  # TODO: 1x1 reduce
        self.bn1 = None  # TODO

        self.conv2 = None  # TODO: 3x3 with stride
        self.bn2 = None  # TODO

        self.conv3 = None  # TODO: 1x1 expand
        self.bn3 = None  # TODO

        self.activation = None  # TODO

        # TODO: shortcut connection logic
        # if stride != 1 or in_channels != out_channels * self.expansion:
        self.shortcut = None  # TODO

        # TODO: SE module for final output channels
        self.se = None  # TODO

        # TODO: residual parameters
        self.residual_scale = None  # TODO
        self.residual_dropout = None  # TODO

    def _build_shortcut(self, in_channels: int, out_channels: int, stride: int, config: Dict[str, Any]) -> nn.Module:
        """Build shortcut connection based on projection type"""
        # TODO: implement projection logic (auto, conv, avg_pool, max_pool)
        return None  # TODO

    def _get_activation(self, config: Dict[str, Any]) -> nn.Module:
        """Get activation function based on config"""
        # TODO: support multiple activation types
        return None  # TODO

    def _get_norm_layer(self, config: Dict[str, Any]):
        """Get normalization layer based on config"""
        # TODO: support different normalization types
        return None  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # TODO: 1x1 conv -> bn -> activation (reduce)
        out = None  # TODO
        out = None  # TODO
        out = None  # TODO

        # TODO: 3x3 conv -> bn -> activation (process)
        out = None  # TODO
        out = None  # TODO
        out = None  # TODO

        # TODO: 1x1 conv -> bn (expand, no activation)
        out = None  # TODO
        out = None  # TODO

        # TODO: SE attention
        out = None  # TODO

        # TODO: residual connection
        shortcut = None  # TODO
        shortcut = None  # TODO: dropout
        out = None  # TODO: add with scaling
        out = None  # TODO: final activation

        return out
