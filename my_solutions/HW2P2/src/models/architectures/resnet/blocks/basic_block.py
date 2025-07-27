"""
Basic ResNet Block with optional SE module
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.attention import SEModule


class BasicBlock(nn.Module):
    """Basic ResNet block with optional SE module"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, config: Dict[str, Any]):
        super().__init__()

        # TODO: get normalization and activation from config
        norm_layer = None  # TODO
        activation = None  # TODO

        # TODO: 3x3 conv -> bn -> 3x3 conv -> bn
        self.conv1 = None  # TODO: 3x3 conv with stride
        self.bn1 = None  # TODO: batch norm
        self.conv2 = None  # TODO: 3x3 conv, stride=1
        self.bn2 = None  # TODO: batch norm

        self.activation = None  # TODO

        # TODO: shortcut connection logic
        # if stride != 1 or in_channels != out_channels:
        self.shortcut = None  # TODO: build shortcut or Identity

        # TODO: SE module conditional creation
        self.se = None  # TODO: SEModule or Identity

        # TODO: residual connection parameters
        self.residual_scale = None  # TODO
        self.residual_dropout = None  # TODO

    def _build_shortcut(self, in_channels: int, out_channels: int, stride: int, config: Dict[str, Any]) -> nn.Module:
        """Build shortcut connection based on projection type"""
        # TODO: implement projection_type logic (auto, conv, avg_pool, max_pool)
        # TODO: handle stride and channel changes
        return None  # TODO

    def _get_activation(self, config: Dict[str, Any]) -> nn.Module:
        """Get activation function based on config"""
        # TODO: support relu, gelu, swish, elu, leaky_relu
        return None  # TODO

    def _get_norm_layer(self, config: Dict[str, Any]):
        """Get normalization layer based on config"""
        # TODO: support batch_norm, group_norm, layer_norm, instance_norm
        return None  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # TODO: conv1 -> bn1 -> activation
        out = None  # TODO
        out = None  # TODO
        out = None  # TODO

        # TODO: conv2 -> bn2 (no activation yet)
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
