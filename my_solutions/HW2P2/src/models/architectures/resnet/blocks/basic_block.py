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

        norm_layer = self._get_norm_layer(config)
        activation = self._get_activation(config)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.activation = activation

        # Shortcut connection with configurable projection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = self._build_shortcut(in_channels, out_channels, stride, config)
        else:
            self.shortcut = nn.Identity()

        # SE module if specified
        if config.get("use_se", False):
            self.se = SEModule.from_config(out_channels, config)
        else:
            self.se = nn.Identity()

        # Residual connection parameters
        self.residual_scale = config.get("residual_scale", 1.0)
        self.residual_dropout = (
            nn.Dropout(config.get("residual_dropout", 0.0))
            if config.get("residual_dropout", 0.0) > 0
            else nn.Identity()
        )

    def _build_shortcut(self, in_channels: int, out_channels: int, stride: int, config: Dict[str, Any]) -> nn.Module:
        """Build shortcut connection based on projection type"""
        projection_type = config.get("projection_type", "auto")
        norm_layer = self._get_norm_layer(config)
        projection_norm = config.get("projection_norm", True)

        # Auto mode: use conv for channel change, pooling for stride-only
        if projection_type == "auto":
            if in_channels != out_channels:
                projection_type = "conv"
            else:
                projection_type = "avg_pool"

        if projection_type == "conv":
            # Standard 1x1 convolution projection
            layers = [nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)]
            if projection_norm:
                layers.append(norm_layer(out_channels))
            return nn.Sequential(*layers)

        elif projection_type == "avg_pool":
            # Average pooling + optional channel adjustment
            layers = []
            if stride > 1:
                layers.append(nn.AvgPool2d(stride, stride))
            if in_channels != out_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
                if projection_norm:
                    layers.append(norm_layer(out_channels))
            return nn.Sequential(*layers) if layers else nn.Identity()

        elif projection_type == "max_pool":
            # Max pooling + optional channel adjustment
            layers = []
            if stride > 1:
                layers.append(nn.MaxPool2d(stride, stride))
            if in_channels != out_channels:
                layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
                if projection_norm:
                    layers.append(norm_layer(out_channels))
            return nn.Sequential(*layers) if layers else nn.Identity()

        else:
            # Fallback to standard conv projection
            layers = [nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)]
            if projection_norm:
                layers.append(norm_layer(out_channels))
            return nn.Sequential(*layers)

    def _get_activation(self, config: Dict[str, Any]) -> nn.Module:
        """Get activation function based on config"""
        activation = config.get("basic_activation", config.get("activation", "relu"))
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "elu":
            return nn.ELU(inplace=True)
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def _get_norm_layer(self, config: Dict[str, Any]):
        """Get normalization layer based on config"""
        norm = config.get("basic_normalization", config.get("normalization", "batch_norm"))
        if norm == "batch_norm":
            return nn.BatchNorm2d
        elif norm == "group_norm":
            return lambda channels: nn.GroupNorm(32, channels)
        elif norm == "layer_norm":
            return lambda channels: nn.GroupNorm(1, channels)
        elif norm == "instance_norm":
            return nn.InstanceNorm2d
        else:
            return nn.BatchNorm2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention if enabled
        out = self.se(out)

        # Apply residual connection with configurable scaling and dropout
        shortcut = self.shortcut(residual)
        shortcut = self.residual_dropout(shortcut)
        out += shortcut * self.residual_scale
        out = self.activation(out)

        return out
