"""
Bottleneck ResNet Block with optional SE module
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_resnet_block import BaseResNetBlock


class BottleneckBlock(BaseResNetBlock):
    """Bottleneck ResNet block with optional SE module"""

    def __init__(
        self,
        in_channels: int,
        neck_channels: int,
        stride: int = 1,
        expansion: int = 4,
        pre_activation: bool = False,
        activation: str = "relu",
        norm: str = "batch_norm",
        activation_params: Dict[str, Any] = None,
        norm_params: Dict[str, Any] = None,
        conv_dropout: float = 0.0,
        projection_type: str = "conv",
        use_se: bool = False,
        **conv_kwargs,
    ):
        self.neck_channels = neck_channels
        self.expansion = expansion
        super().__init__(
            in_channels=in_channels,
            stride=stride,
            pre_activation=pre_activation,
            activation=activation,
            norm=norm,
            activation_params=activation_params,
            norm_params=norm_params,
            conv_dropout=conv_dropout,
            projection_type=projection_type,
            use_se=use_se,
            **conv_kwargs,
        )

    def _build_layers(self, in_channels: int) -> tuple[nn.Module, int]:
        """Build BottleneckBlock layers: 1x1 conv -> 3x3 conv -> 1x1 conv"""
        out_channels = self.expansion * self.neck_channels

        layers = nn.Sequential(
            # 1x1 conv (reduce)
            self.conv_block_cls(
                in_channels=in_channels,
                out_channels=self.neck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=self.activation,
                norm=self.norm,
                activation_params=self.activation_params,
                norm_params=self.norm_params,
                dropout=self.conv_dropout,
                **self.conv_kwargs,
            ),
            # 3x3 conv (process with stride)
            self.conv_block_cls(
                in_channels=self.neck_channels,
                out_channels=self.neck_channels,
                kernel_size=3,
                stride=self.stride,  # Downsampling on 3x3 conv to avoid info loss
                padding=1,
                activation=self.activation,
                norm=self.norm,
                activation_params=self.activation_params,
                norm_params=self.norm_params,
                dropout=self.conv_dropout,
                **self.conv_kwargs,
            ),
            # 1x1 conv (expand, no activation if post-activation, no dropout)
            self.conv_block_cls(
                in_channels=self.neck_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=(self.activation if self.pre_activation else "none"),
                activation_params=self.activation_params if self.pre_activation else {},
                norm=self.norm,
                norm_params=self.norm_params,
                dropout=0.0,  # No dropout before residual connection
                **self.conv_kwargs,
            ),
        )
        return layers, out_channels
