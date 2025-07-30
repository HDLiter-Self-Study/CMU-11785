"""
Basic ResNet Block with optional SE module
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.base_resnet_block import BaseResNetBlock


class BasicBlock(BaseResNetBlock):
    """Basic ResNet block with optional SE module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        pre_activation: bool = False,
        activation: str = "relu",
        norm: str = "batch_norm",
        activation_params: Dict[str, Any] = None,
        norm_params: Dict[str, Any] = None,
        conv_drop_prob: float = 0.0,
        conv_drop_size: int = 1,
        projection_type: str = "conv",
        use_se: bool = False,
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
        **conv_kwargs,
    ):
        self.out_channels = out_channels
        super().__init__(
            in_channels=in_channels,
            stride=stride,
            pre_activation=pre_activation,
            activation=activation,
            norm=norm,
            activation_params=activation_params,
            norm_params=norm_params,
            conv_drop_prob=conv_drop_prob,
            conv_drop_size=conv_drop_size,
            projection_type=projection_type,
            use_se=use_se,
            layer_scale=layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            stochastic_depth_prob=stochastic_depth_prob,
            **conv_kwargs,
        )

    def _build_layers(self, in_channels: int) -> tuple[nn.Module, int]:
        """Build BasicBlock layers: 3x3 conv -> 3x3 conv"""
        layers = nn.Sequential(
            # First 3x3 conv (with stride)
            self.conv_block_cls(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                activation=self.activation,
                norm=self.norm,
                activation_params=self.activation_params,
                norm_params=self.norm_params,
                dropout_prob=self.conv_drop_prob,
                dropout_size=self.conv_drop_size,
                **self.conv_kwargs,
            ),
            # Second 3x3 conv (no activation if post-activation, no dropout)
            self.conv_block_cls(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=(self.activation if self.pre_activation else "none"),
                activation_params=self.activation_params if self.pre_activation else {},
                norm=self.norm,
                norm_params=self.norm_params,
                dropout_prob=0.0,  # No dropout before residual connection
                **self.conv_kwargs,
            ),
        )
        return layers, self.out_channels
