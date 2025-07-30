"""
ConvNeXt R
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.base_resnet_block import BaseResNetBlock


class InvertedBottleneckBlock(BaseResNetBlock):
    """
    Inverted Bottleneck block with optional SE support
    This block is designed to be used in ConvNeXt architectures.
    See https://arxiv.org/abs/2201.03545 for details.

    """

    def __init__(
        self,
        in_channels: int,
        neck_channels: int,
        expansion: int = 4,
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
        self.neck_channels = neck_channels
        self.expansion = expansion
        super().__init__(
            in_channels=in_channels,
            stride=1,  # ConvNeXt has separate downsample layers, so stride is not used here
            pre_activation=False,  # ConvNeXt only uses one activation and normalization per block, pre-activation is meaningless
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
        """Build InvertedBottleneckBlock layers: 7x7 depthwise conv -> 1x1 pointwise conv -> 1x1 pointwise conv"""
        out_channels = self.neck_channels // self.expansion  # Inverted bottleneck reduces channels

        layers = nn.Sequential(
            # 7x7 depthwise conv (reduce, has normalization)
            self.conv_block_cls(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=7,
                stride=1,
                padding="same",
                activation=None,  # No activation after depthwise conv
                norm=self.norm,
                norm_params=self.norm_params,
                dropout_prob=self.conv_drop_prob,
                dropout_size=self.conv_drop_size,
                deep_separable=True,  # Depthwise separable convolution
                **self.conv_kwargs,
            ),
            # 1x1 pointwise conv (expand, has activation)
            self.conv_block_cls(
                in_channels=in_channels,
                out_channels=self.neck_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                activation=self.activation,
                norm=None,  # No normalization after expansion
                activation_params=self.activation_params,
                dropout_prob=self.conv_drop_prob,
                dropout_size=self.conv_drop_size,
                **self.conv_kwargs,
            ),
            # 1x1 conv (expand, no activation, no normalization, no dropout)
            self.conv_block_cls(
                in_channels=self.neck_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                activation=None,
                norm=None,
                dropout_prob=0.0,  # No dropout before residual connection
                **self.conv_kwargs,
            ),
        )
        return layers, out_channels
