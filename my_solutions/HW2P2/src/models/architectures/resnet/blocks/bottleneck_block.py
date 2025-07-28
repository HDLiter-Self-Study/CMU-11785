"""
Bottleneck ResNet Block with optional SE module
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.se_module import SEModule
from ....common_blocks.convolution_block import ConvolutionBlock, PreActivationConvBlock
from ....utils import get_activation


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet block with optional SE module"""

    def __init__(
        self,
        in_channels,
        neck_channels,
        stride=1,
        expansion=4,
        pre_activation=False,  # Arrange layers in pre_activation style, see <https://arxiv.org/abs/1603.05027>
        # Simplified activation and normalization (most common usage)
        activation="relu",
        norm="batch_norm",
        # Advanced configuration (optional)
        activation_params=None,
        norm_params=None,
        conv_dropout=0.0,  # Dropout between conv layers but not after, see <https://arxiv.org/abs/1605.07146>
        projection_type="conv",  # Shortcut projection type: conv, avg_pool, max_pool
        use_se=False,  # Use SE module
        # Conv2d parameters passed directly to all conv layers
        **conv_kwargs,
    ):
        super().__init__()
        # Initialize parameters
        self.use_se = use_se
        self.pre_activation = pre_activation
        out_channels = expansion * neck_channels
        self.final_activation = None
        if not pre_activation:
            self.final_activation = get_activation(activation, **(activation_params or {}))

        if not pre_activation:
            self.layers = nn.Sequential(
                # 1x1 conv (reduce)
                ConvolutionBlock(
                    in_channels,
                    neck_channels,
                    1,
                    1,
                    0,
                    activation,
                    norm,
                    activation_params,
                    norm_params,
                    dropout=conv_dropout,
                    **conv_kwargs,
                ),
                # 3x3 conv (process)
                ConvolutionBlock(
                    neck_channels,
                    neck_channels,
                    3,
                    1,
                    stride,  # Do downsampling on 3*3 conv to avoid info loss see <https://arxiv.org/abs/1812.01187>
                    activation,
                    norm,
                    activation_params,
                    norm_params,
                    dropout=conv_dropout,
                    **conv_kwargs,
                ),
                # 1x1 conv (expand, no activation, no dropout)
                ConvolutionBlock(
                    neck_channels,
                    out_channels,
                    1,
                    1,
                    0,
                    activation="none",
                    norm=norm,
                    norm_params=norm_params,
                    dropout=0.0,  # No dropout before residual connection
                    **conv_kwargs,
                ),
            )
        else:
            self.layers = nn.Sequential(
                # Pre-activation bottleneck
                PreActivationConvBlock(
                    in_channels,
                    neck_channels,
                    1,
                    1,
                    0,
                    activation,
                    norm,
                    activation_params,
                    norm_params,
                    dropout=conv_dropout,
                    **conv_kwargs,
                ),
                PreActivationConvBlock(
                    neck_channels,
                    neck_channels,
                    3,
                    stride,  # Do downsampling on 3*3 conv to avoid info loss see <https://arxiv.org/abs/1812.01187>
                    1,
                    activation,
                    norm,
                    activation_params,
                    norm_params,
                    dropout=conv_dropout,
                    **conv_kwargs,
                ),
                PreActivationConvBlock(
                    neck_channels,
                    out_channels,
                    1,
                    1,
                    0,
                    activation,
                    norm,
                    activation_params,
                    norm_params,
                    dropout=0.0,  # No dropout before residual connection
                    **conv_kwargs,
                ),
            )

        self.shortcut = self._build_shortcut(in_channels, out_channels, stride, projection_type=projection_type)

        if use_se:
            # SE module for final output channels
            self.se = SEModule(out_channels)

        # TODO: Consider residual parameters

    def _build_shortcut(
        self, in_channels: int, out_channels: int, stride: int, projection_type: str = "conv"
    ) -> nn.Module:
        """
        Build shortcut connection based on projection type (conv, avg_pool, max_pool).

        If the input and output channels differ or stride is not 1, applies pooling (avg or max) for downsampling if specified,
        followed by a 1x1 convolution to match dimensions and/or downsample. Otherwise, returns identity mapping.
        """
        if projection_type not in ["conv", "avg_pool", "max_pool"]:
            raise ValueError(
                f"Invalid projection_type: {projection_type}. Must be one of 'conv', 'avg_pool', 'max_pool'."
            )

        if in_channels != out_channels or stride != 1:
            layers = nn.Sequential()
            if stride > 1:
                if projection_type == "avg_pool":
                    # Use avg_pool before downsampling to avoid info lost, see <https://arxiv.org/abs/1812.01187>
                    layers.append(nn.AvgPool2d(stride, stride))
                elif projection_type == "max_pool":
                    # Try max_pool
                    layers.append(nn.MaxPool2d(stride, stride))

            # Use 1x1 convolution to match dimensions and/or downsample
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride if projection_type == "conv" else 1,
                    padding=0,
                    bias=False,  # No bias in conv layers
                )
            )
            return layers
        else:
            # Identity shortcut
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Process through bottleneck layers
        out = self.layers(x)

        # SE attention
        out = self.se(out) if self.use_se else out

        # Residual connection
        shortcut = self.shortcut(residual)
        out = out + shortcut

        # Final activation if not pre-activation style
        out = self.final_activation(out) if self.final_activation else out

        return out
