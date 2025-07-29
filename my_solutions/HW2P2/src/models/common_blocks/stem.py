import torch
import torch.nn as nn

from typing import Dict, Any
from convolution_block import ConvolutionBlock


class Stem(nn.Module):
    """Stem block for ResNet architectures"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        activation: str = "relu",
        norm: str = "batch_norm",
        activation_params: Dict[str, Any] = None,
        norm_params: Dict[str, Any] = None,
        use_maxpool: bool = True,
        # conv_dropout: float = 0.0, # Typically not used in stem
    ):
        """
        Use three 3x3 conv layers to replace the original 7x7 conv layer for better performance.
        See bags of tricks <https://arxiv.org/abs/1812.01187> for details.
        """
        super().__init__()
        if use_maxpool:
            # Divide the stride between the first conv and pooling layer
            if stride == 1:
                conv_stride = 1
                max_pool_stride = 0  # No pooling
            elif stride == 2:
                conv_stride = 1
                max_pool_stride = 2
            elif stride == 4:
                conv_stride = 2
                max_pool_stride = 2
            else:
                raise ValueError("Stride must be 1, 2, or 4 for the stem block.")
        else:
            conv_stride = stride
            max_pool_stride = 0

        # Use three 3x3 conv layers to replace the original 7x7 conv layer
        self.layers = nn.Sequential(
            ConvolutionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=conv_stride,
                padding=1,
                activation=activation,
                norm=norm,
                activation_params=activation_params,
                norm_params=norm_params,
            ),
            ConvolutionBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=activation,
                norm=norm,
                activation_params=activation_params,
                norm_params=norm_params,
            ),
            ConvolutionBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=activation,
                norm=norm,
                activation_params=activation_params,
                norm_params=norm_params,
            ),
        )
        if max_pool_stride > 0:
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=max_pool_stride, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
