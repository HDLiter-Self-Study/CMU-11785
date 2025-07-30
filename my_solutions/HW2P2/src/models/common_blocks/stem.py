import torch
import torch.nn as nn

from typing import Dict, Any
from .convolution_block import ConvolutionBlock


class ResNetStem(nn.Module):
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


class ConvNeXtStem(nn.Module):
    """Patchify stem block for ConvNeXt architectures"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm: str = "layer_norm",
        norm_params: Dict[str, Any] = None,
    ):
        """
        Patchify stem block, see https://arxiv.org/pdf/2201.03545
        """
        super().__init__()

        kernel_size = max(4, stride)  # Ensure kernel size is at least 4, the paper uses 4*4 kernel for stride 4

        # Use one large kernel convolution to patchify the input
        # Add normalization after spatial downsampling
        # For proper downsampling, padding should be (kernel_size - stride) // 2
        padding = (kernel_size - stride) // 2

        self.layers = nn.Sequential(
            ConvolutionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,  # Correct padding for downsampling
                norm=norm,
                norm_params=norm_params,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
