"""
ConvNeXt Architecture with optional SE support
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..base import BaseArchitecture
from .blocks import InvertedBottleneckBlock
from ...common_blocks.head import ClassificationHead
from ...common_blocks.stem import ConvNeXtStem
from ...common_blocks.convolution_block import PreActivationConvBlock


class ConvNeXt(BaseArchitecture):
    """
    ConvNeXt that can be configured via hyperparameters
    """

    # Register ConvNeXt-specific blocks
    BLOCK_REGISTRY = BaseArchitecture.BLOCK_REGISTRY.copy()
    BLOCK_REGISTRY.update(
        {
            "inverted_bottleneck": InvertedBottleneckBlock,
        }
    )

    def _create_stem(self, in_channels: int, out_channels: int, stem_params: Dict[str, Any] = None) -> nn.Module:
        """
        Create the stem convolution layer
        """
        return ConvNeXtStem(in_channels, out_channels, **(stem_params or {}))

    def _handle_downsampling(
        self, in_channels: int, out_channels: int, downsample: int, block_params: Dict[str, Any]
    ) -> List[nn.Module]:
        """
        ConvNeXt uses separate downsampling layers between stages
        """
        if downsample > 1:
            downsample_layer = self._get_downsample_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=downsample,
                norm=block_params.get("norm", "layer_norm"),
                norm_params=block_params.get("norm_params", {}),
            )
            return [downsample_layer]
        return []

    def _get_downsample_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm: str = "layer_norm",
        norm_params: Dict[str, Any] = None,
    ) -> nn.Module:
        """
        Create a downsample layer for ConvNeXt.
        Normalization layer followed by a non-overlapping convolution.
        """
        return PreActivationConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,  # No padding for downsampling
            bias=False,
            norm=norm,
            norm_params=norm_params,
        )
