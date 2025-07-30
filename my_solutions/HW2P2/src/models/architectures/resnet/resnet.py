"""
ResNet Architecture with configurable SE support
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .blocks import BasicBlock, BottleneckBlock
from ...common_blocks.head import ClassificationHead
from ...common_blocks.stem import ResNetStem
from ..base import BaseArchitecture


class ResNet(BaseArchitecture):
    """
    ResNet that can be configured via hyperparameters
    """

    # Register ResNet-specific blocks
    BLOCK_REGISTRY = BaseArchitecture.BLOCK_REGISTRY.copy()
    BLOCK_REGISTRY.update(
        {
            "basic": BasicBlock,
            "bottleneck": BottleneckBlock,
        }
    )

    def _create_stem(self, in_channels: int, out_channels: int, stem_params: Dict[str, Any] = None) -> nn.Module:
        """
        Create the stem convolution layer
        """
        return ResNetStem(in_channels, out_channels, **(stem_params or {}))

    def _handle_downsampling(
        self, in_channels: int, out_channels: int, downsample: int, block_params: Dict[str, Any]
    ) -> List[nn.Module]:
        """
        ResNet handles downsampling within blocks via stride, so no separate modules needed.
        The first block in each stage will use stride=downsample to perform downsampling.
        """
        return []  # ResNet uses stride-based downsampling within blocks
