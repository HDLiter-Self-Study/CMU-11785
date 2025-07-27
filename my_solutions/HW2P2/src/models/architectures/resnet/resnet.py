"""
ResNet Architecture with configurable SE support
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..base import BaseArchitecture
from .blocks import BasicBlock, BottleneckBlock


class ResNet(BaseArchitecture):
    """
    ResNet that can be configured via hyperparameters with optional SE modules
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.config = config
        depth = config.get("resnet_depth", 50)
        width_mult = config.get("width_multiplier", 1.0)
        block_type = config.get("block_type", "bottleneck")
        stem_channels = config.get("stem_channels", 64)

        # Calculate layer configuration based on depth
        if depth == 18:
            layers = [2, 2, 2, 2]
            block_class = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block_class = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block_class = BottleneckBlock
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block_class = BottleneckBlock
        elif depth == 152:
            layers = [3, 8, 36, 3]
            block_class = BottleneckBlock
        else:
            # Custom depth - distribute layers evenly
            total_blocks = max(8, depth // 8)
            layers = [total_blocks // 4] * 4
            block_class = BottleneckBlock if block_type == "bottleneck" else BasicBlock

        # Apply width multiplier
        channels = [int(c * width_mult) for c in [64, 128, 256, 512]]

        # Stem
        self.stem = None  # TODO: 7x7 conv, bn, relu, 3x3 maxpool

        # ResNet layers
        self.layers = None  # TODO: create 4 layers with proper channel progression

        # TODO: implement layer creation loop
        # - Handle stride for downsampling
        # - Calculate proper in_channels for each block
        # - Use block_class with proper arguments

        # Global average pooling and classifier
        self.global_pool = None  # TODO: adaptive average pooling
        self.flatten = None  # TODO: flatten for classifier

        # TODO: calculate final feature dimension after pooling
        final_dim = None  # TODO: depends on block expansion

        # Add dropout if specified
        dropout_rate = config.get("dropout_rate", 0.0)
        if dropout_rate > 0:
            self.dropout = None  # TODO
        else:
            self.dropout = None  # TODO

        self.classifier = None  # TODO: linear layer with final_dim input

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Stem processing
        x = None  # TODO: pass through stem

        # Feature extraction through layers
        feats = []
        # TODO: iterate through self.layers
        # TODO: collect intermediate features in feats list

        # Classification head
        x = None  # TODO: global pooling
        x = None  # TODO: flatten
        features = x  # Save features for verification task

        x = None  # TODO: apply dropout
        x = None  # TODO: classifier

        return {"feats": features, "all_feats": feats, "out": x}
