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
        self.stem = None  # TODO

        # ResNet layers
        self.layers = nn.ModuleList()
        in_channels = stem_channels

        for i, num_blocks in enumerate(layers):
            out_channels = channels[i]
            stride = 1 if i == 0 else 2

            layer_blocks = []
            for j in range(num_blocks):
                block_in_channels = in_channels if j == 0 else out_channels * block_class.expansion
                block = None  # TODO
                layer_blocks.append(block)

            self.layers.append(nn.Sequential(*layer_blocks))
            in_channels = out_channels * block_class.expansion

        # Global average pooling and classifier
        self.global_pool = None  # TODO
        self.flatten = None  # TODO

        # Add dropout if specified
        dropout_rate = config.get("dropout_rate", 0.0)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.classifier = None  # TODO

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Stem
        x = None  # TODO

        # Feature extraction
        feats = []
        for layer in self.layers:
            x = None  # TODO
            feats.append(x)

        # Classification head
        x = None  # TODO
        x = None  # TODO
        features = x  # Save features for verification task

        x = self.dropout(x)
        x = None  # TODO

        return {"feats": features, "all_feats": feats, "out": x}
