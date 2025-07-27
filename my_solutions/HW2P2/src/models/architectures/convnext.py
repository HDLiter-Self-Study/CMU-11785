"""
Dynamic ConvNeXt Architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base import BaseArchitecture
from ..common_blocks.convolution_block import ConvolutionBlock


class ConvNeXt(BaseArchitecture):
    """ConvNeXt implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Simplified ConvNeXt implementation
        # In practice, you'd implement the full ConvNeXt architecture
        variant = config.get("convnext_variant", "tiny")

        if variant == "tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        elif variant == "small":
            depths = [3, 3, 27, 3]
            dims = [96, 192, 384, 768]
        else:  # base
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]

        # Simplified implementation - in practice you'd implement full ConvNeXt blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            # LayerNorm for 2D features needs special handling
            nn.GroupNorm(1, dims[0]),  # Equivalent to LayerNorm for Conv2d
        )

        # For simplicity, using modified ResNet-like structure
        # In practice, implement proper ConvNeXt blocks
        self.stages = nn.ModuleList()

        # Create stages with proper downsampling
        current_dim = dims[0]
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            # First layer of each stage (except first) does downsampling
            if i > 0:
                # Downsample layer
                downsample = nn.Sequential(nn.Conv2d(current_dim, dim, kernel_size=2, stride=2), nn.GroupNorm(1, dim))
                stage_layers = [downsample]
                current_dim = dim
            else:
                stage_layers = []

            # Add depth-1 ConvolutionBlocks (since first may be downsample)
            remaining_depth = depth - 1 if i > 0 else depth
            stage_layers.extend([ConvolutionBlock(dim, dim, 3, 1, 1) for _ in range(remaining_depth)])

            self.stages.append(nn.Sequential(*stage_layers))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(dims[-1], config.get("num_classes", 1000))

    def forward(self, x):
        x = self.stem(x)

        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        features = x

        x = self.classifier(x)

        return {"feats": features, "all_feats": feats, "out": x}
