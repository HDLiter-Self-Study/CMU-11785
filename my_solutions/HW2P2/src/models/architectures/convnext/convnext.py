"""
ConvNeXt Architecture with optional SE support
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..base import BaseArchitecture
from .blocks import ConvNeXtBlock


class ConvNeXt(BaseArchitecture):
    """ConvNeXt implementation with configurable SE support"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        variant = config.get("convnext_variant", "tiny")
        drop_path_rate = config.get("drop_path_rate", 0.0)

        if variant == "tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        elif variant == "small":
            depths = [3, 3, 27, 3]
            dims = [96, 192, 384, 768]
        elif variant == "base":
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]
        elif variant == "large":
            depths = [3, 3, 27, 3]
            dims = [192, 384, 768, 1536]
        else:
            # Default to tiny
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]

        # Stem
        self.stem = None  # TODO

        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        for i in range(4):
            if i == 0:
                # First downsample is handled by stem
                downsample_layer = nn.Identity()
            else:
                downsample_layer = nn.Sequential(
                    nn.GroupNorm(1, dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                )
            self.downsample_layers.append(downsample_layer)

        # ConvNeXt stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(None)  # TODO
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]

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
        x = None  # TODO

        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = None  # TODO
            feats.append(x)

        x = None  # TODO
        x = None  # TODO
        features = x

        x = self.dropout(x)
        x = None  # TODO

        return {"feats": features, "all_feats": feats, "out": x}
