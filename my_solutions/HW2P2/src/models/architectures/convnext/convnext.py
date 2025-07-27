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
        self.stem = None  # TODO: 4x4 conv with stride 4 + layernorm

        # TODO: create downsample layers
        self.downsample_layers = None  # TODO: 4 downsample layers

        # TODO: create stage blocks
        self.stages = None  # TODO: 4 stages with ConvNeXtBlocks

        # TODO: final norm and pooling
        self.norm = None  # TODO: final layer norm
        self.avgpool = None  # TODO: adaptive average pooling
        self.classifier = None  # TODO: linear classifier

        # TODO: dropout
        dropout_rate = config.get("dropout_rate", 0.0)
        self.dropout = None  # TODO

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # TODO: stem processing
        x = None  # TODO

        # TODO: process through 4 stages with downsampling
        feats = []
        # TODO: implement stage processing loop

        # TODO: final processing
        x = None  # TODO: final norm
        x = None  # TODO: global pooling
        x = None  # TODO: flatten
        features = x

        x = None  # TODO: dropout
        x = None  # TODO: classifier

        return {"feats": features, "all_feats": feats, "out": x}
        x = None  # TODO

        return {"feats": features, "all_feats": feats, "out": x}
