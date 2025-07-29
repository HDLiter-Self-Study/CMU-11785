"""
ResNet Architecture with configurable SE support
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .blocks import BasicBlock, BottleneckBlock
from ...common_blocks.head import ClassificationHead
from ...common_blocks.stem import Stem


class ResNet:
    """
    ResNet that can be configured via hyperparameters with optional SE modules
    """

    def __init__(
        self,
        in_channels: int,
        stages: List[int],
        out_channels: List[int],
        downsamplings: List[int],
        block_types: List[str],
        block_params: List[Dict[str, Any]],
        head_type: str = "classification",
        head_params: Dict[str, Any] = None,
        stem_channels: int = 64,
        stem_params: Dict[str, Any] = None,
        width_multiplier: float = 1.0,
    ):
        """
        Initialize the ResNet architecture.

        Args:
            stages: List of integers representing the number of blocks in each stage.
            channels: List of integers representing the number of output channels for each stage.
            downsamplings: List of integers representing the downsampling factor for each stage.
            block_types: List of strings representing the type of block in each stage.
            block_params: List of dictionaries containing parameters for each block type.
            head_type: Type of classification head to use (default: "classification").
            head_params: Parameters for the classification head.
            stem_channels: Number of output channels for the stem convolution (default: 64).
            width_multiplier: Multiplier for the number of channels in each block (default: 1.0).
        """

        # Apply width multiplier
        out_channels = [int(c * width_multiplier) for c in out_channels]

        # Stem
        self.stem = self._create_stem(in_channels, stem_channels, stem_params)

        # ResNet backbone
        self.backbone = self._create_backbone(
            stem_channels,
            stages,
            out_channels,
            block_types,
            block_params,
            downsamplings,
        )

        # Get head layer
        self.head = self._create_head(head_type, head_params, out_channels[-1])

    def _create_stem(self, in_channels, out_channels: int, stem_params: Dict[str, Any] = None) -> nn.Module:
        """
        Create the stem convolution layer
        """
        return Stem(in_channels, out_channels, **(stem_params or {}))

    def _create_head(self, head_type: str, head_params: Dict[str, Any], in_features: int) -> nn.Module:
        """Create the classification head based on type and parameters"""
        if head_type == "classification":
            return ClassificationHead(in_features, **head_params)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def _create_backbone(
        self,
        stem_channels: int,
        stages: List[int],
        out_channels: List[int],
        block_types: List[str],
        block_params: List[Dict[str, Any]],
        downsamplings: List[int] = None,
    ) -> nn.Module:
        """
        Create the ResNet backbone consisting of multiple stages.
        """
        backbone = nn.Sequential()
        for i, (num_blocks, out_ch, block_type, downsample) in enumerate(
            zip(stages, out_channels, block_types, downsamplings)
        ):
            if block_type == "basic":
                block_cls = BasicBlock
            elif block_type == "bottleneck":
                block_cls = BottleneckBlock
            else:
                raise ValueError(f"Unsupported block type: {block_type}")

            for j in range(num_blocks):
                in_ch = out_ch if j > 0 else (out_channels[i - 1] if i > 0 else stem_channels)
                stride = downsample if j == 0 else 1
                block = block_cls(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    **block_params[i],
                )
                backbone.append(block)
        return backbone

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Stem processing
        x = self.stem(x)

        # Process through backbone
        x = self.backbone(x)
        feats = x  # Save features for verification task

        # Classification head
        x = self.head(x)
        # Return features and output
        return {"feats": feats, "out": x}
