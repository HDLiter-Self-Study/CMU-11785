"""
Base architecture class for all neural network models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Callable
from ..common_blocks.head import ClassificationHead


class BaseArchitecture(nn.Module, ABC):
    """
    Abstract base class for all neural network architectures

    Provides common functionality and interface for different architectures
    like ResNet, ConvNeXt, etc.
    """

    # Registry for block types - can be extended by subclasses
    BLOCK_REGISTRY: Dict[str, Type[nn.Module]] = {}

    # Registry for head types
    HEAD_REGISTRY: Dict[str, Type[nn.Module]] = {
        "classification": ClassificationHead,
    }

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
        **kwargs,
    ):
        """
        Initialize base architecture

        Args:
            in_channels: Number of input channels
            stages: List of integers representing the number of blocks in each stage
            out_channels: List of integers representing the number of output channels for each stage
            downsamplings: List of integers representing the downsampling factor for each stage
            block_types: List of strings representing the type of block in each stage
            block_params: List of dictionaries containing parameters for each block type
            head_type: Type of classification head to use (default: "classification")
            head_params: Parameters for the classification head
            stem_channels: Number of output channels for the stem convolution (default: 64)
            stem_params: Parameters for the stem
            width_multiplier: Multiplier for the number of channels in each block (default: 1.0)
            **kwargs: Additional architecture-specific parameters
        """
        super().__init__()

        # Validate inputs
        self._validate_architecture_params(stages, out_channels, downsamplings, block_types, block_params)

        # Store parameters
        self.in_channels = in_channels
        self.stages = stages
        self.downsamplings = downsamplings
        self.block_types = block_types
        self.block_params = block_params
        self.stem_channels = stem_channels
        self.width_multiplier = width_multiplier

        # Process head parameters
        head_params = head_params or {}
        try:
            self.num_classes = head_params["num_classes"]
        except KeyError as e:
            raise ValueError(f"Missing required head parameter: {e}")

        # Apply width multiplier
        self.out_channels = [int(c * width_multiplier) for c in out_channels]

        # Create architecture components
        self.stem = self._create_stem(in_channels, stem_channels, stem_params)
        self.backbone = self._create_backbone(
            stem_channels, stages, self.out_channels, block_types, block_params, downsamplings
        )
        self.head = self._create_head(head_type, head_params, self.out_channels[-1])

    def _validate_architecture_params(
        self,
        stages: List[int],
        out_channels: List[int],
        downsamplings: List[int],
        block_types: List[str],
        block_params: List[Dict[str, Any]],
    ):
        """Validate that all architecture parameters have consistent lengths"""
        lengths = [len(stages), len(out_channels), len(downsamplings), len(block_types), len(block_params)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(
                f"All architecture parameters must have the same length. "
                f"Got lengths: stages={len(stages)}, out_channels={len(out_channels)}, "
                f"downsamplings={len(downsamplings)}, block_types={len(block_types)}, "
                f"block_params={len(block_params)}"
            )

        # Validate block types are registered
        for block_type in block_types:
            if block_type not in self.BLOCK_REGISTRY:
                raise ValueError(
                    f"Block type '{block_type}' not found in registry. "
                    f"Available types: {list(self.BLOCK_REGISTRY.keys())}"
                )

    @classmethod
    def register_block(cls, name: str, block_class: Type[nn.Module]):
        """Register a new block type"""
        cls.BLOCK_REGISTRY[name] = block_class

    @classmethod
    def register_head(cls, name: str, head_class: Type[nn.Module]):
        """Register a new head type"""
        cls.HEAD_REGISTRY[name] = head_class

    def _create_head(self, head_type: str, head_params: Dict[str, Any], in_features: int) -> nn.Module:
        """Create the classification head based on type and parameters"""
        if head_type not in self.HEAD_REGISTRY:
            raise ValueError(f"Unsupported head type: {head_type}. Available: {list(self.HEAD_REGISTRY.keys())}")

        head_class = self.HEAD_REGISTRY[head_type]
        return head_class(in_features, **head_params)

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
        Create the backbone consisting of multiple stages.
        This method provides common logic while allowing architecture-specific customization.
        """
        backbone = []

        for i, (num_blocks, out_ch, block_type, downsample) in enumerate(
            zip(stages, out_channels, block_types, downsamplings)
        ):
            # Get block class from registry
            block_class = self.BLOCK_REGISTRY[block_type]

            # Create blocks for this stage
            stage_blocks = self._create_stage_blocks(
                stage_idx=i,
                num_blocks=num_blocks,
                block_class=block_class,
                out_channels=out_ch,
                downsample=downsample,
                block_params=block_params[i],
                prev_channels=out_channels[i - 1] if i > 0 else stem_channels,
            )

            # Add stage blocks to backbone
            backbone.extend(stage_blocks)

        return nn.Sequential(*backbone)

    def _create_stage_blocks(
        self,
        stage_idx: int,
        num_blocks: int,
        block_class: Type[nn.Module],
        out_channels: int,
        downsample: int,
        block_params: Dict[str, Any],
        prev_channels: int,
    ) -> List[nn.Module]:
        """
        Create blocks for a single stage. This method can be overridden
        by subclasses for architecture-specific logic.
        """
        blocks = []

        for j in range(num_blocks):
            in_ch = out_channels if j > 0 else prev_channels
            stride = downsample if j == 0 else 1

            # Architecture-specific logic for handling downsampling
            if j == 0 and downsample > 1:
                downsample_blocks = self._handle_downsampling(
                    in_channels=in_ch, out_channels=out_channels, downsample=downsample, block_params=block_params
                )
                blocks.extend(downsample_blocks)

                # Update in_channels and reset stride if downsampling blocks were added
                if downsample_blocks:
                    in_ch = out_channels  # Update in_channels after downsampling
                    stride = 1  # Reset stride since downsampling is handled separately

            # Create the main block
            block = block_class(
                in_channels=in_ch,
                out_channels=out_channels,
                stride=stride,
                **block_params,
            )
            blocks.append(block)

        return blocks

    def _handle_downsampling(
        self, in_channels: int, out_channels: int, downsample: int, block_params: Dict[str, Any]
    ) -> List[nn.Module]:
        """
        Handle downsampling logic. This method should be overridden by subclasses
        to implement architecture-specific downsampling strategies.

        Two common strategies:
        1. ResNet-style: Return empty list, downsampling handled by stride in first block
        2. ConvNeXt-style: Return downsampling layer(s), first block uses stride=1

        Returns:
            List of modules to handle downsampling (can be empty for architectures
            that handle downsampling within blocks)
        """
        return []  # Default: no separate downsampling modules (ResNet-style)

    @abstractmethod
    def _create_stem(self, in_channels: int, out_channels: int, stem_params: Dict[str, Any] = None) -> nn.Module:
        """
        Create the stem convolution layer. Must be implemented by subclasses
        as different architectures use different stem designs.
        """
        pass

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network. Provides standard implementation
        that can be overridden if needed.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Dictionary containing features and output:
            - "feats": Feature tensor before classification head
            - "out": Final output tensor of shape (batch_size, num_classes)
        """
        # Stem processing
        x = self.stem(x)

        # Process through backbone
        x = self.backbone(x)
        feats = x  # Save features for verification task

        # Classification head
        x = self.head(x)

        # Return features and output
        return {"feats": feats, "out": x}

    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """
        Freeze all parameters except the final classifier layer
        Useful for transfer learning
        """
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze classifier parameters (assuming it's named 'head' or 'classifier')
        if hasattr(self, "head"):
            for param in self.head.parameters():
                param.requires_grad = True
        elif hasattr(self, "classifier"):
            for param in self.classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreeze all parameters in the model
        """
        for param in self.parameters():
            param.requires_grad = True

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture

        Returns:
            Dictionary containing model information
        """
        return {
            "architecture": self.__class__.__name__,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "total_parameters": self.get_num_parameters(),
            "trainable_parameters": self.get_num_trainable_parameters(),
        }
