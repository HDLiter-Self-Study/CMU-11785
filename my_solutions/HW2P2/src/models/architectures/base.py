"""
Base architecture class for all neural network models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseArchitecture(nn.Module, ABC):
    """
    Abstract base class for all neural network architectures

    Provides common functionality and interface for different architectures
    like ResNet, ConvNeXt, etc.
    """

    def __init__(self, num_classes: int = 1000, in_channels: int = 3, **kwargs):
        """
        Initialize base architecture

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            **kwargs: Additional architecture-specific parameters
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass

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
