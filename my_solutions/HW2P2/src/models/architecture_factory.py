"""
Architecture Factory for dynamic model creation based on hyperparameters
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .architectures import ResNet, ConvNeXt


class ArchitectureFactory:
    """
    Factory class for creating different neural network architectures
    """

    def __init__(self):
        self.builders = {
            "resnet": self._build_resnet,
            "convnext": self._build_convnext,
            # Note: SENet is now handled as ResNet with use_se=True
        }

    def create_model(self, config: Dict[str, Any]) -> nn.Module:
        """
        Create model based on configuration
        """
        arch_type = config["architecture"]
        if arch_type not in self.builders:
            raise ValueError(f"Unknown architecture: {arch_type}")

        return self.builders[arch_type](config)

    def _build_resnet(self, config: Dict[str, Any]) -> nn.Module:
        """Build ResNet architecture (with optional SE support)"""
        return ResNet(config)

    def _build_convnext(self, config: Dict[str, Any]) -> nn.Module:
        """Build ConvNeXt architecture"""
        return ConvNeXt(config)
