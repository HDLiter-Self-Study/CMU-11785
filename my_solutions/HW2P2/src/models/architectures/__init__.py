"""
Neural network architectures
"""

from src.models.architectures.base import BaseArchitecture
from src.models.architectures.resnet.resnet import ResNet
from src.models.architectures.convnext.convnext import ConvNeXt

__all__ = ["BaseArchitecture", "ResNet", "ConvNeXt"]
