"""
Neural network architectures
"""

from .base import BaseArchitecture
from .resnet import ResNet
from .convnext import ConvNeXt

__all__ = ["BaseArchitecture", "ResNet", "ConvNeXt"]
