"""
ResNet blocks with optional SE module support
"""

from src.models.architectures.resnet.blocks.basic_block import BasicBlock
from src.models.architectures.resnet.blocks.bottleneck_block import BottleneckBlock

__all__ = ["BasicBlock", "BottleneckBlock"]
