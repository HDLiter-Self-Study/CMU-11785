"""
Common blocks that can be used across different architectures
"""

from .attention import SEModule
from .convolution_block import ConvolutionBlock

__all__ = ["SEModule", "ConvolutionBlock"]
