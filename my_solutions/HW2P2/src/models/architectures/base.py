"""
Base Architecture Abstract Class
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseArchitecture(nn.Module, ABC):
    """Base class for all architectures"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass should return a dictionary with:
        - 'feats': Final feature representation
        - 'all_feats': List of intermediate features
        - 'out': Classification output
        """
        pass

    def _get_activation(self, config: Dict[str, Any]) -> nn.Module:
        """Get activation function based on config"""
        activation = config.get("activation", "relu")
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def _get_norm_layer(self, config: Dict[str, Any]):
        """Get normalization layer based on config"""
        norm = config.get("normalization", "batch_norm")
        if norm == "batch_norm":
            return nn.BatchNorm2d
        elif norm == "group_norm":
            return lambda channels: nn.GroupNorm(32, channels)
        elif norm == "layer_norm":
            return lambda channels: nn.GroupNorm(1, channels)
        else:
            return nn.BatchNorm2d
