"""
ConvNeXt Block Implementation
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.se_module import SEModule


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with optional SE support"""

    def __init__(
        self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, config: Dict[str, Any] = None
    ):
        super().__init__()

        config = config or {}

        # TODO: ConvNeXt block structure
        self.dwconv = None  # TODO: 7x7 depthwise conv
        self.norm = None  # TODO: LayerNorm
        self.pwconv1 = None  # TODO: 1x1 expand (4x channels)
        self.act = None  # TODO: GELU activation
        self.pwconv2 = None  # TODO: 1x1 contract back to dim

        # TODO: Layer scale parameter
        self.gamma = None  # TODO: learnable scaling

        # TODO: SE module
        self.se = None  # TODO: SEModule or Identity

        # TODO: Drop path
        self.drop_path_prob = None  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x

        # TODO: depthwise conv
        x = None  # TODO

        # TODO: change to channel-last format for norms/linear
        x = None  # TODO: permute to (N, H, W, C)

        # TODO: norm -> expand -> act -> contract
        x = None  # TODO: LayerNorm
        x = None  # TODO: expand channels
        x = None  # TODO: activation
        x = None  # TODO: contract channels

        # TODO: layer scaling
        if self.gamma is not None:
            x = None  # TODO: apply gamma scaling

        # TODO: back to channel-first and apply SE
        x = None  # TODO: permute back to (N, C, H, W)
        x = None  # TODO: SE attention

        # TODO: drop path and residual
        if self.training and self.drop_path_prob > 0:
            x = None  # TODO: apply drop path

        x = None  # TODO: residual connection

        return x
