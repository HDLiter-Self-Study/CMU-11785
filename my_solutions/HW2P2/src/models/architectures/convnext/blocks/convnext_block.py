"""
ConvNeXt Block Implementation
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ....common_blocks.attention import SEModule


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with optional SE support"""

    def __init__(
        self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6, config: Dict[str, Any] = None
    ):
        super().__init__()

        config = config or {}

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer scale parameter
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # SE module if specified
        if config.get("use_se", False):
            self.se = SEModule.from_config(dim, config)
        else:
            self.se = nn.Identity()

        # Drop path for stochastic depth
        self.drop_path_prob = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Apply SE attention if enabled
        x = self.se(x)

        # Apply drop path if training
        if self.training and self.drop_path_prob > 0:
            # Simple drop path implementation
            keep_prob = 1 - self.drop_path_prob
            if torch.rand(1) > keep_prob:
                return input_tensor

        x = input_tensor + x
        return x
