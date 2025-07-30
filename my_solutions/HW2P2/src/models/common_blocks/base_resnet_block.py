"""
Base ResNet Block with common functionality
"""

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from abc import ABC, abstractmethod
from typing import Dict, Any
from .se_module import SEModule
from .convolution_block import ConvolutionBlock, PreActivationConvBlock
from ..utils import get_activation


class BaseResNetBlock(nn.Module, ABC):
    """Base class for ResNet blocks with common functionality"""

    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        pre_activation: bool = False,  # Arrange layers in pre_activation style, see <https://arxiv.org/abs/1603.05027>
        activation: str = "relu",
        norm: str = "batch_norm",
        # Advanced configuration (optional)
        activation_params: Dict[str, Any] = None,
        norm_params: Dict[str, Any] = None,
        conv_drop_prob: float = 0.0,  # Dropout between conv layers, see wide resnet <https://arxiv.org/abs/1605.07146>
        conv_drop_size: int = 1,  # Size of dropout block (1 for standard dropout), DropBlock: <https://arxiv.org/abs/1810.12890>
        projection_type: str = "conv",  # Shortcut projection type: conv, avg_pool, max_pool
        use_se: bool = False,  # Use SE module
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-6,
        stochastic_depth_prob: float = 0.0,  # Probability for stochastic depth
        # Conv2d parameters passed directly to all conv layers
        **conv_kwargs,
    ):
        super().__init__()

        # Store common parameters
        self.use_se = use_se
        self.pre_activation = pre_activation
        self.stride = stride
        self.activation = activation
        self.norm = norm
        self.activation_params = activation_params or {}
        self.norm_params = norm_params or {}
        self.conv_drop_prob = conv_drop_prob
        self.conv_drop_size = conv_drop_size
        self.conv_kwargs = conv_kwargs
        self.layer_scale = layer_scale
        self.stochastic_depth_prob = stochastic_depth_prob

        # Determine block type and final activation
        self.conv_block_cls = ConvolutionBlock if not pre_activation else PreActivationConvBlock
        self.final_activation = get_activation(activation, **self.activation_params) if not pre_activation else None

        # Build layers (implemented by subclasses)
        self.layers, out_channels = self._build_layers(in_channels)

        # Build shortcut connection
        self.shortcut = self._build_shortcut(in_channels, out_channels, stride, projection_type)

        # SE module (conditional)
        if use_se:
            self.se = SEModule(out_channels)

        # Layer scale (conditional)
        if self.layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones(1, out_channels, 1, 1), requires_grad=True
            )

    @abstractmethod
    def _build_layers(self, in_channels: int) -> tuple[nn.Module, int]:
        """
        Build the main computation layers for this block type.

        Returns:
            tuple: (layers_module, output_channels)
        """
        pass

    def _build_shortcut(
        self, in_channels: int, out_channels: int, stride: int, projection_type: str = "conv"
    ) -> nn.Module:
        """Build shortcut connection based on projection type"""
        if projection_type not in ["conv", "avg_pool", "max_pool"]:
            raise ValueError(
                f"Invalid projection_type: {projection_type}. Must be one of 'conv', 'avg_pool', 'max_pool'."
            )

        if in_channels != out_channels or stride != 1:
            layers = nn.Sequential()
            if stride > 1:
                if projection_type == "avg_pool":
                    # Use avg_pool before downsampling to avoid info lost
                    layers.append(nn.AvgPool2d(stride, stride))
                elif projection_type == "max_pool":
                    # Use max_pool
                    layers.append(nn.MaxPool2d(stride, stride))

            # Use 1x1 convolution to match dimensions and/or downsample
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride if projection_type == "conv" else 1,
                    padding=0,
                    bias=False,
                )
            )
            return layers
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Common forward pass logic"""
        residual = x

        # Process through main layers
        out = self.layers(x)

        # SE attention (if enabled)
        out = self.se(out) if self.use_se else out
        # Layer scale (if enabled)
        if self.layer_scale:
            out = out * self.layer_scale
        # Stochastic depth (if enabled)
        if self.stochastic_depth_prob > 0:
            out = StochasticDepth(self.stochastic_depth_prob)(out)
        # Residual connection
        shortcut = self.shortcut(residual)

        out = out + shortcut

        # Final activation (if not pre-activation style)
        out = self.final_activation(out) if self.final_activation else out

        return out
