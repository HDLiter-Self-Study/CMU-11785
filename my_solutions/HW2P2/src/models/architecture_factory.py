"""
Architecture Factory for dynamic model creation based on hyperparameters
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .network import ConvolutionBlock


class ArchitectureFactory:
    """
    Factory class for creating different neural network architectures
    """

    def __init__(self):
        self.builders = {
            "resnet": self._build_resnet,
            "senet": self._build_senet,
            "convnext": self._build_convnext,
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
        """Build ResNet architecture"""
        return DynamicResNet(config)

    def _build_senet(self, config: Dict[str, Any]) -> nn.Module:
        """Build SE-Net architecture"""
        return DynamicSENet(config)

    def _build_convnext(self, config: Dict[str, Any]) -> nn.Module:
        """Build ConvNeXt architecture"""
        return DynamicConvNeXt(config)


class DynamicResNet(nn.Module):
    """
    Dynamic ResNet that can be configured via hyperparameters
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        depth = config.get("resnet_depth", 50)
        width_mult = config.get("width_multiplier", 1.0)
        block_type = config.get("block_type", "bottleneck")
        stem_channels = config.get("stem_channels", 64)

        # Calculate layer configuration based on depth
        if depth == 18:
            layers = [2, 2, 2, 2]
            block_class = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block_class = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block_class = BottleneckBlock
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block_class = BottleneckBlock
        elif depth == 152:
            layers = [3, 8, 36, 3]
            block_class = BottleneckBlock
        else:
            # Custom depth - distribute layers evenly
            total_blocks = max(8, depth // 8)
            layers = [total_blocks // 4] * 4
            block_class = BottleneckBlock if block_type == "bottleneck" else BasicBlock

        # Apply width multiplier
        channels = [int(c * width_mult) for c in [64, 128, 256, 512]]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            self._get_norm_layer(config)(stem_channels),
            self._get_activation(config),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet layers
        self.layers = nn.ModuleList()
        in_channels = stem_channels

        for i, num_blocks in enumerate(layers):
            out_channels = channels[i]
            stride = 1 if i == 0 else 2

            layer_blocks = []
            for j in range(num_blocks):
                block = block_class(
                    in_channels if j == 0 else out_channels, out_channels, stride if j == 0 else 1, config
                )
                layer_blocks.append(block)

            self.layers.append(nn.Sequential(*layer_blocks))
            in_channels = out_channels

        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Add dropout if specified
        dropout_rate = config.get("dropout_rate", 0.0)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(channels[-1], config.get("num_classes", 1000))

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

    def forward(self, x):
        """Forward pass"""
        # Stem
        x = self.stem(x)

        # Feature extraction
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)

        # Classification head
        x = self.global_pool(x)
        x = self.flatten(x)
        features = x  # Save features for verification task

        x = self.dropout(x)
        x = self.classifier(x)

        return {"feats": features, "all_feats": feats, "out": x}


class BasicBlock(nn.Module):
    """Basic ResNet block"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, config):
        super().__init__()

        norm_layer = self._get_norm_layer(config)
        activation = self._get_activation(config)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.activation = activation

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), norm_layer(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # SE module if specified
        if config.get("use_se", False):
            self.se = SEModule(out_channels, reduction=16)
        else:
            self.se = nn.Identity()

    def _get_activation(self, config):
        activation = config.get("activation", "relu")
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU(inplace=True)

    def _get_norm_layer(self, config):
        norm = config.get("normalization", "batch_norm")
        if norm == "batch_norm":
            return nn.BatchNorm2d
        elif norm == "group_norm":
            return lambda channels: nn.GroupNorm(32, channels)
        else:
            return nn.BatchNorm2d

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += self.shortcut(residual)
        out = self.activation(out)

        return out


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet block"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, config):
        super().__init__()

        norm_layer = self._get_norm_layer(config)
        activation = self._get_activation(config)

        # Bottleneck design
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)

        self.activation = activation

        # Shortcut connection
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                norm_layer(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

        # SE module if specified
        if config.get("use_se", False):
            self.se = SEModule(out_channels * self.expansion, reduction=16)
        else:
            self.se = nn.Identity()

    def _get_activation(self, config):
        activation = config.get("activation", "relu")
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU(inplace=True)

    def _get_norm_layer(self, config):
        norm = config.get("normalization", "batch_norm")
        if norm == "batch_norm":
            return nn.BatchNorm2d
        elif norm == "group_norm":
            return lambda channels: nn.GroupNorm(32, channels)
        else:
            return nn.BatchNorm2d

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        out += self.shortcut(residual)
        out = self.activation(out)

        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DynamicSENet(nn.Module):
    """Dynamic SE-Net implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Implementation for SE-Net based on config
        # This would be similar to DynamicResNet but with SE modules integrated
        # For brevity, using a simplified version
        base_model = DynamicResNet({**config, "use_se": True})
        self.backbone = base_model.stem
        self.layers = base_model.layers
        self.global_pool = base_model.global_pool
        self.flatten = base_model.flatten
        self.dropout = base_model.dropout
        self.classifier = base_model.classifier

    def forward(self, x):
        x = self.backbone(x)

        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        features = x

        x = self.dropout(x)
        x = self.classifier(x)

        return {"feats": features, "all_feats": feats, "out": x}


class DynamicConvNeXt(nn.Module):
    """Dynamic ConvNeXt implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Simplified ConvNeXt implementation
        # In practice, you'd implement the full ConvNeXt architecture
        variant = config.get("convnext_variant", "tiny")

        if variant == "tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        elif variant == "small":
            depths = [3, 3, 27, 3]
            dims = [96, 192, 384, 768]
        else:  # base
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]

        # Simplified implementation - in practice you'd implement full ConvNeXt blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )

        # For simplicity, using modified ResNet-like structure
        # In practice, implement proper ConvNeXt blocks
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[ConvolutionBlock(dims[0], dims[0], 3, 1, 1) for _ in range(depths[0])]),
                nn.Sequential(*[ConvolutionBlock(dims[1], dims[1], 3, 1, 1) for _ in range(depths[1])]),
                nn.Sequential(*[ConvolutionBlock(dims[2], dims[2], 3, 1, 1) for _ in range(depths[2])]),
                nn.Sequential(*[ConvolutionBlock(dims[3], dims[3], 3, 1, 1) for _ in range(depths[3])]),
            ]
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(dims[-1], config.get("num_classes", 1000))

    def forward(self, x):
        x = self.stem(x)

        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        features = x

        x = self.classifier(x)

        return {"feats": features, "all_feats": feats, "out": x}
