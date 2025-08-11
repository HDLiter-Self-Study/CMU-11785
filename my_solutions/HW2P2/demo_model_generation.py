#!/usr/bin/env python3
"""
Demo script for generating real models from sampled hyperparameters
"""

import json
import torch
import torch.nn as nn
from typing import Dict, Any


class SimpleRegNetBuilder:
    """
    Simplified RegNet builder for demonstration purposes
    This creates basic models without requiring the full architecture framework
    """

    def __init__(self, num_classes: int = 1000, input_channels: int = 3):
        self.num_classes = num_classes
        self.input_channels = input_channels

    def build_model_from_sample(self, sample_params: Dict[str, Any]) -> nn.Module:
        """Build a simple model from sampled parameters"""

        # Extract architecture parameters
        if "hierarchical" in sample_params:
            arch_params = sample_params["hierarchical"]["architectures"]
            arch_type = arch_params["architecture_type"]
        else:
            arch_params = sample_params.get("architectures", sample_params)
            arch_type = arch_params["architecture_type"]

        # Extract RegNet parameters
        regnet_params = arch_params.get("regnet_rule", {})
        if not regnet_params:
            raise ValueError("RegNet rule parameters not found")

        # Generate stage configuration
        stage_config = self._generate_regnet_stages(regnet_params)

        # Build simple model
        if arch_type == "resnet":
            return self._build_simple_resnet(stage_config, arch_params)
        elif arch_type == "convnext":
            return self._build_simple_convnext(stage_config, arch_params)
        else:
            raise ValueError(f"Unsupported architecture: {arch_type}")

    def _generate_regnet_stages(self, regnet_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stage configuration using RegNet rules"""

        num_stages = regnet_params["num_stages"]
        width_slope = regnet_params["width_slope"]
        initial_width = regnet_params["initial_width"]
        depth_slope = regnet_params.get("depth_slope", 0.0)
        depth_bias = regnet_params.get("depth_bias", 2.0)
        min_stage_depth = regnet_params.get("min_stage_depth", 1)
        max_stage_depth = regnet_params.get("max_stage_depth", 10)

        # Generate widths: w_i = initial_width * (width_slope ** i)
        widths = []
        for i in range(num_stages):
            width = initial_width * (width_slope**i)
            width = max(8, int(round(width / 8)) * 8)  # Round to multiple of 8
            widths.append(width)

        # Generate depths: d_i = depth_bias + depth_slope * i
        depths = []
        for i in range(num_stages):
            depth = depth_bias + depth_slope * i
            depth = max(min_stage_depth, min(max_stage_depth, int(round(depth))))
            depths.append(depth)

        return {
            "num_stages": num_stages,
            "widths": widths,
            "depths": depths,
            "total_blocks": sum(depths),
        }

    def _build_simple_resnet(self, stage_config: Dict[str, Any], arch_params: Dict[str, Any]) -> nn.Module:
        """Build a simple ResNet-like model"""

        # Extract stem parameters
        stem_params = arch_params.get("resnet_stem", {})
        stem_channels = stem_params.get("out_channels", 64)

        # Extract block parameters
        block_type = arch_params.get("block_type", "basic")

        return SimpleResNet(
            input_channels=self.input_channels,
            stem_channels=stem_channels,
            stage_widths=stage_config["widths"],
            stage_depths=stage_config["depths"],
            num_classes=self.num_classes,
            block_type=block_type,
        )

    def _build_simple_convnext(self, stage_config: Dict[str, Any], arch_params: Dict[str, Any]) -> nn.Module:
        """Build a simple ConvNeXt-like model"""

        # Extract stem parameters
        stem_params = arch_params.get("convnext_stem", {})
        stem_channels = stem_params.get("out_channels", 96)

        # Extract block parameters
        convnext_block = arch_params.get("convnext_block", {})
        expansion_ratio = convnext_block.get("expansion_ratio", 4)

        return SimpleConvNeXt(
            input_channels=self.input_channels,
            stem_channels=stem_channels,
            stage_widths=stage_config["widths"],
            stage_depths=stage_config["depths"],
            num_classes=self.num_classes,
            expansion_ratio=expansion_ratio,
        )

    def get_model_info(self, model: nn.Module, stage_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get model information"""

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "architecture_type": model.__class__.__name__,
        }

        if stage_config:
            info.update(
                {
                    "num_stages": stage_config["num_stages"],
                    "stage_widths": stage_config["widths"],
                    "stage_depths": stage_config["depths"],
                    "total_blocks": stage_config["total_blocks"],
                }
            )

        return info


class SimpleResNet(nn.Module):
    """Simple ResNet implementation for demonstration"""

    def __init__(
        self,
        input_channels: int,
        stem_channels: int,
        stage_widths: list,
        stage_depths: list,
        num_classes: int,
        block_type: str = "basic",
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build stages
        self.stages = nn.ModuleList()
        in_channels = stem_channels

        for i, (width, depth) in enumerate(zip(stage_widths, stage_depths)):
            stride = 2 if i > 0 else 1  # First stage no downsampling
            stage = self._make_stage(in_channels, width, depth, stride, block_type)
            self.stages.append(stage)
            in_channels = width

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_widths[-1], num_classes)

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int, block_type: str
    ) -> nn.Module:
        """Create a stage with multiple blocks"""

        blocks = []

        # First block (with potential downsampling)
        blocks.append(SimpleResNetBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(num_blocks - 1):
            blocks.append(SimpleResNetBlock(out_channels, out_channels, 1))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SimpleResNetBlock(nn.Module):
    """Simple ResNet block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class SimpleConvNeXt(nn.Module):
    """Simple ConvNeXt implementation for demonstration"""

    def __init__(
        self,
        input_channels: int,
        stem_channels: int,
        stage_widths: list,
        stage_depths: list,
        num_classes: int,
        expansion_ratio: int = 4,
    ):
        super().__init__()

        # Patchify stem (like ConvNeXt)
        self.stem = nn.Conv2d(input_channels, stem_channels, kernel_size=4, stride=4)

        # Build stages
        self.stages = nn.ModuleList()
        in_channels = stem_channels

        for i, (width, depth) in enumerate(zip(stage_widths, stage_depths)):
            downsample = i > 0  # Downsample after first stage
            stage = self._make_stage(in_channels, width, depth, downsample, expansion_ratio)
            self.stages.append(stage)
            in_channels = width

        self.norm = nn.LayerNorm(stage_widths[-1], eps=1e-6)
        self.head = nn.Linear(stage_widths[-1], num_classes)

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, downsample: bool, expansion_ratio: int
    ) -> nn.Module:
        """Create a stage with multiple blocks"""

        blocks = []

        # Downsampling layer if needed
        if downsample:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
            in_channels = out_channels
        elif in_channels != out_channels:
            # Channel adjustment without spatial downsampling
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            in_channels = out_channels

        # Add ConvNeXt blocks
        for _ in range(num_blocks):
            blocks.append(SimpleConvNeXtBlock(in_channels, expansion_ratio))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        for stage in self.stages:
            for block in stage:
                if isinstance(block, SimpleConvNeXtBlock):
                    x = block(x)
                else:
                    # Handle Conv2d layers (downsampling/channel adjustment)
                    x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                    x = block(x)
                    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x)
        x = x.mean([1, 2])  # Global average pooling
        x = self.head(x)

        return x


class SimpleConvNeXtBlock(nn.Module):
    """Simple ConvNeXt block"""

    def __init__(self, dim: int, expansion_ratio: int = 4):
        super().__init__()

        hidden_dim = expansion_ratio * dim

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)  # Pointwise/1x1 convs implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x: (N, H, W, C)
        input_x = x

        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = input_x + x  # Residual connection
        return x


def demo_model_generation():
    """Demonstrate model generation from sampled parameters"""

    print("🚀 RegNet Model Generation Demo")
    print("=" * 50)

    # Check if sample files exist
    import os

    sample_files = ["sample_resnet_verified.json", "sample_convnext_verified.json"]
    available_files = [f for f in sample_files if os.path.exists(f)]

    if not available_files:
        print("❌ No sample files found. Please run generate_multiple.py first.")
        return

    builder = SimpleRegNetBuilder(num_classes=1000, input_channels=3)

    for sample_file in available_files:
        print(f"\n📁 Processing {sample_file}...")

        try:
            # Load sample parameters
            with open(sample_file, "r") as f:
                sample_params = json.load(f)

            # Extract architecture info
            arch_params = sample_params["hierarchical"]["architectures"]
            arch_type = arch_params["architecture_type"]
            regnet_params = arch_params["regnet_rule"]

            print(f"🏗️  Architecture: {arch_type}")
            print(f"    RegNet stages: {regnet_params['num_stages']}")
            print(f"    Width slope: {regnet_params['width_slope']:.3f}")
            print(f"    Initial width: {regnet_params['initial_width']}")

            # Generate stage configuration
            stage_config = builder._generate_regnet_stages(regnet_params)
            print(f"    Generated widths: {stage_config['widths']}")
            print(f"    Generated depths: {stage_config['depths']}")
            print(f"    Total blocks: {stage_config['total_blocks']}")

            # Build model
            model = builder.build_model_from_sample(sample_params)
            info = builder.get_model_info(model, stage_config)

            print(f"✅ Model created successfully!")
            print(f"    Total parameters: {info['total_parameters']:,}")
            print(f"    Model size: {info['model_size_mb']:.2f} MB")

            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            print(f"    Output shape: {output.shape}")
            print(f"    Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        except Exception as e:
            print(f"❌ Error processing {sample_file}: {e}")

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    demo_model_generation()
