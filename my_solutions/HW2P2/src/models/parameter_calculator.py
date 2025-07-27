"""
Parameter Calcula            "resnet": {
                "conv_ratio": 0.85,  # ~85% in convolutions
                "bn_ratio": 0.05,  # ~5% in batch norm
                "linear_ratio": 0.10,  # ~10% in final linear layer
            },
            "convnext": {"conv_ratio": 0.75, "norm_ratio": 0.15, "linear_ratio": 0.10},  # LayerNorm has more parametersmating model parameters before actual construction
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ParameterCalculator:
    """
    Calculator for estimating neural network parameters
    """

    def __init__(self):
        # Parameter estimation formulas for different components
        self.conv_params = lambda in_c, out_c, k: in_c * out_c * k * k
        self.linear_params = lambda in_f, out_f: in_f * out_f
        self.bn_params = lambda channels: channels * 2  # weight + bias

        # Typical parameter ratios for different architectures
        self.arch_ratios = {
            "resnet": {
                "conv_ratio": 0.85,  # ~85% parameters in conv layers
                "bn_ratio": 0.05,  # ~5% in batch norm
                "linear_ratio": 0.10,  # ~10% in final linear layer
            },
            "convnext": {"conv_ratio": 0.75, "norm_ratio": 0.15, "linear_ratio": 0.10},  # LayerNorm has more parameters
        }

    def estimate_params(self, config: Dict[str, Any]) -> int:
        """
        Estimate model parameters based on configuration
        """
        arch_type = config["architecture"]

        if arch_type == "resnet":
            return self._estimate_resnet_params(config)
        elif arch_type == "convnext":
            return self._estimate_convnext_params(config)
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

    def _estimate_resnet_params(self, config: Dict[str, Any]) -> int:
        """Estimate ResNet parameters"""
        depth = config.get("resnet_depth", 50)
        width_mult = config.get("width_multiplier", 1.0)
        block_type = config.get("block_type", "bottleneck")
        stem_channels = config.get("stem_channels", 64)
        num_classes = config.get("num_classes", 1000)

        # Base parameter counts for different ResNet depths
        base_params = {18: 11_689_512, 34: 21_797_672, 50: 25_557_032, 101: 44_549_160, 152: 60_192_808}

        # Get base parameters or interpolate for custom depths
        if depth in base_params:
            estimated_params = base_params[depth]
        else:
            # Linear interpolation based on depth
            if depth < 50:
                ratio = depth / 50
                estimated_params = int(base_params[50] * ratio)
            else:
                ratio = depth / 152
                estimated_params = int(base_params[152] * ratio)

        # Apply width multiplier (quadratic effect for conv layers)
        width_effect = width_mult**2
        conv_params = estimated_params * self.arch_ratios["resnet"]["conv_ratio"]
        other_params = estimated_params * (1 - self.arch_ratios["resnet"]["conv_ratio"])

        estimated_params = int(conv_params * width_effect + other_params * width_mult)

        # Adjust for stem channels
        stem_ratio = stem_channels / 64
        estimated_params = int(estimated_params * (0.9 + 0.1 * stem_ratio))

        # Adjust for number of classes
        class_ratio = num_classes / 1000
        final_layer_params = estimated_params * self.arch_ratios["resnet"]["linear_ratio"]
        estimated_params = int(estimated_params - final_layer_params + final_layer_params * class_ratio)

        return estimated_params

    def _estimate_convnext_params(self, config: Dict[str, Any]) -> int:
        """Estimate ConvNeXt parameters"""
        variant = config.get("convnext_variant", "tiny")
        num_classes = config.get("num_classes", 1000)

        # Base parameter counts for ConvNeXt variants
        base_params = {
            "tiny": 28_589_128,
            "small": 50_223_688,
            "base": 88_591_464,
        }

        estimated_params = base_params.get(variant, base_params["tiny"])

        # Adjust for number of classes
        class_ratio = num_classes / 1000
        final_layer_params = estimated_params * self.arch_ratios["convnext"]["linear_ratio"]
        estimated_params = int(estimated_params - final_layer_params + final_layer_params * class_ratio)

        return estimated_params

    def count_exact_params(self, model: nn.Module) -> int:
        """Count exact parameters of a constructed model"""
        return sum(p.numel() for p in model.parameters())

    def get_param_breakdown(self, model: nn.Module) -> Dict[str, int]:
        """Get detailed parameter breakdown by layer type"""
        breakdown = {"conv": 0, "linear": 0, "norm": 0, "other": 0}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                breakdown["conv"] += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                breakdown["linear"] += sum(p.numel() for p in module.parameters())
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                breakdown["norm"] += sum(p.numel() for p in module.parameters())
            elif len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                # Leaf module with parameters
                breakdown["other"] += sum(p.numel() for p in module.parameters())

        return breakdown

    def validate_estimation_accuracy(self, config: Dict[str, Any], actual_model: nn.Module) -> Dict[str, Any]:
        """Validate estimation accuracy against actual model"""
        estimated = self.estimate_params(config)
        actual = self.count_exact_params(actual_model)

        error = abs(estimated - actual)
        error_percentage = (error / actual) * 100

        return {
            "estimated": estimated,
            "actual": actual,
            "error": error,
            "error_percentage": error_percentage,
            "breakdown": self.get_param_breakdown(actual_model),
        }


def test_parameter_estimation():
    """Test parameter estimation accuracy"""
    from .architecture_factory import ArchitectureFactory

    factory = ArchitectureFactory()
    calculator = ParameterCalculator()

    # Test configurations
    test_configs = [
        {
            "architecture": "resnet",
            "resnet_depth": 50,
            "width_multiplier": 1.0,
            "block_type": "bottleneck",
            "stem_channels": 64,
            "num_classes": 8631,
            "activation": "relu",
            "normalization": "batch_norm",
            "dropout_rate": 0.0,
            "use_se": False,
        },
        {
            "architecture": "resnet",
            "resnet_depth": 18,
            "width_multiplier": 1.5,
            "block_type": "basic",
            "stem_channels": 64,
            "num_classes": 8631,
            "activation": "relu",
            "normalization": "batch_norm",
            "dropout_rate": 0.1,
            "use_se": True,
        },
    ]

    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config['architecture']} depth={config.get('resnet_depth', 'N/A')}")

        # Create model
        model = factory.create_model(config)

        # Validate estimation
        validation = calculator.validate_estimation_accuracy(config, model)

        print(f"Estimated: {validation['estimated']:,} parameters")
        print(f"Actual: {validation['actual']:,} parameters")
        print(f"Error: {validation['error']:,} ({validation['error_percentage']:.2f}%)")
        print(f"Breakdown: {validation['breakdown']}")


if __name__ == "__main__":
    test_parameter_estimation()
