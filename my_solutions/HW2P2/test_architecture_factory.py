#!/usr/bin/env python3
"""
Test script for the new modular architecture factory
"""

import sys

sys.path.append(".")

from src.models.architecture_factory import ArchitectureFactory
import torch


def test_architecture_factory():
    """Test the architecture factory with different configurations"""

    factory = ArchitectureFactory()

    # Test configurations
    test_configs = [
        {
            "name": "ResNet-18",
            "config": {
                "architecture": "resnet",
                "resnet_depth": 18,
                "num_classes": 10,
                "activation": "relu",
                "normalization": "batch_norm",
            },
        },
        {
            "name": "ResNet-50 with SE",
            "config": {
                "architecture": "resnet",
                "resnet_depth": 50,
                "use_se": True,
                "se_reduction": 16,
                "se_activation": "relu",
                "num_classes": 1000,
                "activation": "swish",
                "normalization": "batch_norm",
            },
        },
        {
            "name": "ResNet-34 with SE (GELU)",
            "config": {
                "architecture": "resnet",
                "resnet_depth": 34,
                "use_se": True,
                "se_reduction": 8,
                "se_activation": "gelu",
                "num_classes": 8631,
                "activation": "gelu",
                "normalization": "batch_norm",
            },
        },
        {
            "name": "ConvNeXt-Tiny",
            "config": {"architecture": "convnext", "convnext_variant": "tiny", "num_classes": 1000},
        },
        {
            "name": "ConvNeXt-Base with SE",
            "config": {
                "architecture": "convnext",
                "convnext_variant": "base",
                "use_se": True,
                "se_reduction": 4,
                "num_classes": 1000,
            },
        },
    ]

    print("🏗️  Testing Architecture Factory")
    print("=" * 50)

    for test_case in test_configs:
        name = test_case["name"]
        config = test_case["config"]

        try:
            # Create model
            model = factory.create_model(config)

            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)

            # Verify output format
            assert "feats" in output, "Missing 'feats' in output"
            assert "all_feats" in output, "Missing 'all_feats' in output"
            assert "out" in output, "Missing 'out' in output"

            print(f"✅ {name}: {type(model).__name__}")
            print(f"   - Output classes: {output['out'].shape[1]}")
            print(f"   - Feature dims: {output['feats'].shape[1]}")
            print(f"   - Intermediate features: {len(output['all_feats'])}")

        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)}")

        print()

    print("🎉 All tests completed!")


if __name__ == "__main__":
    test_architecture_factory()
