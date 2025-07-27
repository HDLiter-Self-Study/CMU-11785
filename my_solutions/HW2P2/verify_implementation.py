#!/usr/bin/env python
"""
Verification script for the hollowed-out deep learning framework
Run this script to check if your implementations are correct
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_se_module():
    """Test SEModule implementation"""
    print("🧪 Testing SEModule...")

    try:
        from models.common_blocks.attention.se_module import SEModule

        # Test basic functionality
        channels = 64
        se_module = SEModule(channels, reduction=16)

        # Test forward pass
        x = torch.randn(2, channels, 32, 32)
        output = se_module(x)

        assert output.shape == x.shape, f"Output shape mismatch: expected {x.shape}, got {output.shape}"

        # Test from_config
        config = {"se_reduction": 8, "se_activation": "relu"}
        se_from_config = SEModule.from_config(channels, config)
        output2 = se_from_config(x)
        assert output2.shape == x.shape, "from_config output shape mismatch"

        print("   ✅ SEModule tests passed")
        return True

    except Exception as e:
        print(f"   ❌ SEModule tests failed: {e}")
        return False


def test_convolution_block():
    """Test ConvolutionBlock implementation"""
    print("🧪 Testing ConvolutionBlock...")

    try:
        from models.common_blocks.convolution_block import ConvolutionBlock

        # Test basic functionality
        conv_block = ConvolutionBlock(64, 128, 3, 2, 1)
        x = torch.randn(2, 64, 32, 32)
        output = conv_block(x)

        expected_shape = (2, 128, 16, 16)  # stride=2 should halve spatial dimensions
        assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

        print("   ✅ ConvolutionBlock tests passed")
        return True

    except Exception as e:
        print(f"   ❌ ConvolutionBlock tests failed: {e}")
        return False


def test_basic_block():
    """Test BasicBlock implementation"""
    print("🧪 Testing BasicBlock...")

    try:
        from models.architectures.resnet.blocks.basic_block import BasicBlock

        # Test basic functionality
        config = {"use_se": False, "activation": "relu", "normalization": "batch_norm"}
        basic_block = BasicBlock(64, 64, 1, config)

        x = torch.randn(2, 64, 32, 32)
        output = basic_block(x)

        assert output.shape == x.shape, f"Output shape mismatch: expected {x.shape}, got {output.shape}"

        # Test with SE module
        config_se = {"use_se": True, "se_reduction": 16}
        basic_block_se = BasicBlock(64, 64, 1, config_se)
        output_se = basic_block_se(x)
        assert output_se.shape == x.shape, "SE BasicBlock output shape mismatch"

        print("   ✅ BasicBlock tests passed")
        return True

    except Exception as e:
        print(f"   ❌ BasicBlock tests failed: {e}")
        return False


def test_bottleneck_block():
    """Test BottleneckBlock implementation"""
    print("🧪 Testing BottleneckBlock...")

    try:
        from models.architectures.resnet.blocks.bottleneck_block import BottleneckBlock

        # Test basic functionality
        config = {"use_se": False, "activation": "relu", "normalization": "batch_norm"}
        bottleneck_block = BottleneckBlock(64, 64, 1, config)

        x = torch.randn(2, 64, 32, 32)
        output = bottleneck_block(x)

        expected_channels = 64 * 4  # expansion = 4
        expected_shape = (2, expected_channels, 32, 32)
        assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

        print("   ✅ BottleneckBlock tests passed")
        return True

    except Exception as e:
        print(f"   ❌ BottleneckBlock tests failed: {e}")
        return False


def test_convnext_block():
    """Test ConvNeXtBlock implementation"""
    print("🧪 Testing ConvNeXtBlock...")

    try:
        from models.architectures.convnext.blocks.convnext_block import ConvNeXtBlock

        # Test basic functionality
        dim = 96
        config = {"use_se": False}
        convnext_block = ConvNeXtBlock(dim, drop_path=0.0, config=config)

        x = torch.randn(2, dim, 32, 32)
        output = convnext_block(x)

        assert output.shape == x.shape, f"Output shape mismatch: expected {x.shape}, got {output.shape}"

        print("   ✅ ConvNeXtBlock tests passed")
        return True

    except Exception as e:
        print(f"   ❌ ConvNeXtBlock tests failed: {e}")
        return False


def test_resnet():
    """Test ResNet architecture"""
    print("🧪 Testing ResNet...")

    try:
        from models.architectures.resnet.resnet import ResNet

        # Test ResNet-18
        config = {
            "depth": 18,
            "num_classes": 1000,
            "use_se": False,
            "activation": "relu",
            "normalization": "batch_norm",
        }

        model = ResNet(config)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert "feats" in output, "Missing 'feats' in output"
        assert "all_feats" in output, "Missing 'all_feats' in output"
        assert "out" in output, "Missing 'out' in output"
        assert output["out"].shape == (2, 1000), f"Wrong output shape: {output['out'].shape}"

        print("   ✅ ResNet tests passed")
        return True

    except Exception as e:
        print(f"   ❌ ResNet tests failed: {e}")
        return False


def test_convnext():
    """Test ConvNeXt architecture"""
    print("🧪 Testing ConvNeXt...")

    try:
        from models.architectures.convnext.convnext import ConvNeXt

        # Test ConvNeXt-Tiny
        config = {"convnext_variant": "tiny", "num_classes": 1000, "drop_path_rate": 0.0}

        model = ConvNeXt(config)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert "feats" in output, "Missing 'feats' in output"
        assert "all_feats" in output, "Missing 'all_feats' in output"
        assert "out" in output, "Missing 'out' in output"
        assert output["out"].shape == (2, 1000), f"Wrong output shape: {output['out'].shape}"

        print("   ✅ ConvNeXt tests passed")
        return True

    except Exception as e:
        print(f"   ❌ ConvNeXt tests failed: {e}")
        return False


def test_architecture_factory():
    """Test ArchitectureFactory"""
    print("🧪 Testing ArchitectureFactory...")

    try:
        from models.architecture_factory import ArchitectureFactory

        factory = ArchitectureFactory()

        # Test ResNet creation
        resnet_config = {"architecture": "resnet", "depth": 18, "num_classes": 1000}

        resnet_model = factory.create_model(resnet_config)
        x = torch.randn(1, 3, 224, 224)
        resnet_output = resnet_model(x)
        assert resnet_output["out"].shape == (1, 1000), "ResNet factory test failed"

        # Test ConvNeXt creation
        convnext_config = {"architecture": "convnext", "convnext_variant": "tiny", "num_classes": 1000}

        convnext_model = factory.create_model(convnext_config)
        convnext_output = convnext_model(x)
        assert convnext_output["out"].shape == (1, 1000), "ConvNeXt factory test failed"

        print("   ✅ ArchitectureFactory tests passed")
        return True

    except Exception as e:
        print(f"   ❌ ArchitectureFactory tests failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Running Deep Learning Framework Verification Tests")
    print("=" * 60)

    test_functions = [
        test_se_module,
        test_convolution_block,
        test_basic_block,
        test_bottleneck_block,
        test_convnext_block,
        test_resnet,
        test_convnext,
        test_architecture_factory,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        if test_func():
            passed += 1
        print()  # Add spacing between tests

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your implementation is correct!")
        print("\n📚 Learning Summary:")
        print("   ✅ You've successfully implemented a modern deep learning framework")
        print("   ✅ You understand residual connections and attention mechanisms")
        print("   ✅ You grasp the factory pattern and modular design")
        print("   ✅ You can work with both classic (ResNet) and modern (ConvNeXt) architectures")
    else:
        print(f"⚠️  {total - passed} tests failed. Please review your implementations.")
        print("\n🔍 Debugging Tips:")
        print("   1. Check that all TODO items have been implemented")
        print("   2. Ensure tensor shapes are correct throughout the forward pass")
        print("   3. Verify that modules are properly initialized")
        print("   4. Make sure all imports are working correctly")


if __name__ == "__main__":
    main()
