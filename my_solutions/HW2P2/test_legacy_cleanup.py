#!/usr/bin/env python
"""
Test script to verify legacy system cleanup and new architecture system
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_legacy_cleanup():
    """Test that legacy system has been properly cleaned up"""

    print("🧹 Testing Legacy System Cleanup")
    print("=" * 50)

    # Test 1: Check if network.py is removed
    network_path = "src/models/network.py"
    if not os.path.exists(network_path):
        print("✅ network.py successfully removed")
    else:
        print("❌ network.py still exists")
        return False

    # Test 2: Test new imports work
    try:
        from models import ArchitectureFactory, ConvolutionBlock

        print("✅ New imports work correctly")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test 3: Test architecture factory works
    try:
        factory = ArchitectureFactory()
        print(f"✅ ArchitectureFactory created: {list(factory.builders.keys())}")
    except Exception as e:
        print(f"❌ ArchitectureFactory error: {e}")
        return False

    # Test 4: Test ConvolutionBlock is available from new location
    try:
        conv_block = ConvolutionBlock(3, 64, 3, 1, 1)
        test_input = torch.randn(1, 3, 32, 32)
        output = conv_block(test_input)
        print(f"✅ ConvolutionBlock works: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"❌ ConvolutionBlock error: {e}")
        return False

    # Test 5: Test model creation with new system
    try:
        config = {
            "architecture": "resnet",
            "depth": 18,
            "block_type": "basic",
            "width_multiplier": 1.0,
            "num_classes": 1000,
            "use_se": True,
            "se_reduction": 16,
        }

        model = factory.create_model(config)
        test_input = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(test_input)

        if isinstance(output, dict) and "out" in output:
            print(f"✅ Model creation and forward pass work: {output['out'].shape}")
        else:
            print(f"❌ Unexpected output format: {type(output)}")
            return False

    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

    # Test 6: Test ConvNeXt still works after import change
    try:
        convnext_config = {"architecture": "convnext", "variant": "tiny", "num_classes": 1000}

        convnext_model = factory.create_model(convnext_config)
        with torch.no_grad():
            convnext_output = convnext_model(test_input)

        if isinstance(convnext_output, dict) and "out" in convnext_output:
            print(f"✅ ConvNeXt works after import change: {convnext_output['out'].shape}")
        else:
            print(f"❌ ConvNeXt output issue: {type(convnext_output)}")
            return False

    except Exception as e:
        print(f"❌ ConvNeXt error: {e}")
        return False

    print("\n🎉 All legacy cleanup tests passed!")
    print("💡 Benefits achieved:")
    print("   • Removed legacy Network class")
    print("   • Unified architecture system")
    print("   • Moved ConvolutionBlock to common location")
    print("   • Updated train.py and evaluate.py to use modern system")
    print("   • Maintained backward compatibility for ConvNeXt")

    return True


if __name__ == "__main__":
    success = test_legacy_cleanup()
    if not success:
        print("\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1)
    else:
        print("\n✨ Legacy system cleanup completed successfully!")
