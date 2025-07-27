#!/usr/bin/env python
"""
Test script to check ConvNeXt duplicate files and clean them up
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_convnext_duplicates():
    """Test ConvNeXt duplicate files and determine which version to keep"""

    print("🔍 检查ConvNeXt重复文件情况")
    print("=" * 50)

    # Check file existence
    simple_convnext = "src/models/architectures/convnext.py"
    full_convnext = "src/models/architectures/convnext/convnext.py"
    convnext_block = "src/models/architectures/convnext/blocks/convnext_block.py"

    print("文件存在情况:")
    print(f"📁 简化版ConvNeXt: {'✅' if os.path.exists(simple_convnext) else '❌'} {simple_convnext}")
    print(f"📁 完整版ConvNeXt: {'✅' if os.path.exists(full_convnext) else '❌'} {full_convnext}")
    print(f"📁 ConvNeXtBlock: {'✅' if os.path.exists(convnext_block) else '❌'} {convnext_block}")

    # Test current imports
    print("\n🧪 测试当前导入和模型创建:")
    try:
        from src.models.architecture_factory import ArchitectureFactory

        config = {"architecture": "convnext", "convnext_variant": "tiny", "num_classes": 1000}

        factory = ArchitectureFactory()
        model = factory.create_model(config)

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        print("✅ ConvNeXt模型创建成功")
        print(f"   输出形状: {output['out'].shape}")
        print(f"   参数总量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   输出键: {list(output.keys())}")

        # Check which blocks are being used
        convolution_blocks = []
        convnext_blocks = []

        for name, module in model.named_modules():
            if "ConvolutionBlock" in str(type(module)):
                convolution_blocks.append(name)
            elif "ConvNeXtBlock" in str(type(module)):
                convnext_blocks.append(name)

        print(f"\n📊 Block使用情况:")
        print(f"   ConvolutionBlock数量: {len(convolution_blocks)}")
        print(f"   ConvNeXtBlock数量: {len(convnext_blocks)}")

        if convnext_blocks:
            print("   🎯 使用的是完整版ConvNeXt (有ConvNeXtBlock)")
            current_version = "完整版"
        else:
            print("   🎯 使用的是简化版ConvNeXt (使用ConvolutionBlock)")
            current_version = "简化版"

    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        current_version = "未知"

    # Recommendation
    print(f"\n💡 建议:")
    if os.path.exists(full_convnext) and os.path.exists(convnext_block):
        print("   ✅ 检测到完整版ConvNeXt实现")
        print("   📝 建议删除简化版，使用完整版")

        if current_version == "简化版":
            print("   ⚠️  当前使用的是简化版，需要更新导入")
        elif current_version == "完整版":
            print("   ✅ 当前已使用完整版，可删除简化版")
    else:
        print("   ⚠️  完整版实现不完整，保留简化版")

    return {
        "simple_exists": os.path.exists(simple_convnext),
        "full_exists": os.path.exists(full_convnext),
        "block_exists": os.path.exists(convnext_block),
        "current_version": current_version,
    }


def cleanup_convnext_duplicates():
    """Clean up ConvNeXt duplicate files"""

    print("\n🧹 开始清理ConvNeXt重复文件")
    print("=" * 50)

    simple_convnext = "src/models/architectures/convnext.py"

    if os.path.exists(simple_convnext):
        try:
            os.remove(simple_convnext)
            print(f"✅ 删除简化版: {simple_convnext}")
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    else:
        print(f"ℹ️  简化版已不存在: {simple_convnext}")

    # Test after cleanup
    print("\n🧪 清理后测试:")
    try:
        # Force reload modules
        if "src.models.architectures" in sys.modules:
            del sys.modules["src.models.architectures"]
        if "src.models.architecture_factory" in sys.modules:
            del sys.modules["src.models.architecture_factory"]

        from src.models.architecture_factory import ArchitectureFactory

        config = {"architecture": "convnext", "convnext_variant": "tiny", "num_classes": 1000}

        factory = ArchitectureFactory()
        model = factory.create_model(config)

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        # Check blocks again
        convnext_blocks = [name for name, module in model.named_modules() if "ConvNeXtBlock" in str(type(module))]

        print("✅ 清理后模型创建成功")
        print(f"   ConvNeXtBlock数量: {len(convnext_blocks)}")

        if convnext_blocks:
            print("   🎉 现在使用完整版ConvNeXt!")
        else:
            print("   ⚠️  仍在使用简化版，可能需要手动更新导入")

        return True

    except Exception as e:
        print(f"❌ 清理后测试失败: {e}")
        return False


if __name__ == "__main__":
    # First check the situation
    result = test_convnext_duplicates()

    # If we have duplicates and full version is better, clean up
    if result["simple_exists"] and result["full_exists"] and result["block_exists"]:
        print("\n" + "=" * 50)
        user_input = input("是否要删除简化版ConvNeXt文件? (y/n): ")
        if user_input.lower() in ["y", "yes"]:
            cleanup_success = cleanup_convnext_duplicates()
            if cleanup_success:
                print("\n🎉 ConvNeXt重复文件清理完成!")
            else:
                print("\n❌ 清理过程中出现问题")
        else:
            print("\n⏭️  跳过清理")
    else:
        print("\n✅ 无需清理或条件不满足")
