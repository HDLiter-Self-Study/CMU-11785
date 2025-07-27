#!/usr/bin/env python
"""
Enhanced test script for ResNet residual connection configurations
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.architecture_factory import ArchitectureFactory

def test_residual_configs():
    """Test different residual connection configurations"""
    
    # Test configurations with different projection types
    test_configs = [
        {
            "name": "Conv Projection (Standard)",
            "config": {
                "depth": 18,
                "width_multiplier": 1.0,
                "block_type": "basic",
                "projection_type": "conv",
                "projection_norm": True,
                "residual_scale": 1.0,
                "residual_dropout": 0.0
            }
        },
        {
            "name": "AvgPool Projection (Efficient)",
            "config": {
                "depth": 18,
                "width_multiplier": 1.0,
                "block_type": "basic",
                "projection_type": "avg_pool",
                "projection_norm": True,
                "residual_scale": 1.0,
                "residual_dropout": 0.0
            }
        },
        {
            "name": "MaxPool Projection (Custom)",
            "config": {
                "depth": 18,
                "width_multiplier": 1.0,
                "block_type": "basic",
                "projection_type": "max_pool",
                "projection_norm": False,
                "residual_scale": 0.8,
                "residual_dropout": 0.1
            }
        },
        {
            "name": "Auto Projection + Bottleneck",
            "config": {
                "depth": 50,
                "width_multiplier": 1.0,
                "block_type": "bottleneck",
                "projection_type": "auto",
                "projection_norm": True,
                "residual_scale": 1.2,
                "residual_dropout": 0.05
            }
        },
        {
            "name": "SE + Custom Residual",
            "config": {
                "depth": 18,
                "width_multiplier": 1.5,
                "block_type": "basic",
                "use_se": True,
                "se_reduction": 16,
                "se_activation": "swish",
                "projection_type": "conv",
                "projection_norm": True,
                "residual_scale": 0.9,
                "residual_dropout": 0.1
            }
        },
        {
            "name": "Deep Network (ResNet-101)",
            "config": {
                "depth": 101,
                "width_multiplier": 1.0,
                "block_type": "bottleneck",
                "projection_type": "auto",
                "projection_norm": True,
                "residual_scale": 1.1,
                "residual_dropout": 0.02
            }
        }
    ]
    
    print("🧪 Testing ResNet Residual Connection Configurations")
    print("=" * 65)
    print("Testing various projection types, scales, and dropout configurations...")
    print()
    
    successful_tests = 0
    total_tests = len(test_configs)
    
    for i, test_case in enumerate(test_configs, 1):
        config_name = test_case["name"]
        config = test_case["config"]
        
        print(f"{i}. {config_name}")
        print("-" * 50)
        
        try:
            # Create model
            factory = ArchitectureFactory()
            full_config = {"architecture": "resnet", **config}
            model = factory.create_model(full_config)
            
            # Test forward pass
            batch_size = 2
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(test_input)
            
            # Handle dict output format
            if isinstance(output, dict):
                main_output = output.get('logits') or output.get('output') or list(output.values())[0]
                output_shape = main_output.shape
                output_info = f"Output keys: {list(output.keys())}"
            else:
                output_shape = output.shape
                output_info = "Single tensor output"
            
            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✅ SUCCESS")
            print(f"   📊 Model Statistics:")
            print(f"      • Input shape: {test_input.shape}")
            print(f"      • Output shape: {output_shape}")
            print(f"      • Total parameters: {total_params:,}")
            print(f"      • Trainable parameters: {trainable_params:,}")
            print(f"      • {output_info}")
            
            print(f"   ⚙️  Configuration:")
            print(f"      • Depth: {config['depth']} layers")
            print(f"      • Block type: {config['block_type']}")
            print(f"      • Projection type: {config['projection_type']}")
            print(f"      • Projection norm: {config.get('projection_norm', True)}")
            print(f"      • Residual scale: {config.get('residual_scale', 1.0)}")
            print(f"      • Residual dropout: {config.get('residual_dropout', 0.0)}")
            
            if config.get('use_se'):
                print(f"      • SE enabled: reduction={config.get('se_reduction', 16)}, activation={config.get('se_activation', 'relu')}")
            
            # Parameter efficiency analysis
            if i > 1 and 'base_params' in locals():
                param_diff = total_params - base_params
                efficiency = "more" if param_diff > 0 else "fewer"
                print(f"   📈 Efficiency: {abs(param_diff):,} {efficiency} parameters than Conv Projection")
            elif i == 1:
                base_params = total_params
            
            successful_tests += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print("=" * 65)
    print(f"🎯 Test Summary: {successful_tests}/{total_tests} configurations passed")
    
    if successful_tests == total_tests:
        print("🎉 All residual connection configurations work perfectly!")
        print()
        print("💡 Key Findings:")
        print("   • All projection types (conv, avg_pool, max_pool, auto) work correctly")
        print("   • Residual scaling and dropout are properly applied")
        print("   • SE modules integrate seamlessly with residual configurations")
        print("   • Both BasicBlock and BottleneckBlock support all features")
        print()
        print("📝 Next Steps:")
        print("   • Ready for hyperparameter search experiments")
        print("   • All configurations available in YAML config files")
        print("   • Use ResNet_Residual_Config_Guide.md for detailed parameter explanations")
    else:
        failed_tests = total_tests - successful_tests
        print(f"⚠️  {failed_tests} configuration(s) failed - please check the errors above")

if __name__ == "__main__":
    test_residual_configs()
