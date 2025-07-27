# Models Architecture Documentation

## Overview

The models module has been refactored into a modular, hierarchical structure that separates different architectures and building blocks into organized files. This design follows the same pattern as the configuration system.

## Directory Structure

```
src/models/
├── __init__.py
├── architecture_factory.py       # Main factory for creating models
├── parameter_calculator.py       # Parameter calculation utilities
├── architectures/               # Different neural network architectures
│   ├── __init__.py
│   ├── base.py                  # Base abstract class for all architectures
│   ├── resnet.py               # Dynamic ResNet implementation with SE support
│   └── convnext.py             # Dynamic ConvNeXt implementation
└── common_blocks/              # Reusable building blocks
    ├── __init__.py
    ├── attention/              # Attention mechanisms
    │   └── se_module.py        # Squeeze-and-Excitation module
    └── convolution_block.py    # Basic convolution block
```

## Usage

### Creating a Model

```python
from src.models.architecture_factory import ArchitectureFactory

# Create factory
factory = ArchitectureFactory()

# Create ResNet model
config = {
    "architecture": "resnet",
    "resnet_depth": 50,
    "num_classes": 8631,
    "activation": "relu",
    "normalization": "batch_norm"
}
model = factory.create_model(config)

# Create ResNet with SE modules
config = {
    "architecture": "resnet",
    "depth": 50,
    "use_se": True,
    "se_reduction": 16,
    "se_activation": "swish",
    "num_classes": 8631
}
model = factory.create_model(config)
```

### Adding New Architectures

1. Create a new file in `architectures/` directory
2. Inherit from `BaseArchitecture` class
3. Implement the required methods
4. Add import to `architectures/__init__.py`
5. Register in `ArchitectureFactory`

Example:
```python
# architectures/my_new_arch.py
from .base import BaseArchitecture

class MyNewArchitecture(BaseArchitecture):
    def __init__(self, config):
        super().__init__(config)
        # Implementation here
    
    def forward(self, x):
        # Must return dict with 'feats', 'all_feats', 'out'
        return {"feats": features, "all_feats": all_features, "out": output}
```

### Adding New Blocks

1. Create a new file in `blocks/` directory
2. Implement your block as a `nn.Module`
3. Add import to `blocks/__init__.py`
4. Use in your architectures

## Architecture Specifications

### ResNet
- Supports depths: 18, 34, 50, 101, 152, or custom
- Configurable width multiplier
- Support for different block types (basic/bottleneck)
- Optional SE modules with configurable reduction and activation
- Configurable activation and normalization
- Advanced residual connection configurations (projection type, scaling, dropout)

### ConvNeXt
- Supports variants: tiny, small, base
- Simplified implementation using ConvolutionBlock
- Configurable depths and dimensions
- Optional SE module integration

## Configuration Options

### Common Options
- `num_classes`: Number of output classes
- `activation`: relu, gelu, swish, leaky_relu
- `normalization`: batch_norm, group_norm, layer_norm
- `dropout_rate`: Dropout rate (0.0 = no dropout)

### ResNet Specific
- `resnet_depth`: 18, 34, 50, 101, 152
- `width_multiplier`: Scale factor for channels
- `block_type`: "basic" or "bottleneck"
- `stem_channels`: Number of channels in stem
- `use_se`: Enable SE modules

### ConvNeXt Specific
- `convnext_variant`: "tiny", "small", "base"

## Base Architecture Interface

All architectures inherit from `BaseArchitecture` and must implement:

```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Returns:
        Dict containing:
        - 'feats': Final feature representation
        - 'all_feats': List of intermediate features  
        - 'out': Classification output
    """
```

This ensures consistent output format for both classification and verification tasks.
