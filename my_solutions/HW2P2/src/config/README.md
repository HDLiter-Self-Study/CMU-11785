# Hierarchical Configuration System

This project now uses a hierarchical configuration system based on Hydra/OmegaConf for better modularity and flexibility.

## Directory Structure

```
src/config/
├── __init__.py                 # Main config module exports
├── config_manager.py           # Core configuration management
├── compat.py                   # Backward compatibility layer
└── configs/                    # Configuration files
    ├── main.yaml              # Main configuration
    ├── training/              # Training configurations
    │   ├── default.yaml
    │   ├── fast.yaml
    │   └── production.yaml
    ├── data/                  # Data configurations
    │   └── default.yaml
    ├── wandb/                 # WandB configurations
    │   ├── default.yaml
    │   ├── offline.yaml
    │   └── disabled.yaml
    └── optuna/                # Optuna configurations
        ├── default.yaml
        └── quick.yaml
```

## Usage Examples

### Basic Usage

```python
from config import get_config

# Load default configuration
cfg = get_config()
print(f"Batch size: {cfg.training.batch_size}")
print(f"Learning rate: {cfg.training.lr}")
```

### Using Configuration Variants

```python
# Use fast training configuration
cfg = get_config("main", overrides=["training=fast"])

# Use production training with offline wandb
cfg = get_config("main", overrides=["training=production", "wandb=offline"])

# Disable wandb and use quick optuna search
cfg = get_config("main", overrides=["wandb=disabled", "optuna=quick"])
```

### Runtime Parameter Overrides

```python
# Override specific parameters
cfg = get_config("main", overrides=[
    "training.batch_size=64",
    "training.lr=0.002",
])
```

### Command Line Usage

You can also override parameters from command line:

```bash
python train.py training.batch_size=128 wandb=offline
python train.py training=fast optuna=quick training.epochs=10
```

### Backward Compatibility

Existing code continues to work without changes:

```python
from config import config

# Old dictionary-style access still works
batch_size = config['batch_size']
lr = config['lr']
data_dir = config['data_dir']
```

## Configuration Categories

### Training (`training/`)
- `default.yaml`: Standard training parameters
- `fast.yaml`: Quick training for experiments
- `production.yaml`: Production-ready parameters

### Data (`data/`)
- `default.yaml`: Data paths and loading settings

### WandB (`wandb/`)
- `default.yaml`: Standard online logging
- `offline.yaml`: Offline logging
- `disabled.yaml`: Disabled logging

### Optuna (`optuna/`)
- `default.yaml`: Standard hyperparameter search
- `quick.yaml`: Quick search for testing

## Adding New Configurations

1. Create new YAML files in appropriate subdirectories
2. Use `# @package <category>` at the top of each file
3. Reference them in overrides: `category=new_config_name`

## Benefits

1. **Modularity**: Separate concerns into different configuration files
2. **Flexibility**: Easy to create variants for different experiments
3. **Composability**: Mix and match different configuration components
4. **Validation**: Built-in configuration validation
5. **IDE Support**: Better autocomplete and type checking
6. **Backward Compatibility**: Existing code continues to work
