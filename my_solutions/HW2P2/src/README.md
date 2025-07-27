# HW2P2: Image Recognition and Verification

This project contains a simplified Python deep learning implementation for face recognition and verification tasks.

## Project Structure

```
src/
├── __init__.py
├── config.py                  # Configuration settings (all hyperparameters)
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script for test dataset
├── training_functions.py      # Training and validation functions
├── optimizers/                # Optimizer modules (Optuna-based)
│   ├── __init__.py
│   ├── base_optimizer.py      # Base optimizer class
│   ├── classification_optimizer.py    # Classification-specific optimizer
│   └── verification_optimizer.py     # Verification-specific optimizer
├── adapters/                  # Task adapter modules (Adapter Pattern)
│   ├── __init__.py
│   ├── base_adapter.py        # Base adapter interface
│   ├── classification_adapter.py     # Classification task adapter
│   └── verification_adapter.py      # Verification task adapter
├── data/                      # Data handling modules
│   ├── __init__.py
│   ├── datasets.py            # Dataset classes
│   └── dataloaders.py         # Data loading utilities
├── models/                    # Model definitions
│   ├── __init__.py
│   ├── architecture_factory.py  # Factory for creating models
│   ├── architectures/         # Modern neural network architectures
│   └── common_blocks/         # Reusable building blocks
└── utils/                     # Utility functions
    ├── __init__.py
    ├── metrics.py              # Evaluation metrics
    └── checkpoint.py           # Model checkpointing

```

## Requirements

```bash
torch
torchvision
numpy
pandas
scikit-learn
scipy
tqdm
Pillow
wandb (optional)
pytorch_metric_learning (optional)
```

## Usage

### Training

**Simple training (recommended):**
```bash
python src/train.py
```

All hyperparameters are controlled by `src/config.py`. Edit the config file to:
- Change batch size, learning rate, epochs
- Enable/disable WandB logging
- Resume from checkpoint
- Set data paths

### Configuration

Edit `src/config.py` to customize:

```python
config = {
    # Training hyperparameters
    "batch_size": 256,
    "lr": 0.005,
    "epochs": 50,
    
    # Training options
    "resume_from": "./checkpoints/last.pth",  # Resume training
    
    # WandB logging
    "use_wandb": True,
    "wandb_online": True,
}
```

### Evaluation

After training, evaluate your model on the test dataset:

```bash
# Use best verification model (default)
python src/evaluate.py

# Specify custom model path and output file
python src/evaluate.py --model_path ./checkpoints/best_cls.pth --output_file my_submission.csv
```

## Key Features

- **Simplified Structure**: Single training script with config-based control
- **No Command-line Arguments**: All settings in config.py
- **Automatic Checkpointing**: Saves best models for both tasks
- **Resume Training**: Set `resume_from` in config to continue training
- **Optional WandB**: Enable/disable logging via config
- **Mixed Precision**: Built-in for faster training

## Scripts Overview

- **train.py**: Main training script (use this for training)
- **evaluate.py**: Evaluation script for generating test predictions
- **config.py**: All hyperparameters and settings
- **training_functions.py**: Core training/validation functions
- **main.py**: Legacy script (for backward compatibility)
