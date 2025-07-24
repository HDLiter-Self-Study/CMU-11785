"""
Model checkpointing utilities
"""

import torch
import os


def save_model(model, optimizer, scheduler, metrics, epoch, path):
    """Save model checkpoint"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metrics,
        "epoch": epoch,
    }

    # Add scheduler if provided
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_model(model, optimizer=None, scheduler=None, path="./checkpoint.pth"):
    """Load model checkpoint"""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = None

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        scheduler = None

    epoch = checkpoint["epoch"]
    metrics = checkpoint["metric"]

    return model, optimizer, scheduler, epoch, metrics
