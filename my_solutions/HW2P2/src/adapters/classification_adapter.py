"""
Classification Task Adapter
Encapsulates classification-specific logic
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna

from .base_adapter import TaskAdapter

from models.architecture_factory import ArchitectureFactory
from data import get_classification_dataloaders


class ClassificationAdapter(TaskAdapter):
    """
    Task adapter for image classification

    Handles classification-specific:
    - Model creation with classification heads
    - Classification data loading
    - Cross-entropy loss computation
    - Accuracy metric evaluation
    - Classification-specific search space
    """

    def __init__(self, config: Any = None):
        """
        Initialize classification adapter

        Args:
            config: Classification configuration object (optional for testing)
        """
        super().__init__(config or {})
        # Create arch_factory only if needed
        try:
            self.arch_factory = ArchitectureFactory()
        except Exception:
            self.arch_factory = None  # For testing without dependencies

    @property
    def task_name(self) -> str:
        """Return task name"""
        return "classification"

    @property
    def primary_metric(self) -> str:
        """Return primary optimization metric"""
        return "accuracy"

    def create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """
        Create classification model using existing architecture factory

        Args:
            model_config: Model configuration dictionary

        Returns:
            Classification model
        """
        # Ensure model has classification configuration
        classification_config = self.prepare_model_config(model_config, {"task": "classification"})

        # Use existing architecture factory
        model = self.arch_factory.create_model(classification_config)
        return model

    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Create classification data loaders

        Args:
            config: Data configuration dictionary

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Use existing data loading function
        train_dataloader, val_dataloader = get_classification_dataloaders(self.config)
        return train_dataloader, val_dataloader

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for classification

        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Class labels [batch_size]

        Returns:
            Cross-entropy loss
        """
        return F.cross_entropy(outputs, targets)

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics

        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Class labels [batch_size]

        Returns:
            Dictionary with accuracy metrics
        """
        with torch.no_grad():
            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total

            # Calculate top-5 accuracy if num_classes > 5
            top5_accuracy = 0.0
            if outputs.size(1) > 5:
                _, top5_pred = torch.topk(outputs, 5, dim=1)
                top5_correct = sum([targets[i] in top5_pred[i] for i in range(len(targets))])
                top5_accuracy = top5_correct / total

            return {"accuracy": accuracy, "top5_accuracy": top5_accuracy, "correct": correct, "total": total}

    def create_task_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create classification-specific search space
        Currently returns empty dict as classification parameters
        are handled by the base optimizer architecture parameters

        Args:
            trial: Optuna trial object

        Returns:
            Empty dictionary (classification uses base architecture search space)
        """
        # Classification task uses the existing architecture search space
        # No additional task-specific parameters needed currently
        return {}

    def get_wandb_tags(self) -> list:
        """
        Get WandB tags for classification experiments

        Returns:
            List of tags for WandB logging
        """
        return ["classification", "image_classification", "optuna", "nas"]

    def get_wandb_config_update(self, model_config: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get WandB configuration updates specific to classification

        Args:
            model_config: Model configuration
            metrics: Latest metrics

        Returns:
            Dictionary of WandB config updates
        """
        return {
            "task": self.task_name,
            "num_classes": model_config.get("num_classes", "unknown"),
            "current_accuracy": metrics.get("accuracy", 0.0),
            "current_top5_accuracy": metrics.get("top5_accuracy", 0.0),
        }
