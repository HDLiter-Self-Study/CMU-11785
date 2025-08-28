"""
Base Task Adapter Interface
Abstract base class defining the interface for task-specific adapters
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna


class TaskAdapter(ABC):
    """
    Abstract base class for task adapters

    Task adapters encapsulate task-specific logic for:
    - Model creation and configuration
    - Data loading and preprocessing
    - Loss computation
    - Metric evaluation
    - Search space definition
    """

    def __init__(self, config: Any):
        """
        Initialize task adapter with configuration

        Args:
            config: Task-specific configuration object
        """
        self.config = config

    @abstractmethod
    def create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """
        Create task-specific model

        Args:
            model_config: Model configuration dictionary

        Returns:
            PyTorch model for the specific task
        """
        pass

    @abstractmethod
    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Create task-specific data loaders

        Args:
            config: Data configuration dictionary

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute task-specific loss

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute task-specific evaluation metrics

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def create_task_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Create task-specific search space parameters

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of task-specific parameters
        """
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return the name of this task"""
        pass

    @property
    @abstractmethod
    def primary_metric(self) -> str:
        """Return the primary metric name for optimization"""
        pass

    def get_model_save_path(self, trial_number: int, score: float) -> str:
        """
        Generate model save path for this task

        Args:
            trial_number: Optuna trial number
            score: Model score

        Returns:
            Path string for saving model
        """
        return f"./checkpoints/{self.task_name}_trial_{trial_number}_score_{score:.4f}.pth"

    def prepare_model_config(self, base_config: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with task-specific configuration

        Args:
            base_config: Base model configuration
            task_config: Task-specific configuration

        Returns:
            Merged configuration dictionary
        """
        merged_config = {**base_config, **task_config}
        merged_config["task"] = self.task_name
        return merged_config
