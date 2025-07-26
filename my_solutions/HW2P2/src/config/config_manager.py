"""
Configuration manager using OmegaConf/Hydra for hierarchical config management
"""

import os
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


class ConfigManager:
    """
    Configuration manager that handles hierarchical configuration loading
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager

        Args:
            config_dir: Path to the configuration directory
        """
        if config_dir is None:
            config_dir = os.path.dirname(__file__)

        self.config_dir = os.path.abspath(config_dir)
        self._config = None
        self._initialized = False

    def load_config(self, config_name: str = "main", overrides: Optional[list] = None) -> DictConfig:
        """
        Load configuration using Hydra

        Args:
            config_name: Name of the main config file (without .yaml extension)
            overrides: List of parameter overrides

        Returns:
            Loaded configuration
        """
        if overrides is None:
            overrides = []

        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize Hydra with config directory
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)

        self._config = cfg
        self._initialized = True
        return cfg

    def get_config(self) -> DictConfig:
        """
        Get the current configuration

        Returns:
            Current configuration
        """
        if not self._initialized or self._config is None:
            return self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration values

        Args:
            updates: Dictionary of updates to apply
        """
        if self._config is None:
            self.load_config()

        # Set struct to False to allow new keys
        OmegaConf.set_struct(self._config, False)
        self._config = OmegaConf.merge(self._config, updates)
        # Re-enable struct mode
        OmegaConf.set_struct(self._config, True)

    def save_config(self, path: str) -> None:
        """
        Save current configuration to file

        Args:
            path: Path to save the configuration
        """
        if self._config is not None:
            OmegaConf.save(self._config, path)

    def merge_hyperparameters(self, hyperparams: Dict[str, Any]) -> DictConfig:
        """
        Merge hyperparameters into training configuration

        This method is specifically designed for hyperparameter search scenarios
        where we need to override training parameters with trial-specific values.

        Args:
            hyperparams: Dictionary of hyperparameters to merge into training config

        Returns:
            Updated configuration with merged hyperparameters
        """
        if self._config is None:
            self.load_config()

        # Create a copy to avoid modifying the original config
        merged_config = OmegaConf.create(self._config)

        # Disable struct mode for updates
        OmegaConf.set_struct(merged_config, False)

        # Map common hyperparameters to their config locations
        hyperparam_mapping = {
            "learning_rate": "training.lr",
            "batch_size": "training.batch_size",
            "weight_decay": "training.weight_decay",
            "optimizer": "training.optimizer",
            "scheduler": "training.scheduler",
        }

        # Apply hyperparameter overrides
        for param_name, param_value in hyperparams.items():
            if param_name in hyperparam_mapping:
                config_path = hyperparam_mapping[param_name]
                # Use OmegaConf.set_item or direct assignment based on version
                try:
                    OmegaConf.set(merged_config, config_path, param_value)
                except AttributeError:
                    # Fallback for older OmegaConf versions
                    keys = config_path.split(".")
                    current = merged_config
                    for key in keys[:-1]:
                        current = current[key]
                    current[keys[-1]] = param_value
            else:
                # For architecture-specific parameters, add to training section
                try:
                    OmegaConf.set(merged_config, f"training.{param_name}", param_value)
                except AttributeError:
                    # Fallback for older OmegaConf versions
                    merged_config.training[param_name] = param_value

        # Re-enable struct mode
        OmegaConf.set_struct(merged_config, True)

        return merged_config


# Global config manager instance
_config_manager = ConfigManager()


def get_config(config_name: str = "main", overrides: Optional[list] = None) -> DictConfig:
    """
    Get configuration using the global config manager

    Args:
        config_name: Name of the main config file
        overrides: List of parameter overrides

    Returns:
        Configuration object
    """
    return _config_manager.load_config(config_name, overrides)


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update global configuration

    Args:
        updates: Dictionary of updates to apply
    """
    _config_manager.update_config(updates)


def get_config_with_hyperparams(
    config_name: str = "main", overrides: Optional[list] = None, hyperparams: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Get configuration with hyperparameter overrides applied

    This is specifically designed for hyperparameter search scenarios.

    Args:
        config_name: Name of the main config file
        overrides: List of parameter overrides
        hyperparams: Dictionary of hyperparameters to merge

    Returns:
        Configuration object with hyperparameters applied
    """
    # Load base configuration
    cfg = _config_manager.load_config(config_name, overrides)

    # Apply hyperparameters if provided
    if hyperparams:
        cfg = _config_manager.merge_hyperparameters(hyperparams)

    return cfg
