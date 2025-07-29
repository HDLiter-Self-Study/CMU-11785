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
