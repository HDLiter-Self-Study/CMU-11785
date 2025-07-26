"""
Configuration module for HW2P2
"""

from .config_manager import ConfigManager, get_config, get_config_with_hyperparams
from .compat import config  # Backward compatibility

__all__ = ["ConfigManager", "get_config", "get_config_with_hyperparams", "config"]
