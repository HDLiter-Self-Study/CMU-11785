"""
Configuration module for HW2P2
"""

from .config_manager import ConfigManager, get_config
from .compat import config  # Backward compatibility

__all__ = ["ConfigManager", "get_config", "config"]
