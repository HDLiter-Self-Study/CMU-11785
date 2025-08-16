"""
Configuration module for HW2P2
"""

from src.config.config_manager import ConfigManager, get_config
from src.config.compat import config  # Backward compatibility

__all__ = ["ConfigManager", "get_config", "config"]
