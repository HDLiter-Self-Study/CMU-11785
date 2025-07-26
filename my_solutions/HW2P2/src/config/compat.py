"""
Backward compatibility layer for existing config usage
"""

from omegaconf import DictConfig
from .config_manager import get_config


class ConfigDict(dict):
    """
    Dictionary-like wrapper for OmegaConf config that maintains backward compatibility
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        # Convert nested configs to regular dict for backward compatibility
        super().__init__(self._flatten_config(cfg))

    def _flatten_config(self, cfg: DictConfig, prefix: str = "") -> dict:
        """
        Flatten hierarchical config to a flat dictionary for backward compatibility
        Priority: training config values take precedence over hyperparameter definitions
        """
        result = {}

        for key, value in cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, DictConfig):
                # For nested configs, flatten them
                nested_result = self._flatten_config(value, full_key)

                # Special handling for hyperparameters to avoid overriding training values
                if key == "hyperparameters":
                    # Don't add hyperparameter definitions to flat namespace
                    # to avoid conflicts with actual training values
                    for nested_key, nested_value in nested_result.items():
                        # Only add if it doesn't conflict with existing training values
                        hp_key = nested_key.split(".")[-1]  # Get the last part (e.g., "batch_size")
                        if hp_key not in result:
                            result[f"hyperparams.{hp_key}"] = nested_value
                else:
                    result.update(nested_result)

                # Also keep the nested structure as a whole
                result[key] = dict(value)
            else:
                result[key] = value

        return result

    def __getitem__(self, key):
        # Try to get from flattened dict first
        if key in dict.keys(self):
            return super().__getitem__(key)

        # Try to get from original config using dot notation
        try:
            from omegaconf import OmegaConf

            return OmegaConf.select(self._cfg, key)
        except:
            raise KeyError(f"Key '{key}' not found in config")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# Create the backward-compatible config object
_config_dict = None


def _initialize_config():
    """Initialize the config dictionary"""
    global _config_dict
    if _config_dict is None:
        cfg = get_config()
        _config_dict = ConfigDict(cfg)
    return _config_dict


# This maintains backward compatibility with existing code
config = _initialize_config()
