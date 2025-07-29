"""
Utility functions for managing activation functions and 2D normalization layers in PyTorch models.

This module provides a unified interface for creating activation functions and normalization layers
with dynamic parameter discovery and validation. All normalization layers are specifically designed
for 2D feature maps with shape (B, C, H, W).

Key Features:
- Dynamic parameter discovery using introspection
- Automatic parameter validation and filtering
- Unified API for all activation and normalization functions
- Clear error messages for missing required parameters

Example Usage:
    >>> # Activation functions
    >>> relu = get_activation('relu')
    >>> leaky_relu = get_activation('leaky_relu', negative_slope=0.2)
    >>> threshold = get_activation('threshold', threshold=0.5, value=0.0)

    >>> # 2D Normalization layers
    >>> batch_norm = get_2d_normalization('batch_norm', 64)
    >>> group_norm = get_2d_normalization('group_norm', 64, num_groups=16)

    >>> # Parameter inspection
    >>> relu_params = get_activation_params('relu')
    >>> bn_params = get_normalization_params('batch_norm')
"""

import torch.nn as nn
import inspect
from typing import Dict, Any

# Global activation function mapping
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "logsigmoid": nn.LogSigmoid,
    "hardtanh": nn.Hardtanh,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "threshold": nn.Threshold,
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
    "hardsigmoid": nn.Hardsigmoid,
    "glu": nn.GLU,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "log_softmax": nn.LogSoftmax,
    "none": nn.Identity,
    "identity": nn.Identity,
}

# Global 2D normalization layer mapping
# Note: All normalization layers here are designed for 2D feature maps (B, C, H, W)
NORMALIZATION_2D_MAP = {
    "batch_norm": nn.BatchNorm2d,  # Normalize across batch dimension: μ,σ computed over (N,H,W)
    "instance_norm": nn.InstanceNorm2d,  # Normalize per instance: μ,σ computed over (H,W) for each sample
    "group_norm": nn.GroupNorm,  # Normalize per group: divide C into groups, μ,σ computed over (H,W) per group
    "layer_norm": nn.GroupNorm,  # Approximated by GroupNorm(1, C): μ,σ computed over (C,H,W) per sample
    # WARNING: This is NOT true LayerNorm! True LayerNorm for 2D should normalize over spatial dims only
    "none": nn.Identity,
    "identity": nn.Identity,
}


def _get_function_params(func, exclude_params=None):
    """Dynamically get function parameter signatures and default values"""
    if exclude_params is None:
        exclude_params = {"self"}

    sig = inspect.signature(func)
    params = {}

    for name, param in sig.parameters.items():
        if name in exclude_params:
            continue

        if param.default != inspect.Parameter.empty:
            params[name] = param.default
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # Handle **kwargs
            continue
        else:
            # Required parameter
            params[name] = None

    return params


def _create_activation_with_params(activation_class, provided_params):
    """Create activation function with provided parameters, automatically validating them"""
    # Get activation function parameter signature
    default_params = _get_function_params(activation_class.__init__, exclude_params={"self"})

    # Only use supported parameters
    filtered_params = {}
    for key, value in provided_params.items():
        if key in default_params:
            filtered_params[key] = value

    return activation_class(**filtered_params)


def _create_2d_norm_with_params(norm_class, num_features, provided_params):
    """Create normalization layer with provided parameters, automatically validating them"""
    # Get normalization layer parameter signature
    default_params = _get_function_params(norm_class.__init__, exclude_params={"self", "num_features", "num_channels"})

    # Only use supported parameters
    filtered_params = {}
    for key, value in provided_params.items():
        if key in default_params:
            filtered_params[key] = value

    # Handle special cases
    if norm_class == nn.GroupNorm:
        # GroupNorm requires num_groups parameter - user must provide it explicitly
        return norm_class(num_channels=num_features, **filtered_params)
    else:
        return norm_class(num_features, **filtered_params)


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation function by name with optional parameters.

    Args:
        name (str): Name of the activation function.
        **kwargs: Additional parameters for the activation function.

    Returns:
        nn.Module: The activation function module.

    Examples:
        >>> get_activation('relu')
        >>> get_activation('leaky_relu', negative_slope=0.1)
        >>> get_activation('gelu', approximate='tanh')
        >>> get_activation('threshold', threshold=0.5, value=0.0)

    Raises:
        ValueError: If activation function is unknown.
        TypeError: If required parameters are missing (e.g., threshold requires 'threshold' and 'value').
    """
    if name is None:
        name = "identity"  # Default to identity if no name provided

    name_lower = name.lower()
    if name_lower not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation function: {name}. Available: {list(ACTIVATION_MAP.keys())}")

    activation_class = ACTIVATION_MAP[name_lower]
    return _create_activation_with_params(activation_class, kwargs)


def get_2d_normalization(name: str, num_features: int, **kwargs) -> nn.Module:
    """Get 2D normalization layer by name, with optional parameters.

    All normalization layers are designed for 2D feature maps with shape (B, C, H, W).

    Args:
        name (str): Name of the normalization layer.
        num_features (int): Number of features/channels (C dimension). Required for all normalization layers.
        **kwargs: Additional parameters for the normalization layer.

    Returns:
        nn.Module: The 2D normalization layer module.

    Examples:
        >>> get_2d_normalization('batch_norm', 64)      # BatchNorm2d
        >>> get_2d_normalization('group_norm', 64, num_groups=16)  # GroupNorm - num_groups required
        >>> get_2d_normalization('instance_norm', 64, eps=1e-6)    # InstanceNorm2d
        >>> get_2d_normalization('none', 64)           # Identity layer (ignores num_features)

    Note:
        - 'group_norm' requires explicit num_groups parameter
        - 'layer_norm' is approximated using GroupNorm(1, num_features), which normalizes
          over all channels and spatial dimensions. This is NOT equivalent to true LayerNorm.
        - For true layer normalization in 2D, consider using instance_norm instead.
        - 'none' and 'identity' return Identity layer but still require num_features for API consistency
    """
    if name is None:
        name = "identity"  # Default to identity if no name provided

    name_lower = name.lower()
    if name_lower not in NORMALIZATION_2D_MAP:
        raise ValueError(f"Unknown 2D normalization: {name}. Available: {list(NORMALIZATION_2D_MAP.keys())}")

    norm_class = NORMALIZATION_2D_MAP[name_lower]

    # Handle none/identity cases - return Identity but require num_features for API consistency
    if name_lower in ["none", "identity"]:
        return nn.Identity()

    # Handle layer_norm approximation
    if name_lower == "layer_norm":
        return nn.GroupNorm(1, num_features, **kwargs)

    return _create_2d_norm_with_params(norm_class, num_features, kwargs)


def get_activation_params(name: str) -> Dict[str, Any]:
    """Get parameter info for the specified activation function

    Args:
        name (str): Activation function name

    Returns:
        Dict[str, Any]: Dictionary of parameter names and default values

    Examples:
        >>> get_activation_params('leaky_relu')
        {'negative_slope': 0.01, 'inplace': False}
        >>> get_activation_params('threshold')
        {'threshold': None, 'value': None, 'inplace': False}
    """
    name_lower = name.lower()
    if name_lower not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation function: {name}")

    return _get_function_params(ACTIVATION_MAP[name_lower].__init__, exclude_params={"self"})


def get_normalization_params(name: str) -> Dict[str, Any]:
    """Get parameter info for the specified 2D normalization layer

    Args:
        name (str): 2D normalization layer name

    Returns:
        Dict[str, Any]: Dictionary of parameter names and default values

    Examples:
        >>> get_normalization_params('batch_norm')
        {'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}
        >>> get_normalization_params('group_norm')
        {'num_groups': None, 'eps': 1e-05, 'affine': True}

    Note:
        All normalization layers are for 2D feature maps (B, C, H, W).
        'layer_norm' uses GroupNorm approximation - see get_2d_normalization() for details.
    """
    name_lower = name.lower()
    if name_lower not in NORMALIZATION_2D_MAP:
        raise ValueError(f"Unknown 2D normalization: {name}")

    # Get basic parameters
    params = _get_function_params(
        NORMALIZATION_2D_MAP[name_lower].__init__, exclude_params={"self", "num_features", "num_channels"}
    )

    # Add contextual information for special cases
    if name_lower == "group_norm":
        if "num_groups" in params and params["num_groups"] is None:
            params["num_groups"] = "Required: number of groups to divide channels into (must be provided explicitly)"
    elif name_lower == "layer_norm":
        params["num_groups"] = "Fixed to 1 (GroupNorm approximation of LayerNorm)"

    return params


def list_available_activations() -> list:
    """List all available activation functions"""
    return list(ACTIVATION_MAP.keys())


def list_available_normalizations() -> list:
    """List all available 2D normalization layers

    Returns:
        list: List of available 2D normalization layer names

    Note:
        All normalization layers are designed for 2D feature maps (B, C, H, W).
        'layer_norm' is approximated using GroupNorm - see documentation for details.
    """
    return list(NORMALIZATION_2D_MAP.keys())
