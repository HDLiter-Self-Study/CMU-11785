"""
Enumerations for the search space sampling system.

This module defines the core enumerations used throughout the sampling system
to ensure type safety and consistency.
"""

from enum import Enum


class GranularityLevel(Enum):
    """
    Enumeration of supported parameter granularity levels.

    The granularity levels define how parameters are applied across the architecture:
    - GLOBAL: Single parameter value used throughout the entire network
    - BLOCK_TYPE: Different parameter values for different block types (e.g., basic vs bottleneck)
    - STAGE: Different parameter values for each stage of the network
    - BLOCK_STAGE: Different parameter values for each (stage, block_type) combination
    - STEM: Special parameter value for the stem/input processing layer
    """

    GLOBAL = "global"
    BLOCK_TYPE = "block_type"
    STAGE = "stage"
    BLOCK_STAGE = "block_stage"
    STEM = "stem"


class ConfigClass(Enum):
    """
    Enumeration of configuration node classes.

    These classes define the taxonomy of configuration nodes:
    - STRATEGY: Top-level strategy configurations that group techniques
    - TECHNIQUE: Specific techniques or approaches within a strategy
    - INSTANCE: Concrete implementations or configurations within a technique
    - PARAM: Individual hyperparameters that get sampled
    """

    STRATEGY = "strategy"
    TECHNIQUE = "technique"
    INSTANCE = "instance"
    PARAM = "param"


def parse_config_class(config_class: str) -> str:
    for cls in ConfigClass:
        if cls.value == config_class:
            return cls.value
    return "unknown"


def parse_granularity_level(granularity_level: str) -> str:
    for level in GranularityLevel:
        if level.value == granularity_level:
            return level.value
    return "unknown"
