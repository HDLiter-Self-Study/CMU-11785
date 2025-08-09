"""
Enumerations and registries for the sampling/resolving system.

This module centralizes enums (granularity, config classes, pipeline modes/slots)
and name registries (optimizer/scheduler/etc.) to avoid scattered literals.
New developers should treat this file as the single source of truth for these
constants.
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
    # Fast-fail instead of returning ambiguous 'unknown'
    raise ValueError(f"Unknown config class: {config_class}")


def parse_granularity_level(granularity_level: str) -> str:
    for level in GranularityLevel:
        if level.value == granularity_level:
            return level.value
    raise ValueError(f"Unknown granularity level: {granularity_level}")


class SelectionMode(Enum):
    """Supported group selection/combination modes for pipelines."""

    SINGLE = "single"
    RANDOM_CHOICE = "random_choice"
    SUM = "sum"
    WEIGHTED_SUM = "weighted_sum"
