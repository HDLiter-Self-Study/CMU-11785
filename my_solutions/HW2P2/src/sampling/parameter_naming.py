"""
Parameter naming convention utilities.

This module provides utilities for generating consistent parameter names
across different granularity levels and architecture types. The naming
convention ensures unique parameter names for Optuna tracking while
maintaining readability and structure.

Parameter Naming Convention:
- Global parameters: {architecture}_{param_name}
- Stage parameters: {architecture}_{param_name}_stage_{N}_of_{total}
- Block-stage parameters: {architecture}_{param_name}_stage_{N}_of_{total}_{block_type}
- Block-type parameters: {architecture}_{param_name}_{block_type}
- Stem parameters: {architecture}_stem_{param_name}

Example parameter names:
- resnet_activation_granularity
- resnet_stage_block_type_stage_1_of_4
- resnet_block_type_activation_basic
- convnext_stem_normalization
"""

from dataclasses import dataclass


@dataclass
class ParameterNaming:
    """
    Parameter naming convention utilities.

    This class provides static methods for generating consistent parameter names
    across different granularity levels and architecture types.
    """

    @staticmethod
    def build_param_name(arch_prefix: str, base_name: str, granularity_info: str = "") -> str:
        """
        Build parameter name following the convention: {arch}_{param}_{granularity_info}

        Args:
            arch_prefix: Architecture prefix (e.g., 'resnet', 'convnext')
            base_name: Base parameter name (e.g., 'activation', 'normalization')
            granularity_info: Additional granularity information (e.g., 'stage_1_of_4')

        Returns:
            Formatted parameter name
        """
        parts = [arch_prefix, base_name]
        if granularity_info:
            parts.append(granularity_info)
        return "_".join(filter(None, parts))

    @staticmethod
    def build_stage_param_name(arch_prefix: str, base_name: str, stage_num: int, total_stages: int) -> str:
        """
        Build stage-specific parameter name.

        Args:
            arch_prefix: Architecture prefix
            base_name: Base parameter name
            stage_num: Stage number (1-based)
            total_stages: Total number of stages

        Returns:
            Stage-specific parameter name
        """
        granularity_info = f"stage_{stage_num}_of_{total_stages}"
        return ParameterNaming.build_param_name(arch_prefix, base_name, granularity_info)

    @staticmethod
    def build_block_stage_param_name(
        arch_prefix: str, base_name: str, stage_num: int, total_stages: int, block_type: str
    ) -> str:
        """
        Build block-stage-specific parameter name.

        Args:
            arch_prefix: Architecture prefix
            base_name: Base parameter name
            stage_num: Stage number (1-based)
            total_stages: Total number of stages
            block_type: Block type (e.g., 'basic', 'bottleneck')

        Returns:
            Block-stage-specific parameter name
        """
        granularity_info = f"stage_{stage_num}_of_{total_stages}_{block_type}"
        return ParameterNaming.build_param_name(arch_prefix, base_name, granularity_info)

    @staticmethod
    def build_block_type_param_name(arch_prefix: str, base_name: str, block_type: str) -> str:
        """
        Build block-type-specific parameter name.

        Args:
            arch_prefix: Architecture prefix
            base_name: Base parameter name
            block_type: Block type

        Returns:
            Block-type-specific parameter name
        """
        return ParameterNaming.build_param_name(arch_prefix, f"{base_name}_{block_type}")

    @staticmethod
    def build_stem_param_name(arch_prefix: str, base_name: str) -> str:
        """
        Build stem-specific parameter name.

        Args:
            arch_prefix: Architecture prefix
            base_name: Base parameter name

        Returns:
            Stem-specific parameter name
        """
        return ParameterNaming.build_param_name(arch_prefix, f"stem_{base_name}")
