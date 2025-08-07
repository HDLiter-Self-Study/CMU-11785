"""
Handler for different parameter granularity levels.

This module manages the sampling of parameters at different granularity levels,
providing specialized handling for stage, block-stage, and block-type parameters.
It handles the complex logic of expanding single parameter configurations into
multiple architecture-specific parameters.

Granularity Levels:
- Stage: One parameter per stage (e.g., stage_1_of_4, stage_2_of_4, ...)
- Block-stage: One parameter per (stage, block_type) combination
- Block-type: One parameter per unique block type (e.g., basic, bottleneck)
"""

from typing import Dict, Any, List, Set, Optional
import optuna
from omegaconf import DictConfig

from .parameter_naming import ParameterNaming
from .enums import ConfigClass


class GranularityHandler:
    """
    Handler for different parameter granularity levels.

    This class manages the sampling of parameters at different granularity levels:
    - Stage: One parameter per stage
    - Block-stage: One parameter per (stage, block_type) combination
    - Block-type: One parameter per unique block type
    """

    def __init__(self, sampler, silent: bool = False):
        """
        Initialize granularity handler.

        Args:
            sampler: Reference to the main SearchSpaceSampler instance
            silent: If True, suppress all log output.
        """
        self.sampler = sampler
        self.naming = ParameterNaming()
        self.silent = silent

    def _log(self, message: str):
        """Prints a log message if the silent flag is not set."""
        if not self.silent:
            print(message)

    def sample_stage_params(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Sample stage-level parameters by expanding into multiple stage-specific parameters.

        Args:
            trial: Optuna trial object
            param_config: Parameter configuration from YAML
            base_param_name: Base parameter name (e.g., 'activation')
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix (e.g., 'resnet')

        Returns:
            Dictionary of stage-specific parameters {param_name: value}
        """
        num_stages = self.sampler.num_stages
        self._log(f"      📊 [STAGE_HANDLER] Sampling stage params for '{base_param_name}'")
        self._log(f"      📊 [STAGE_HANDLER] Architecture: {arch_prefix}, Stages: {num_stages}")
        stage_params = {}

        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1  # 1-based indexing for human readability
            stage_param_name = self.naming.build_stage_param_name(
                arch_prefix, base_param_name, stage_number, num_stages
            )

            stage_value = self.sampler._sample_single_param(trial, param_config, stage_param_name, sampled_params)
            stage_params[stage_param_name] = stage_value
            self._log(f"      📋 [STAGE_HANDLER] Stage {stage_idx}: {stage_param_name} = {stage_value}")

        self._log(f"      ✅ [STAGE_HANDLER] Final stage params: {stage_params}")
        return stage_params

    def sample_block_stage_params(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Sample block-stage-level parameters by expanding into stage and block-type specific parameters.

        This creates one parameter for each (stage, block_type) combination, allowing
        fine-grained control over parameters at the intersection of stage and block type.

        Args:
            trial: Optuna trial object
            param_config: Parameter configuration from YAML
            base_param_name: Base parameter name
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix

        Returns:
            Dictionary of block-stage-specific parameters {param_name: value}
        """
        num_stages = self.sampler.num_stages
        self._log(f"      📊 [BLOCK_STAGE_HANDLER] Sampling block-stage params for '{base_param_name}'")
        self._log(f"      📊 [BLOCK_STAGE_HANDLER] Architecture: {arch_prefix}, Stages: {num_stages}")
        block_stage_params = {}

        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            block_type = self._get_stage_block_type(stage_idx, num_stages, sampled_params, arch_prefix)
            self._log(f"      📋 [BLOCK_STAGE_HANDLER] Stage {stage_idx}: block_type = {block_type}")

            block_stage_param_name = self.naming.build_block_stage_param_name(
                arch_prefix, base_param_name, stage_number, num_stages, block_type
            )

            stage_value = self.sampler._sample_single_param(trial, param_config, block_stage_param_name, sampled_params)
            block_stage_params[block_stage_param_name] = stage_value
            self._log(f"      📋 [BLOCK_STAGE_HANDLER] Stage {stage_idx}: {block_stage_param_name} = {stage_value}")

        self._log(f"      ✅ [BLOCK_STAGE_HANDLER] Final block-stage params: {block_stage_params}")
        return block_stage_params

    def sample_block_type_params(
        self,
        trial: optuna.Trial,
        param_config: DictConfig,
        base_param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Sample block-type-level parameters by creating one parameter per unique block type.

        This creates one parameter for each unique block type found in the configuration,
        allowing different parameter values for different block types (e.g., basic vs bottleneck).

        Args:
            trial: Optuna trial object
            param_config: Parameter configuration from YAML
            base_param_name: Base parameter name
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix

        Returns:
            Dictionary of block-type-specific parameters {param_name: value}
        """
        unique_block_types = self._collect_unique_block_types(sampled_params, arch_prefix)
        self._log(f"      📊 [BLOCK_TYPE_HANDLER] Sampling block-type params for '{base_param_name}'")
        self._log(
            f"      📊 [BLOCK_TYPE_HANDLER] Architecture: {arch_prefix}, Unique block types: {unique_block_types}"
        )

        if not unique_block_types:
            raise ValueError(f"Cannot determine unique block types for architecture '{arch_prefix}'")

        block_type_params = {}
        for block_type in unique_block_types:
            param_name = self.naming.build_block_type_param_name(arch_prefix, base_param_name, block_type)
            value = self.sampler._sample_single_param(trial, param_config, param_name, sampled_params)
            block_type_params[param_name] = value
            self._log(f"      📋 [BLOCK_TYPE_HANDLER] Block type '{block_type}': {param_name} = {value}")

        self._log(f"      ✅ [BLOCK_TYPE_HANDLER] Final block-type params: {block_type_params}")
        return block_type_params

    def build_stage_list_for_block_type(
        self,
        param_name: str,
        sampled_params: Dict[str, Any],
        arch_prefix: str,
        block_type_params: Dict[str, Any],
    ) -> List[Any]:
        """
        Build a stage list for block-type granularity parameters, using the sampled_params
        as the single source of truth for block type information.

        Args:
            param_name: Parameter name (e.g., 'activation', 'normalization')
            sampled_params: The flat dictionary of already sampled parameters.
            arch_prefix: Architecture prefix
            block_type_params: Block-type parameter dictionary from the current sampling.

        Returns:
            List of parameter values for each stage
        """
        num_stages = self.sampler.num_stages
        stage_list = []

        # Determine if block types are defined globally or per-stage from sampled_params
        global_block_type = sampled_params.get(self.naming.build_param_name(arch_prefix, "global_block_type_selection"))

        if global_block_type:
            # Global block type: use the same value for all stages
            block_type_param_key = self.naming.build_block_type_param_name(arch_prefix, param_name, global_block_type)
            stage_value = block_type_params.get(block_type_param_key)
            return [stage_value] * num_stages

        # Stage-specific block types
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = self.naming.build_stage_param_name(
                arch_prefix, "stage_block_type_selection", stage_number, num_stages
            )
            block_type_for_stage = sampled_params.get(stage_block_type_key)

            if block_type_for_stage:
                block_type_param_key = self.naming.build_block_type_param_name(
                    arch_prefix, param_name, block_type_for_stage
                )
                stage_value = block_type_params.get(block_type_param_key)
                stage_list.append(stage_value)
            else:
                # This case should ideally not be reached if dependency order is correct
                stage_list.append(None)

        return stage_list

    def _get_stage_block_type(
        self, stage_idx: int, num_stages: int, sampled_params: Dict[str, Any], arch_prefix: str = ""
    ) -> str:
        """
        Get the block type for a specific stage.

        Args:
            stage_idx: Stage index (0-based)
            num_stages: Total number of stages
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix

        Returns:
            Block type for the specified stage

        Raises:
            ValueError: If block type cannot be determined
        """
        stage_number = stage_idx + 1

        # Try stage-specific block type first
        stage_block_type_key = self.naming.build_stage_param_name(
            arch_prefix, "stage_block_type_selection", stage_number, num_stages
        )
        stage_block_type = sampled_params.get(stage_block_type_key)
        if stage_block_type is not None:
            return stage_block_type

        # Fall back to global block type
        global_block_type_key = self.naming.build_param_name(arch_prefix, "global_block_type_selection")
        global_block_type = sampled_params.get(global_block_type_key)
        if global_block_type is not None:
            return global_block_type

        raise ValueError(
            f"Cannot determine block_type for stage {stage_idx}. "
            f"Checked keys: [{stage_block_type_key}, {global_block_type_key}]"
        )

    def _collect_unique_block_types(self, sampled_params: Dict[str, Any], arch_prefix: str) -> Set[str]:
        """
        Collect all unique block types from sampled parameters.

        Args:
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix

        Returns:
            Set of unique block types
        """
        unique_block_types = set()
        num_stages = self.sampler.num_stages

        # Check global block type
        global_block_type_key = self.naming.build_param_name(arch_prefix, "global_block_type_selection")
        if global_block_type_key in sampled_params:
            unique_block_types.add(sampled_params[global_block_type_key])
            return unique_block_types

        # Check stage-specific block types
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = self.naming.build_stage_param_name(
                arch_prefix, "stage_block_type_selection", stage_number, num_stages
            )
            if stage_block_type_key in sampled_params:
                unique_block_types.add(sampled_params[stage_block_type_key])

        if not unique_block_types:
            raise ValueError(
                f"Cannot collect unique block types for architecture '{arch_prefix}'. "
                "Ensure that 'global_block_type_selection' or 'stage_block_type_selection' "
                "is sampled before dependent parameters."
            )

        return unique_block_types

    def _find_param_config(self, param_name: str) -> Optional[DictConfig]:
        """
        Recursively find parameter configuration by name.

        Args:
            param_name: Name of parameter to find

        Returns:
            Parameter configuration if found, None otherwise
        """

        def search_recursive(node: DictConfig) -> Optional[DictConfig]:
            for key, child in node.items():
                if not isinstance(child, DictConfig):
                    continue

                if child.get("class") == ConfigClass.PARAM.value:
                    child_param_name = child.get("param_name", key)
                    if child_param_name == param_name:
                        return child

                # Recursive search
                result = search_recursive(child)
                if result is not None:
                    return result

            return None

        return search_recursive(self.sampler.search_spaces.architectures)

    def _ensure_stage_block_type_sampled(
        self, sampled_params: Dict[str, Any], arch_prefix: str, num_stages: int
    ) -> None:
        """
        Ensure stage_block_type parameters are sampled for block_type granularity.

        Args:
            sampled_params: Already sampled parameters
            arch_prefix: Architecture prefix
            num_stages: Number of stages
        """
        # Check if all stages have block_type sampled
        all_sampled = True
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = self.naming.build_stage_param_name(
                arch_prefix, "stage_block_type", stage_number, num_stages
            )
            if stage_block_type_key not in sampled_params:
                all_sampled = False
                break

        if all_sampled:
            return

        # Find stage_block_type configuration
        stage_block_type_config = self._find_param_config("stage_block_type")
        if stage_block_type_config is None:
            return  # Cannot sample without configuration

        # Sample missing stage block types
        # Note: This is a simplified approach - in a real implementation,
        # we would need access to the trial object to properly sample
        # For now, we'll use default values based on architecture
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_param_name = self.naming.build_stage_param_name(
                arch_prefix, "stage_block_type", stage_number, num_stages
            )
            if stage_param_name not in sampled_params:

                raise ValueError(
                    f"Cannot determine block type for stage {stage_number} in architecture '{arch_prefix}'"
                )
