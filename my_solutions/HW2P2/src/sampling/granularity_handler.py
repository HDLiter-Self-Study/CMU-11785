"""
Granularity handler.

Expands single PARAM configs into multiple architecture-specific parameters:
- Stage: one value per stage (stage_1_of_N, ...)
- Block-type: one value per unique block type (basic, bottleneck, ...)
- Block-stage: one value per (stage, block_type) pair

Relies on `sampler.num_stages` and block-type selections already available
in the current trial context (via sampler.sampled params).
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING, Set
import re
import optuna
from omegaconf import DictConfig

from .parameter_naming import ParameterNaming
from .enums import ConfigClass
import inspect
import optuna
from omegaconf import DictConfig, OmegaConf, ListConfig

from config.config_manager import get_config
from .parameter_naming import ParameterNaming


if TYPE_CHECKING:
    from .sampler import SearchSpaceSampler


class GranularityHandler:
    """
    Handler for different parameter granularity levels.

    This class manages the sampling of parameters at different granularity levels:
    - Stage: One parameter per stage
    - Block-stage: One parameter per (stage, block_type) combination
    - Block-type: One parameter per unique block type
    """

    def __init__(self, sampler: "SearchSpaceSampler", silent: bool = False):
        """
        Initialize granularity handler.

        Args:
            sampler: Reference to the main SearchSpaceSampler instance
            silent: If True, suppress all log output.
        """
        # GranularityHandler needs access to the sampler's trial-scoped context
        self.sampler = sampler
        self.naming = sampler.naming
        self.evaluator = sampler.evaluator
        self.silent = silent

    # -------------------- Build per-unit evaluation context (no variable parsing required) --------------------
    def _build_unit_eval_ctx(
        self,
        param_config: Dict[str, Any],
        base_ctx: Dict[str, Any],
        granularity: str,
        arch_prefix: str,
        sampled_params: Dict[str, Any],
        stage_number: Optional[int] = None,
        block_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an evaluation context for a single unit by scanning sampled_params.

        Priority order for resolving base parameter names (e.g., 'normalization'):
        - block_stage > stage > block_type > global
        The function derives values from naming conventions without hardcoding
        variable lists or re-parsing condition expressions.
        """
        eval_ctx: Dict[str, Any] = dict(base_ctx)
        if not isinstance(arch_prefix, str) or not arch_prefix:
            return eval_ctx
        prefix = f"{arch_prefix}_"
        num_stages = self.sampler.num_stages

        # Priority cache: base_name -> priority value
        # Priority depends on current granularity:
        # - block_stage: block_stage(3) > stage(2) > block_type(1) > global(0)
        # - stage: stage(2) > block_type(1) > global(0)
        # - block_type: block_type(2) > stage(1, only stages matching this block_type) > global(0)
        resolved: Dict[str, Any] = {}
        priorities: Dict[str, int] = {}

        for full_key, val in sampled_params.items():
            if not isinstance(full_key, str) or not full_key.startswith(prefix):
                continue
            suffix = full_key[len(prefix) :]

            # 1) Block-stage match: {base}_stage_{i}_of_{N}_{bt}
            m_bs = re.match(r"^(?P<base>.+)_stage_(?P<stage>\d+)_of_(?P<total>\d+)_(?P<bt>.+)$", suffix)
            if m_bs and stage_number is not None and block_type is not None:
                i = int(m_bs.group("stage"))
                total = int(m_bs.group("total"))
                bt = m_bs.group("bt")
                if i == stage_number and total == num_stages and bt == block_type:
                    base = m_bs.group("base")
                    prio = 3 if granularity == "block_stage" else 2
                    if priorities.get(base, -1) < prio:
                        resolved[base] = val
                        priorities[base] = prio
                    continue

            # 2) Stage match: {base}_stage_{i}_of_{N}
            m_s = re.match(r"^(?P<base>.+)_stage_(?P<stage>\d+)_of_(?P<total>\d+)$", suffix)
            if m_s and stage_number is not None:
                i = int(m_s.group("stage"))
                total = int(m_s.group("total"))
                if i == stage_number and total == num_stages:
                    base = m_s.group("base")
                    # When building context for block_type, only use the stage value if
                    # the current stage's block_type equals the current block_type.
                    if granularity == "block_type" and block_type is not None:
                        bt_key = self.naming.build_stage_param_name(arch_prefix, "block_type", stage_number, num_stages)
                        if sampled_params.get(bt_key) != block_type:
                            pass
                        else:
                            prio = 1
                            if priorities.get(base, -1) < prio:
                                resolved[base] = val
                                priorities[base] = prio
                    else:
                        prio = 2
                        if priorities.get(base, -1) < prio:
                            resolved[base] = val
                            priorities[base] = prio
                    continue

            # 3) Block-type match: {base}_{bt}
            if block_type is not None and suffix.endswith(f"_{block_type}") and ("_stage_" not in suffix):
                base = suffix[: -len(block_type) - 1]
                prio = 1 if granularity == "stage" else 2
                if priorities.get(base, -1) < prio:
                    resolved[base] = val
                    priorities[base] = prio
                continue

            # 4) Global match: {base}
            if ("_stage_" not in suffix) and (block_type is None or not suffix.endswith(f"_{block_type}")):
                base = suffix
                if priorities.get(base, -1) < 0:
                    resolved[base] = val
                    priorities[base] = 0

        # Merge into eval_ctx
        eval_ctx.update(resolved)
        return eval_ctx

    def _should_sample(self, param_config: Dict[str, Any], eval_ctx: Dict[str, Any]) -> bool:
        if hasattr(param_config, "condition"):
            try:
                return self.sampler.evaluator.evaluate_condition(param_config.condition, eval_ctx)
            except Exception:
                return False
        return True

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

            base_ctx = self.sampler._get_eval_context(sampled_params)
            eval_ctx = self._build_unit_eval_ctx(
                param_config,
                base_ctx,
                granularity="stage",
                arch_prefix=arch_prefix,
                sampled_params=sampled_params,
                stage_number=stage_number,
            )
            if not self._should_sample(param_config, eval_ctx):
                continue

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
            block_type = self._get_block_type_for_stage(stage_idx, sampled_params, arch_prefix)
            self._log(f"      📋 [BLOCK_STAGE_HANDLER] Stage {stage_idx}: block_type = {block_type}")

            block_stage_param_name = self.naming.build_block_stage_param_name(
                arch_prefix, base_param_name, stage_number, num_stages, block_type
            )

            base_ctx = self.sampler._get_eval_context(sampled_params)
            eval_ctx = self._build_unit_eval_ctx(
                param_config,
                base_ctx,
                granularity="block_stage",
                arch_prefix=arch_prefix,
                sampled_params=sampled_params,
                stage_number=stage_number,
                block_type=block_type,
            )
            if not self._should_sample(param_config, eval_ctx):
                continue

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
            base_ctx = self.sampler._get_eval_context(sampled_params)
            eval_ctx = self._build_unit_eval_ctx(
                param_config,
                base_ctx,
                granularity="block_type",
                arch_prefix=arch_prefix,
                sampled_params=sampled_params,
                block_type=block_type,
            )
            if not self._should_sample(param_config, eval_ctx):
                continue

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

        # Determine if block types are defined globally or per-stage from sampled_params (new unified naming)
        global_block_type = sampled_params.get(self.naming.build_param_name(arch_prefix, "block_type"))

        if global_block_type:
            # Global block type: use the same value for all stages
            block_type_param_key = self.naming.build_block_type_param_name(arch_prefix, param_name, global_block_type)
            stage_value = block_type_params.get(block_type_param_key)
            return [stage_value] * num_stages

        # Stage-specific block types
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = self.naming.build_stage_param_name(
                arch_prefix, "block_type", stage_number, num_stages
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

    def _get_block_type_for_stage(self, stage_idx: int, sampled: Dict[str, Any], arch_prefix: str) -> str:
        """Helper to find the block_type for a given stage."""
        num_stages = self.sampler.num_stages
        stage_num = stage_idx + 1

        # 1. Check for stage-specific flat key (e.g., resnet_block_type_stage_1_of_4)
        stage_key = self.naming.build_stage_param_name(arch_prefix, "block_type", stage_num, num_stages)
        if stage_key in sampled:
            return sampled[stage_key]

        # 2. Check for global flat key (e.g., resnet_block_type)
        global_key = self.naming.build_param_name(arch_prefix, "block_type")
        if global_key in sampled:
            return sampled[global_key]

        # 3. Fallback to raw 'block_type' key which might hold hierarchical info
        raw_key = "block_type"
        if raw_key in sampled:
            val = sampled[raw_key]
            # It could be an expanded list from 'stage' or 'all_stage' granularity
            if isinstance(val, list):
                if stage_idx < len(val):
                    item = val[stage_idx]
                    # The list can contain dicts {'block_type': '...'} or plain strings
                    return item.get("block_type") if isinstance(item, dict) else item
            # It could be a single string from 'global'/'all_stage' before instance expansion
            elif isinstance(val, str):
                return val

        raise ValueError(
            f"Cannot determine block_type for stage {stage_idx}. "
            f"Checked keys: [{stage_key}, {global_key}] and raw '{raw_key}' in sampled data."
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

        # Check global block type (new unified naming)
        global_block_type_key = self.naming.build_param_name(arch_prefix, "block_type")
        if global_block_type_key in sampled_params:
            unique_block_types.add(sampled_params[global_block_type_key])
            return unique_block_types

        # Check stage-specific block types (new unified naming)
        for stage_idx in range(num_stages):
            stage_number = stage_idx + 1
            stage_block_type_key = self.naming.build_stage_param_name(
                arch_prefix, "block_type", stage_number, num_stages
            )
            if stage_block_type_key in sampled_params:
                unique_block_types.add(sampled_params[stage_block_type_key])

        if not unique_block_types:
            raise ValueError(
                f"Cannot collect unique block types for architecture '{arch_prefix}'. "
                "Ensure that 'block_type' (global or stage) is sampled before dependent parameters."
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
